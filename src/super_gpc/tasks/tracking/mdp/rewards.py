"""SuperMimic Table A.2 tracking reward with BeyondMimic body-pose alignment.

Per-step weighted sum without a control-dt multiplier; maximum unpenalized
value 5. Used identically for simulator transitions, replay and imagination.
Body index 0 is the pelvis and index 1 the torso anchor (TRACKED_BODIES).
An optional BlinkMimic contact bonus is separate from the base reward and must
only be added by the imagination caller, before overall reward scaling.
"""

from __future__ import annotations

import torch

from super_gpc.tasks.tracking.mdp import kinematics as K

ANCHOR = 1
FEET = (5, 8)  # left/right_ankle_roll_link in canonical TRACKED_BODIES order
CONTACT_Z = 0.045  # metres, ankle-roll link origin (not sole or COM)
MAX_REWARD = 5.0
TERMS = (
  "body_pos",
  "body_ori",
  "body_lin_vel",
  "body_ang_vel",
  "anchor_pos",
  "anchor_ori",
  "action_rate",
  "joint_limit",
)


def align_reference(
  robot: K.BodyStates, ref: K.BodyStates, anchor: int = ANCHOR
) -> tuple[torch.Tensor, torch.Tensor]:
  """BeyondMimic alignment: robot anchor XY, reference anchor Z, yaw-only
  rotation of (robot anchor x inverse reference anchor). Defaults to torso;
  pass ``anchor=0`` for pelvis without changing encoder conventions.
  """
  ra_p, ra_q = robot.pos[..., anchor, :], robot.quat[..., anchor, :]
  fa_p, fa_q = ref.pos[..., anchor, :], ref.quat[..., anchor, :]
  delta_p = torch.cat((ra_p[..., :2], fa_p[..., 2:3]), -1)
  R = K.yaw_matrix_from(K.quat_to_matrix(K.quat_mul(ra_q, K.quat_conj(fa_q))))
  # Yaw rotation as a quaternion about z: (1 + cos, sin) is parallel to the
  # half-angle (cos, sin) pair.
  c, s = R[..., 0, 0], R[..., 1, 0]
  zero = torch.zeros_like(c)
  dq = K.normalize(torch.stack((1 + c + 1e-9, zero, zero, s), -1))
  pos = delta_p[..., None, :] + (ref.pos - fa_p[..., None, :]) @ R.transpose(-1, -2)
  quat = K.quat_mul(dq[..., None, :].expand_as(ref.quat), ref.quat)
  return pos, quat


def tracking_reward(
  robot: K.BodyStates,
  ref: K.BodyStates,
  action: torch.Tensor,
  prev_action: torch.Tensor,
  joint_pos: torch.Tensor,
  soft_lower: torch.Tensor,
  soft_upper: torch.Tensor,
  *,
  anchor: int = ANCHOR,
) -> dict[str, torch.Tensor]:
  """Returns unscaled weighted terms/errors; ``total`` sums base terms only.

  ``anchor=0`` selects pelvis for both alignment and global anchor rewards.
  The default remains torso. Contact is deliberately not included here.
  """
  pos, quat = align_reference(robot, ref, anchor=anchor)
  e_pos = ((pos - robot.pos) ** 2).sum(-1).mean(-1)
  e_ori = K.quat_angle_sq(robot.quat, quat).mean(-1)
  e_lin = ((ref.lin_vel - robot.lin_vel) ** 2).sum(-1).mean(-1)
  e_ang = ((ref.ang_vel - robot.ang_vel) ** 2).sum(-1).mean(-1)
  e_apos = ((ref.pos[..., anchor, :] - robot.pos[..., anchor, :]) ** 2).sum(-1)
  e_aori = K.quat_angle_sq(robot.quat[..., anchor, :], ref.quat[..., anchor, :])
  rate = ((action - prev_action) ** 2).sum(-1)
  limit = (torch.relu(joint_pos - soft_upper) + torch.relu(soft_lower - joint_pos)).sum(-1)
  out = {
    "body_pos": 1.0 * torch.exp(-e_pos / 0.3**2),
    "body_ori": 1.0 * torch.exp(-e_ori / 0.4**2),
    "body_lin_vel": 1.0 * torch.exp(-e_lin / 1.0**2),
    "body_ang_vel": 1.0 * torch.exp(-e_ang / 3.14**2),
    "anchor_pos": 0.5 * torch.exp(-e_apos / 0.3**2),
    "anchor_ori": 0.5 * torch.exp(-e_aori / 0.4**2),
    "action_rate": -0.1 * rate,
    "joint_limit": -2.0 * limit,
  }
  out["total"] = sum(out[k] for k in TERMS)
  out["err_body_pos_m"] = torch.sqrt(((pos - robot.pos) ** 2).sum(-1)).mean(-1)
  out["err_body_ori_rad"] = torch.sqrt(K.quat_angle_sq(robot.quat, quat)).mean(-1)
  out["err_body_lin_vel"] = torch.sqrt(((ref.lin_vel - robot.lin_vel) ** 2).sum(-1)).mean(-1)
  out["err_anchor_pos_m"] = torch.sqrt(e_apos)
  out["err_anchor_ori_rad"] = torch.sqrt(e_aori)
  return out


@torch.no_grad()
def calibrated_reference_contact(
  clip: K.BodyStates, margin: float = 0.015, vmax: float = 0.5, q: float = 0.05
) -> torch.Tensor:
  """BlinkMimic ``MotionLib.set_ref_contact`` for ONE complete reference clip.

  Input positions/velocities must be world-frame link origins in canonical
  tracked-body order, shaped (frames, bodies, 3), after motion resampling/FK.
  Returns float contacts (frames, 2), left then right. Each foot's floor is
  its own clip-height q-quantile; contact requires z < floor + margin AND
  planar speed < vmax. Never calibrate on minibatches, future windows or the
  concatenated corpus: preprocess clips separately, concatenate the results,
  and gather contacts alongside the matching reference frames in imagination.
  """
  if clip.pos.ndim != 3 or clip.pos.shape[0] == 0:
    raise ValueError("Contact calibration requires one nonempty complete clip")
  z = clip.pos[:, FEET, 2]
  speed = clip.lin_vel[:, FEET, :2].norm(dim=-1)
  floor = z.quantile(q, dim=0)
  return ((z < floor + margin) & (speed < vmax)).float()


def contact_bonus(
  robot: K.BodyStates, ref_contact: torch.Tensor, weight: float = 0.25
) -> torch.Tensor:
  """Optional imagination-only BlinkMimic contact matching/anti-slip bonus.

  Returns ``weight * mean_feet(1 - (c_sim-c_ref)^2
  + c_ref * exp(-||v_xy||^2 / 0.3^2))``, without dt or reward-scale factors.
  c_sim = sigmoid((0.045 - z) / 0.005), using world ankle-roll link origins.
  ``ref_contact`` has shape (..., 2), gathered from clip-calibrated contacts.
  No fixed-height reference fallback is used: calibration is explicit.
  Add this to tracking_reward()["total"] ONLY in imagination, then multiply
  the sum by cfg.reward_scale. Real collection/evaluation stay unchanged.
  """
  z = robot.pos[..., FEET, 2]
  c_sim = torch.sigmoid((CONTACT_Z - z) / 0.005)
  slip = robot.lin_vel[..., FEET, :2].square().sum(-1)
  match = 1 - (c_sim - ref_contact).square()
  return weight * (match + ref_contact * torch.exp(-slip / 0.3**2)).mean(-1)


def soft_failure(robot_state: torch.Tensor, ref_state: torch.Tensor) -> torch.Tensor:
  """BlinkMimic analytic failure probability for differentiable imagination.

  Height and gravity-Z error thresholds match real collection. Sigmoid widths
  are 0.02 m and 0.04 respectively; timeout/clip boundaries remain separate.
  """
  _, dz, dg = physical_failure(robot_state, ref_state)
  alive = (1 - torch.sigmoid((dz - 0.25) / 0.02)) * (1 - torch.sigmoid((dg - 0.8) / 0.04))
  return 1 - alive


def physical_failure(
  robot_state: torch.Tensor, ref_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """SuperMimic hard failure on the pelvis against the original reference.

  Returns (failed, |dz| metres, |dgz| dimensionless).
  """
  down = torch.tensor([0.0, 0.0, -1.0], device=robot_state.device).expand_as(robot_state[..., :3])
  gz_robot = K.quat_rotate_inv(robot_state[..., 3:7], down)[..., 2]
  gz_ref = K.quat_rotate_inv(ref_state[..., 3:7], down)[..., 2]
  dz = (robot_state[..., 2] - ref_state[..., 2]).abs()
  dg = (gz_robot - gz_ref).abs()
  return (dz > 0.25) | (dg > 0.8), dz, dg
