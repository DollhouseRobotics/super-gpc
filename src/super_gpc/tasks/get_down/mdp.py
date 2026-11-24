"""Get-down task: task input, impact reward, starts, AMP transition features,
pelvis-height reward and the reference transitions.

The task input uses only proprioceptive torso orientation and angular
velocity, so it needs no state estimation. The impact terms penalize floor
contact forces above per-body thresholds, so the robot lowers itself instead
of dropping. All weights, kernels, gates, frames and schedules are our
choices, recorded in docs/spec.md and marked Choice in ``GetDownCfg``.

Every function acts on the clean generalized state convention of
``tracking.mdp.kinematics`` (quaternions WXYZ, pelvis angular velocity in the
pelvis frame), so simulator states, prepared reference frames and the
exported graph share one implementation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import torch

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1
from super_gpc.tasks.tracking.mdp import kinematics as K
from super_gpc.tasks.tracking.mdp.commands import (
  RESET_POSE_RANGE,
  RESET_VELOCITY_RANGE,
  write_start_state,
)
from super_gpc.tasks.tracking.mdp.motion_library import POLICY_HZ, MotionLibrary

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

# Bodies read by the task input, the critic and the lying gate, in this order.
BODIES = ("pelvis", "torso_link", "head_tilt_link")
PELVIS, TORSO, HEAD = 0, 1, 2
TORSO_CONTACT_BODIES = ("torso_link",)
WRIST_CONTACT_BODIES = (
  "left_wrist_roll_link",
  "left_wrist_hand_link",
  "right_wrist_roll_link",
  "right_wrist_hand_link",
)
TASK_FRAME_DIM = 6
TASK_FRAMES = 5
TASK_DIM = TASK_FRAME_DIM * TASK_FRAMES
# Joystick-rollout start labels.
LABELS = ("standing", "moving", "disturbed")

# Key bodies of the AMP frame, in this order (the torso first: on_back reads it).
AMP_BODIES = (
  "torso_link",
  "head_tilt_link",
  "left_wrist_hand_link",
  "right_wrist_hand_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)
AMP_FRAME_DIM = 1 + 6 + 3 + 3 + 2 * K.NJ + 3 * len(AMP_BODIES)  # 85
AMP_DIM = 2 * AMP_FRAME_DIM


def build_kinematics(device: str, bodies: tuple[str, ...] = BODIES) -> K.Kinematics:
  mj = mujoco.MjModel.from_xml_path(str(G1.mjcf_path()))
  return K.Kinematics(mj, G1.ACTION_JOINT_NAMES, bodies).to(device)


# --- task input and critic features -------------------------------------------
def task_frame(body: K.BodyStates, state: torch.Tensor) -> torch.Tensor:
  """One task-input frame (6): torso orientation as gravity in the torso
  frame (3) and torso angular velocity in the torso frame (3). Both follow
  from the pelvis IMU (orientation, gyro) and joint encoders through forward
  kinematics: no base position, height or linear velocity, so no state
  estimation."""
  q = body.quat[..., TORSO, :]
  zero = torch.zeros_like(state[..., 0])
  down = torch.stack((zero, zero, zero - 1), -1)  # no in-place writes: they export as ScatterND
  return torch.cat(
    (K.quat_rotate_inv(q, down), K.quat_rotate_inv(q, body.ang_vel[..., TORSO, :])), -1
  )


def estimated_features(body: K.BodyStates, state: torch.Tensor) -> torch.Tensor:
  """Critic-only features a robot needs state estimation for (6): torso,
  pelvis and head heights and pelvis linear velocity in the heading frame."""
  heading = K.yaw_matrix_from(K.quat_to_matrix(state[..., 3:7]))
  v = (heading.transpose(-1, -2) @ state[..., 7:10, None])[..., 0]
  heights = torch.stack([body.pos[..., k, 2] for k in (TORSO, PELVIS, HEAD)], -1)
  return torch.cat((heights, v), -1)


def torso_forward_z(body: K.BodyStates, torso: int = TORSO) -> torch.Tensor:
  """World-z component of the torso x (chest) axis: +1 face up, -1 face down."""
  w, x, y, z = body.quat[..., torso, :].unbind(-1)
  return 2 * (x * z - w * y)


def lying(body: K.BodyStates, cfg) -> torch.Tensor:
  """Choice: the lying pose is a low torso within the horizontal band, on
  either side."""
  return (body.pos[..., TORSO, 2] < cfg.lying_height) & (
    torso_forward_z(body).abs() > cfg.lying_horizontal
  )


# --- impact and regularizer reward ----------------------------------------------
def impact_weights(cfg, progress: float) -> tuple[float, float, float]:
  """Whole-body, torso and wrist impact weights, ramped linearly from the
  initial to the final values over ``impact_ramp_iterations``."""
  a = min(max(progress, 0.0), 1.0)
  pairs = (
    (cfg.body_impact_initial, cfg.body_impact_weight),
    (cfg.torso_impact_initial, cfg.torso_impact_weight),
    (cfg.wrist_impact_initial, cfg.wrist_impact_weight),
  )
  return tuple(lo + a * (hi - lo) for lo, hi in pairs)


def impact_reward(
  floor_force: torch.Tensor,
  torso_ids: torch.Tensor,
  wrist_ids: torch.Tensor,
  target_rate: torch.Tensor,
  joint_pos: torch.Tensor,
  soft_lo: torch.Tensor,
  soft_hi: torch.Tensor,
  impact: tuple[float, float, float],
  impact_mask: torch.Tensor,
  cfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """Per-env impact and regularizer terms, summed without dt scaling.

  ``floor_force`` (N, B) is each body's largest net floor-contact force over
  the step's physics substeps (N). ``target_rate`` is the physical
  joint-target change (rad); ``impact_mask`` disables the impact terms on the
  first step after a reset, whose contact transient comes from the written
  start state rather than the policy.
  """
  over = (joint_pos - soft_hi).clamp_min(0) + (soft_lo - joint_pos).clamp_min(0)
  body_w, torso_w, wrist_w = impact
  m = impact_mask.float()
  terms = {
    "body_impact": -body_w * m * (floor_force - cfg.body_impact_n).clamp_min(0).sum(-1),
    "torso_impact": -torso_w
    * m
    * (floor_force[:, torso_ids] - cfg.torso_impact_n).clamp_min(0).sum(-1),
    "wrist_impact": -wrist_w
    * m
    * (floor_force[:, wrist_ids] - cfg.wrist_impact_n).clamp_min(0).sum(-1),
    "action_rate": -cfg.action_rate_weight * target_rate.norm(dim=-1),
    "joint_limit": -cfg.joint_limit_weight * over.sum(-1),
  }
  return sum(terms.values()), terms


# --- AMP style reference and pelvis height ---------------------------------------
def heading_frame(q: torch.Tensor) -> torch.Tensor:
  """Yaw-only frame from the pelvis lateral (y) axis. The usual forward-axis
  heading degenerates when lying on the back, where the forward axis points
  up; the lateral axis stays horizontal standing and lying on the back."""
  R = K.quat_to_matrix(q)
  ly, lx = R[..., 1, 1], R[..., 0, 1]
  n = torch.sqrt(lx * lx + ly * ly + 1e-12)
  s, c = -lx / n, ly / n  # heading angle of the forward axis = lateral axis - 90 deg
  z, o = torch.zeros_like(c), torch.ones_like(c)
  return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(*c.shape, 3, 3)


def amp_frame(body: K.BodyStates, state: torch.Tensor) -> torch.Tensor:
  """One AMP frame (85) over ``AMP_BODIES``: pelvis height (1); pelvis
  orientation in the heading frame as its forward and up axes (6); pelvis
  linear velocity in the heading frame (3); pelvis angular velocity in the
  pelvis frame (3); joint positions and velocities (54); key-body positions
  relative to the pelvis in the heading frame (18)."""
  q = state[..., 3:7]
  Ht = heading_frame(q).transpose(-1, -2)
  R = Ht @ K.quat_to_matrix(q)
  v = (Ht @ state[..., 7:10, None])[..., 0]
  rel = body.pos - state[..., None, 0:3]
  rel = (Ht[..., None, :, :] @ rel[..., None])[..., 0]
  return torch.cat(
    (
      state[..., 2:3],
      R[..., :, 0],
      R[..., :, 2],
      v,
      state[..., 10:13],
      state[..., 13:67],
      rel.flatten(-2),
    ),
    -1,
  )


def chest_up(amp_body: K.BodyStates) -> torch.Tensor:
  """World-z component of the torso x (chest) axis over ``AMP_BODIES``: +1
  lying on the back."""
  return torso_forward_z(amp_body, torso=0)


def torso_on_back(amp_body: K.BodyStates, cfg) -> torch.Tensor:
  """Lying on the back: the lying gate with the chest facing up."""
  return (amp_body.pos[..., 0, 2] < cfg.lying_height) & (chest_up(amp_body) > cfg.lying_horizontal)


def pelvis_height_reward(height: torch.Tensor, up: torch.Tensor, cfg) -> torch.Tensor:
  """Linear from 0 at the standing pelvis height to the full weight at the
  reference's lying pelvis height and below, scaled by how far the chest
  faces up (``up`` from ``chest_up``, clamped to [0, 1]), so a squat or a
  face-down pose earns little of it."""
  span = cfg.pelvis_stand - cfg.pelvis_low
  low = (1 - (height - cfg.pelvis_low).clamp_min(0) / span).clamp(0, 1)
  return cfg.pelvis_weight * low * up.clamp(0, 1)


@torch.no_grad()
def reference_transitions(
  library: MotionLibrary, tail_s: float, kin: K.Kinematics, cfg
) -> tuple[torch.Tensor, list[float]]:
  """Consecutive-frame AMP transitions (M, 170) of every library clip and
  each clip's end time. A clip runs from its first frame to ``tail_s``
  before the end of its first stretch lying on the back (``torso_on_back``),
  which drops the start of standing up again."""
  out, ends = [], []
  for c in range(library.num_clips):
    frames = torch.arange(int(library.lengths[c]), device=library.device)
    s = library.state(torch.full_like(frames, c), frames)
    body = kin(s, velocities=False)
    on_back = torso_on_back(body, cfg).tolist()
    if True not in on_back:
      raise ValueError(f"reference clip {library.clips[c]['name']} never lies on its back")
    first = on_back.index(True)
    stop = on_back.index(False, first) if False in on_back[first:] else len(on_back)
    last = stop - 1 - round(tail_s * POLICY_HZ)
    if last <= first:
      raise ValueError(f"reference clip {library.clips[c]['name']} lies down too briefly")
    f = amp_frame(body, s)[: last + 1]
    out.append(torch.cat((f[:-1], f[1:]), -1))
    ends.append(last / POLICY_HZ)
  return torch.cat(out), ends


# --- start states -------------------------------------------------------------
def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
  half = 0.5 * yaw
  z = torch.zeros_like(half)
  return torch.stack((half.cos(), z, z, half.sin()), -1)


def rotate_start(state: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """Turn generalized states about world z by ``yaw`` and place the pelvis at
  the env origin in the plane. Body-frame angular velocity is unchanged."""
  out = state.clone()
  out[:, 0:2] = 0.0
  out[:, 3:7] = K.normalize(K.quat_mul(yaw_quat(yaw), state[:, 3:7]))
  c, s = yaw.cos(), yaw.sin()
  vx, vy = state[:, 7], state[:, 8]
  out[:, 7], out[:, 8] = c * vx - s * vy, s * vx + c * vy
  return out


def heading(state: torch.Tensor) -> torch.Tensor:
  """Pelvis heading angle (yaw of the projected chest axis)."""
  w, x, y, z = state[:, 3:7].unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def push(state: torch.Tensor, forward: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
  """Add a planar velocity impulse in the pelvis heading frame (m/s)."""
  out = state.clone()
  heading = K.yaw_matrix_from(K.quat_to_matrix(state[:, 3:7]))
  out[:, 7:9] += heading[:, 0:2, 0] * forward[:, None] + heading[:, 0:2, 1] * left[:, None]
  return out


class GetDownStarts:
  """Reset-mode event term owning the start states.

  With probability ``joystick_fraction`` an env starts from a uniformly
  drawn joystick-rollout state turned to a uniform heading; otherwise from a
  uniform one-second bin of the motion archive with the Stage 1 root pose,
  root velocity and +-0.1 rad joint perturbations (the archive supplies
  starts only, never a reference). ``assignment`` (N, 67), set by an
  evaluator, writes exact states instead. The library, bank and joint map
  are attached after environment construction; the startup reset before
  that leaves the default pose.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self.env = env
    self.robot = env.scene["robot"]
    self.library: MotionLibrary | None = None
    self.bank: torch.Tensor | None = None
    self.action_to_tree: torch.Tensor | None = None
    self.joystick_fraction = 0.5
    self.gen = torch.Generator(device=env.device)
    self.source = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.assignment: torch.Tensor | None = None

  def attach(self, library, bank, action_to_tree, joystick_fraction: float, seed: int) -> None:
    self.library, self.bank, self.action_to_tree = library, bank, action_to_tree
    self.joystick_fraction = joystick_fraction
    self.gen.manual_seed(seed)

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    if self.action_to_tree is None:
      return
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    origins, a2t = env.scene.env_origins, self.action_to_tree
    if self.assignment is not None:
      self.source[env_ids] = 2
      write_start_state(
        self.robot, origins, env_ids, self.assignment[env_ids], {}, {}, (0.0, 0.0), a2t
      )
      return
    n, dev = len(env_ids), env.device
    u = torch.rand(n, 3, device=dev, generator=self.gen)
    joy = u[:, 0] < self.joystick_fraction
    self.source[env_ids] = joy.long()
    a, j = env_ids[~joy], env_ids[joy]
    if len(a):
      lib = self.library
      bins = (u[~joy, 1] * lib.num_bins).long().clamp(max=lib.num_bins - 1)
      clip, frame = lib.sample_start(bins)
      ref = lib.state(clip, frame)
      write_start_state(
        self.robot, origins, a, ref, RESET_POSE_RANGE, RESET_VELOCITY_RANGE, (-0.1, 0.1), a2t
      )
    if len(j):
      pick = (u[joy, 1] * len(self.bank)).long().clamp(max=len(self.bank) - 1)
      yaw = (2 * u[joy, 2] - 1) * math.pi
      write_start_state(
        self.robot, origins, j, rotate_start(self.bank[pick], yaw), {}, {}, (0.0, 0.0), a2t
      )
