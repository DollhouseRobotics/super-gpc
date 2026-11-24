"""Actor, encoder and privileged observation functions.

All functions are stateless and differentiable. They act on the generalized
state convention in ``kinematics.py`` so that simulator collection and
imagined rollouts share one implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from super_gpc.tasks.tracking.mdp import kinematics as K

# Widths of the 27-joint policy interface.
ACTION_DIM = K.NJ
OBS_DIM = 87
PREV_ACTION = slice(60, 87)  # previous normalized action channels of a frame
HISTORY = 10
HISTORY_DIM = HISTORY * OBS_DIM
FUTURE = 10
ENCODER_FRAME_DIM = 69
ENCODER_DIM = FUTURE * ENCODER_FRAME_DIM
LOCAL_STATE_DIM = 64

DEFAULT_REFERENCE_OFFSETS = tuple(range(0, 46, 5))


def validate_reference_offsets(offsets: Sequence[int]) -> tuple[int, ...]:
  """Ten chronological policy-frame samples; zero is a conditioning sample only."""
  offsets = tuple(offsets)
  if (
    len(offsets) != FUTURE
    or any(type(v) is not int or v < 0 for v in offsets)
    or any(a >= b for a, b in zip(offsets, offsets[1:], strict=False))
  ):
    raise ValueError("reference_offsets must be ten strictly increasing nonnegative integers")
  return offsets


# Additive uniform sensor-noise half widths per observation block (SM Table A.1):
# joint position offsets, joint velocities, IMU angular velocity, projected gravity.
NOISE_HALF_WIDTH = (0.01, 0.5, 0.2, 0.05)


def robot_generalized_state(
  robot, env_origins: torch.Tensor, action_to_tree: torch.Tensor
) -> torch.Tensor:
  """Clean env-local generalized state (67) read from MJLab qpos/qvel.

  qpos/qvel are current after a step, unlike derived body quantities, which
  MJLab leaves one substep stale until its next forward().
  """
  data, idx = robot.data.data, robot.indexing
  qpos = data.qpos[:, idx.free_joint_q_adr]
  qvel = data.qvel[:, idx.free_joint_v_adr]
  return torch.cat(
    (
      qpos[:, 0:3] - env_origins,
      qpos[:, 3:7],
      qvel,
      robot.data.joint_pos[:, action_to_tree],
      robot.data.joint_vel[:, action_to_tree],
    ),
    -1,
  )


def noise_scale(device) -> torch.Tensor:
  """Per-channel half widths of a frame; the previous-action channels are noiseless."""
  n = K.NJ
  blocks = zip((n, n, 3, 3, n), (*NOISE_HALF_WIDTH, 0.0), strict=True)
  return torch.cat([torch.full((k,), w, device=device) for k, w in blocks])


def actor_observation(
  state: torch.Tensor,
  prev_action: torch.Tensor,
  encoder_bias: torch.Tensor,
  default_q: torch.Tensor,
) -> torch.Tensor:
  """Clean 87-scalar proprioceptive frame.

  Joint positions are sensed with the persistent calibration bias and
  expressed relative to the default pose. Gyro and gravity use the pelvis IMU
  frame; the IMU site carries no rotation relative to the pelvis body. There
  is no base linear velocity: the frame needs no state estimation.
  """
  quat = state[..., 3:7]
  gravity = K.quat_rotate_inv(
    quat, torch.tensor([0.0, 0.0, -1.0], device=state.device).expand_as(state[..., :3])
  )
  q, v = state[..., 13:40] + encoder_bias - default_q, state[..., 40:67]
  return torch.cat(
    (
      q,
      v,
      state[..., 10:13],
      gravity,
      prev_action,
    ),
    -1,
  )


def add_sensor_noise(obs: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  return obs + (2.0 * torch.rand_like(obs) - 1.0) * scale


def local_state(state: torch.Tensor) -> torch.Tensor:
  """Heading- and XY-invariant physical features (64)."""
  quat = state[..., 3:7]
  down = torch.tensor([0.0, 0.0, -1.0], device=state.device).expand_as(state[..., :3])
  return torch.cat(
    (
      state[..., 2:3],
      K.quat_rotate_inv(quat, down),
      K.quat_rotate_inv(quat, state[..., 7:10]),
      state[..., 10:13],
      state[..., 13:67],
    ),
    -1,
  )


def _torso(kin_torso: K.Kinematics, state: torch.Tensor) -> K.BodyStates:
  b = kin_torso(state)
  return K.BodyStates(
    b.pos[..., 0, :], b.quat[..., 0, :], b.lin_vel[..., 0, :], b.ang_vel[..., 0, :]
  )


def encoder_features(
  kin_torso: K.Kinematics,
  robot_state: torch.Tensor,
  ref_now: torch.Tensor,
  ref_future: torch.Tensor,
) -> torch.Tensor:
  """BeyondMimic-aligned future reference window (690 scalars).

  ``robot_state`` and ``ref_now`` are (..., 67); ``ref_future`` holds the
  ten configured policy-frame samples, (..., 10, 67), CQ offsets 0, 5, ..., 45.
  One alignment transform from the
  current robot and reference torso poses is applied to every future frame
  (not recentred per frame); results are expressed relative to the actual
  torso position in robot-yaw axes. ``kin_torso`` outputs only torso_link.
  """
  robot, now = _torso(kin_torso, robot_state), _torso(kin_torso, ref_now)
  fut = _torso(kin_torso, ref_future)
  R_robot = K.quat_to_matrix(robot.quat)
  R_delta = K.yaw_matrix_from(R_robot @ K.quat_to_matrix(now.quat).transpose(-1, -2))
  R_yaw = K.yaw_matrix_from(R_robot)
  anchor = torch.cat((robot.pos[..., :2], now.pos[..., 2:3]), -1)
  M = R_yaw.transpose(-1, -2) @ R_delta  # aligned window in robot-yaw axes
  Mt = M.transpose(-1, -2)[..., None, :, :]
  rel = ((fut.pos - now.pos[..., None, :])[..., None, :] @ Mt)[..., 0, :]
  rel = rel + ((anchor - robot.pos)[..., None, :] @ R_yaw)
  rot6 = (M[..., None, :, :] @ K.quat_to_matrix(fut.quat))[..., :, :2].transpose(-1, -2)
  rot6 = rot6.reshape(*rel.shape[:-1], 6)
  lin = (fut.lin_vel[..., None, :] @ Mt)[..., 0, :]
  ang = (fut.ang_vel[..., None, :] @ Mt)[..., 0, :]
  frames = torch.cat((rel, rot6, lin, ang, ref_future[..., 13:67]), -1)
  return frames.reshape(*frames.shape[:-2], ENCODER_DIM)


def anchor_errors(
  kin_torso: K.Kinematics, robot_state: torch.Tensor, ref_state: torch.Tensor
) -> torch.Tensor:
  """Original-global torso anchor position/orientation error in robot-yaw axes (6)."""
  robot, ref = _torso(kin_torso, robot_state), _torso(kin_torso, ref_state)
  R_yaw = K.yaw_matrix_from(K.quat_to_matrix(robot.quat))
  dp = ((ref.pos - robot.pos)[..., None, :] @ R_yaw)[..., 0, :]
  dr = K.quat_to_rotvec(K.quat_mul(K.quat_conj(robot.quat), ref.quat))
  return torch.cat((dp, dr), -1)
