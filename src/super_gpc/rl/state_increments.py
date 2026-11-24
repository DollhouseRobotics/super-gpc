"""Velocity-delta dynamics with SuperMimic's analytic pose integration.

Prediction: current-pelvis-frame linear/angular velocity deltas (3 each),
then 27 joint velocity deltas. Stored angular velocity is body-frame, so
next-state angular velocity must be transported between pelvis frames.
Pose integrates the current velocity plus 0.625 of the delta, matching four
semi-implicit physics substeps per control step. No pose correction is learned.
Full 66-channel state errors still supervise pose and velocity over rollouts.
"""

from __future__ import annotations

import torch

from super_gpc.tasks.tracking.mdp import kinematics as K
from super_gpc.tasks.tracking.tracking_env_cfg import DECIMATION, PHYSICS_DT

INCREMENT_DIM = 33
STATE_ERROR_DIM = 66


def _heading(state: torch.Tensor) -> torch.Tensor:
  return K.yaw_matrix_from(K.quat_to_matrix(state[..., 3:7]))


def _to_heading(H: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return (v[..., None, :] @ H)[..., 0, :]


def increment(state: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
  q, q_next = state[..., 3:7], nxt[..., 3:7]
  return torch.cat(
    (
      K.quat_rotate_inv(q, nxt[..., 7:10] - state[..., 7:10]),
      K.quat_rotate_inv(q, K.quat_rotate(q_next, nxt[..., 10:13])) - state[..., 10:13],
      nxt[..., 40:67] - state[..., 40:67],
    ),
    -1,
  )


def state_error(state: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
  """Pose and velocity error in heading axes, with tangent-space rotation error."""
  H = _heading(state)
  dq = K.quat_to_rotvec(K.quat_mul(K.quat_conj(state[..., 3:7]), nxt[..., 3:7]))
  return torch.cat(
    (
      _to_heading(H, nxt[..., 0:3] - state[..., 0:3]),
      dq,
      _to_heading(H, nxt[..., 7:10] - state[..., 7:10]),
      nxt[..., 10:67] - state[..., 10:67],
    ),
    -1,
  )


def compose(state: torch.Tensor, inc: torch.Tensor) -> torch.Tensor:
  """Next state from the predicted increment."""
  dt = PHYSICS_DT * DECIMATION
  c = (DECIMATION + 1) / (2 * DECIMATION)
  q, w = state[..., 3:7], state[..., 10:13]
  dv, dw, dvj = inc[..., :3], inc[..., 3:6], inc[..., 6:]
  quat = K.normalize(K.quat_mul(q, K.quat_from_rotvec((w + c * dw) * dt)))
  joint_q = state[..., 13:40] + (state[..., 40:67] + c * dvj) * dt
  joint_v = state[..., 40:67] + dvj
  return torch.cat(
    (
      state[..., 0:3] + (state[..., 7:10] + c * K.quat_rotate(q, dv)) * dt,
      quat,
      state[..., 7:10] + K.quat_rotate(q, dv),
      K.quat_rotate_inv(quat, K.quat_rotate(q, w + dw)),
      joint_q,
      joint_v,
    ),
    -1,
  )
