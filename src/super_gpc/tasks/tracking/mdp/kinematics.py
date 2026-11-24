"""Differentiable kinematics shared by references, rewards and the world model.

Generalized state convention (``STATE_DIM`` = 67 scalars, env-local frame):

  [0:3]   pelvis position, world
  [3:7]   pelvis orientation, WXYZ unit quaternion (body to world)
  [7:10]  pelvis link-origin linear velocity, world
  [10:13] pelvis angular velocity, pelvis body frame (MuJoCo free-joint qvel)
  [13:40] joint positions, canonical action order
  [40:67] joint velocities, canonical action order

Forward kinematics follows the compiled MJCF: child orientation = parent x
fixed body quaternion x hinge rotation about the joint axis (all hinges sit at
their body origin with zero ``qpos0``), and link-origin velocities propagate as
``v_c = v_p + w_p x (p_c - p_p)``, ``w_c = w_p + R_c axis qdot``. These are
link-origin (not COM) velocities, matching ``body_link_*_w`` in MJLab.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

STATE_DIM = 67
NJ = 27


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    -1,
  )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
  return torch.cat((q[..., :1], -q[..., 1:]), -1)


def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Explicit cross product (exports to plain ONNX arithmetic)."""
  ax, ay, az = a.unbind(-1)
  bx, by, bz = b.unbind(-1)
  return torch.stack((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), -1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  w, u = q[..., :1], q[..., 1:]
  t = 2.0 * cross(u, v)
  return v + w * t + cross(u, t)


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return quat_rotate(quat_conj(q), v)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.stack(
    (
      1 - 2 * (y * y + z * z),
      2 * (x * y - w * z),
      2 * (x * z + w * y),
      2 * (x * y + w * z),
      1 - 2 * (x * x + z * z),
      2 * (y * z - w * x),
      2 * (x * z - w * y),
      2 * (y * z + w * x),
      1 - 2 * (x * x + y * y),
    ),
    -1,
  ).reshape(*q.shape[:-1], 3, 3)


def quat_from_rotvec(r: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
  """Exponential map; smooth at zero."""
  theta2 = (r * r).sum(-1, keepdim=True)
  theta = torch.sqrt(theta2 + eps)
  half = 0.5 * theta
  # sin(theta/2)/theta with a series near zero.
  small = theta2 < 1e-8
  k = torch.where(small, 0.5 - theta2 / 48.0, torch.sin(half) / theta)
  return torch.cat((torch.cos(half), r * k), -1)


def quat_to_rotvec(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
  """Logarithm map on the short arc."""
  q = torch.where(q[..., :1] < 0, -q, q)
  v = q[..., 1:]
  s = torch.sqrt((v * v).sum(-1, keepdim=True) + eps)
  angle = 2.0 * torch.atan2(s, q[..., :1])
  return v * (angle / s)


def quat_angle_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Squared magnitude of log(a^-1 b), the relative rotation angle squared."""
  r = quat_to_rotvec(quat_mul(quat_conj(a), b))
  return (r * r).sum(-1)


def normalize(q: torch.Tensor) -> torch.Tensor:
  return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def yaw_matrix_from(R: torch.Tensor) -> torch.Tensor:
  """Heading rotation of the body x axis projected on the ground plane.

  Uses no atan2 so it exports to plain ONNX arithmetic.
  """
  hx, hy = R[..., 0, 0], R[..., 1, 0]
  n = torch.sqrt(hx * hx + hy * hy + 1e-12)
  c, s = hx / n, hy / n
  z, o = torch.zeros_like(c), torch.ones_like(c)
  return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(*c.shape, 3, 3)


@dataclass
class BodyStates:
  pos: torch.Tensor  # (..., B, 3)
  quat: torch.Tensor  # (..., B, 4) WXYZ
  lin_vel: torch.Tensor  # (..., B, 3) link origin, world
  ang_vel: torch.Tensor  # (..., B, 3) world


class Kinematics(torch.nn.Module):
  """Forward kinematics for a fixed set of output bodies."""

  def __init__(
    self, mj_model: mujoco.MjModel, joint_names: tuple[str, ...], body_names: tuple[str, ...]
  ):
    super().__init__()
    needed: set[int] = set()
    for name in body_names:
      b = mj_model.body(name).id
      while b > 0:
        needed.add(b)
        b = int(mj_model.body_parentid[b])
    order = sorted(needed)  # MuJoCo bodies are in topological order.
    roots = [b for b in order if int(mj_model.body_parentid[b]) == 0]
    if len(roots) != 1:
      raise ValueError("expected exactly one floating root body")
    self.root = roots[0]
    self.order = order
    slot = {b: i for i, b in enumerate(order)}
    parents, joint_index, axes, offsets, fixed = [], [], [], [], []
    for b in order:
      parents.append(slot.get(int(mj_model.body_parentid[b]), -1))
      offsets.append(mj_model.body_pos[b].copy())
      fixed.append(mj_model.body_quat[b].copy())
      jn = int(mj_model.body_jntnum[b])
      if b == self.root:
        if jn != 1 or mj_model.jnt_type[mj_model.body_jntadr[b]] != mujoco.mjtJoint.mjJNT_FREE:
          raise ValueError("root body must carry the free joint")
        joint_index.append(-1)
        axes.append(np.zeros(3))
        continue
      if jn == 0:
        joint_index.append(-1)
        axes.append(np.zeros(3))
        continue
      j = int(mj_model.body_jntadr[b])
      if jn != 1 or mj_model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError(f"body {mj_model.body(b).name} must have one hinge")
      if np.abs(mj_model.jnt_pos[j]).max() > 0 or mj_model.qpos0[mj_model.jnt_qposadr[j]] != 0:
        raise ValueError("hinges must sit at the body origin with zero qpos0")
      joint_index.append(joint_names.index(mj_model.joint(j).name))
      axes.append(mj_model.jnt_axis[j].copy())
    self.parents = parents
    self.joint_index = joint_index
    self.register_buffer("offsets", torch.tensor(np.array(offsets), dtype=torch.float32))
    self.register_buffer("fixed", torch.tensor(np.array(fixed), dtype=torch.float32))
    self.register_buffer("axes", torch.tensor(np.array(axes), dtype=torch.float32))
    self.out_slots = [slot[mj_model.body(n).id] for n in body_names]
    self.body_names = body_names

  def forward(self, state: torch.Tensor, velocities: bool = True) -> BodyStates:
    pos0, quat0 = state[..., 0:3], normalize(state[..., 3:7])
    q, qd = state[..., 13:40], state[..., 40:67]
    P: list[torch.Tensor] = []
    Q: list[torch.Tensor] = []
    V: list[torch.Tensor] = []
    W: list[torch.Tensor] = []
    for i, _ in enumerate(self.order):
      p = self.parents[i]
      if p < 0:
        P.append(pos0)
        Q.append(quat0)
        if velocities:
          V.append(state[..., 7:10])
          W.append(quat_rotate(quat0, state[..., 10:13]))
        continue
      off = self.offsets[i].expand_as(pos0)
      pos = P[p] + quat_rotate(Q[p], off)
      quat = quat_mul(Q[p], self.fixed[i].expand_as(quat0))
      j = self.joint_index[i]
      axis = self.axes[i].expand_as(pos0)
      if j >= 0:
        quat = quat_mul(quat, quat_from_rotvec(axis * q[..., j : j + 1]))
      P.append(pos)
      Q.append(quat)
      if velocities:
        w = W[p]
        if j >= 0:
          w = w + quat_rotate(quat, axis) * qd[..., j : j + 1]
        V.append(V[p] + cross(W[p], pos - P[p]))
        W.append(w)
    s = self.out_slots
    stack = lambda xs: torch.stack([xs[k] for k in s], -2)  # noqa: E731
    if velocities:
      return BodyStates(stack(P), stack(Q), stack(V), stack(W))
    return BodyStates(stack(P), stack(Q), torch.empty(0), torch.empty(0))
