"""Joystick steering commands and rewards, ported from the G1 gpc steering task.

Source: DollhouseRobotics/g1-gpc ``src/dollhouse_g1/steering.py``
(``SteeringConfig``, ``SteeringCommands``, ``steering_rewards``). Angles are
counter-clockwise, x forward, y left; quaternions are WXYZ here. Headings are
the pelvis (root) yaw, as the source uses its root IMU yaw.

Our extension (spec "Idle gaze"): a gaze command
(yaw offset from the commanded facing and pitch, positive up) that is
straight ahead while moving and, at zero commanded speed, jumps to random
look targets; a gaze reward on the head link's forward axis, stillness and
planted-feet terms once a stopped robot faces its command, and an
arm-posture penalty. The frozen decoder cannot turn the neck from a standing
state (measured; the corpus has neck motion, but little while standing), so
the two neck joints follow the gaze command analytically (``neck_targets``)
and the CoLA observes them at the pose its own neck actions command
(``mask_neck``); the gaze command slews toward each new look target at
``gaze_rate``.
"""

from __future__ import annotations

import math

import torch

from super_gpc.tasks.tracking.mdp import kinematics as K

NECK = slice(25, 27)  # head pan, head tilt in canonical action order
WAIST_YAW = 12  # waist yaw, roll, pitch follow in action order


def rotate_xy(v: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  c, s = angle.cos(), angle.sin()
  x, y = v.unbind(-1)
  return torch.stack((c * x - s * y, s * x + c * y), -1)


def yaw_wxyz(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def neck_targets(state: torch.Tensor, command: torch.Tensor, pan_max: float, tilt_max: float):
  """(N, 2) head pan and tilt that point the head link's x axis along the
  commanded gaze: the local command's (cos, sin) gaze direction in pelvis-yaw
  axes and pitch, solved exactly through the pelvis orientation and the
  waist yaw (z), roll (x) and pitch (y) joints. All rest orientations are
  identity; positive tilt pitches the head down."""
  quat = state[:, 3:7]
  g = yaw_wxyz(quat) + torch.atan2(command[:, 5], command[:, 4])
  p = command[:, 6]
  d = K.quat_rotate_inv(quat, torch.stack((p.cos() * g.cos(), p.cos() * g.sin(), p.sin()), -1))
  q = state[:, 13 + WAIST_YAW : 16 + WAIST_YAW]
  x, y, z = d.unbind(-1)
  c, s = q[:, 0].cos(), q[:, 0].sin()
  x, y = c * x + s * y, -s * x + c * y
  c, s = q[:, 1].cos(), q[:, 1].sin()
  y, z = c * y + s * z, -s * y + c * z
  c, s = q[:, 2].cos(), q[:, 2].sin()
  x, z = c * x - s * z, s * x + c * z
  pan = torch.atan2(y, x).clamp(-pan_max, pan_max)
  tilt = (-torch.atan2(z, (x * x + y * y).sqrt())).clamp(-tilt_max, tilt_max)
  return torch.stack((pan, tilt), -1)


def mask_neck(state, prev_action, default_q, scale):
  """The CoLA's view of the robot: neck joints at the pose its previous
  normalized neck action commanded, at rest."""
  out = state.clone()
  out[:, 13 + NECK.start : 13 + NECK.stop] = (default_q + scale * prev_action)[:, NECK]
  out[:, 40 + NECK.start : 40 + NECK.stop] = 0.0
  return out


def uprightness_wxyz(q: torch.Tensor) -> torch.Tensor:
  """World-z component of the body z axis."""
  return 1 - 2 * q[..., 1:3].square().sum(-1)


class SteeringCommands:
  """A facing command anchors at the current yaw until facing changes again.

  Velocity-only updates and a repeated facing value keep the anchor; an
  episode reset starts a new anchor. Commanded velocity is in the commanded
  facing frame. Commands advance after their transition has been scored.
  """

  def __init__(self, count: int, device: str, cfg, seed: int):
    self.cfg = cfg
    self.gen = torch.Generator(device=device).manual_seed(seed)
    self.anchor = torch.zeros(count, device=device)
    self.facing = torch.zeros(count, device=device)
    self.velocity = torch.zeros(count, 2, device=device)
    self.remaining = torch.zeros(count, dtype=torch.long, device=device)
    self.gaze = torch.zeros(count, 2, device=device)  # yaw offset from facing, pitch (up)
    self.gaze_goal = torch.zeros(count, 2, device=device)  # look target the gaze slews to
    self.gaze_remaining = torch.zeros(count, dtype=torch.long, device=device)

  def issue(self, ids, yaw, velocity, facing, reset: bool = False) -> None:
    delta = facing - self.facing[ids]
    changed = torch.atan2(delta.sin(), delta.cos()).abs() > 1e-6
    if reset:
      changed = torch.ones_like(changed)
    self.anchor[ids] = torch.where(changed, yaw, self.anchor[ids])
    self.facing[ids] = torch.where(changed, facing, self.facing[ids])
    self.velocity[ids] = velocity

  def idle(self) -> torch.Tensor:
    return self.velocity.norm(dim=-1) < self.cfg.near_zero_speed

  def world(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World velocity, facing direction (cos, sin) and unit gaze target (3)."""
    angle = self.anchor + self.facing
    gaze_yaw, pitch = angle + self.gaze[:, 0], self.gaze[:, 1]
    gaze = torch.stack(
      (pitch.cos() * gaze_yaw.cos(), pitch.cos() * gaze_yaw.sin(), pitch.sin()), -1
    )
    return rotate_xy(self.velocity, angle), torch.stack((angle.cos(), angle.sin()), -1), gaze

  def local(self, yaw: torch.Tensor) -> torch.Tensor:
    """(N, 7) policy command in robot-yaw axes: velocity, facing direction and
    the gaze direction's horizontal (cos, sin) and pitch; (N, 4) without idle
    gaze."""
    velocity, facing, _ = self.world()
    rel = self.anchor + self.facing + self.gaze[:, 0] - yaw
    return torch.cat(
      (
        rotate_xy(velocity, -yaw),
        rotate_xy(facing, -yaw),
        torch.stack((rel.cos(), rel.sin()), -1),
        self.gaze[:, 1:],
      ),
      -1,
    )

  def sample_gaze(self, ids: torch.Tensor) -> None:
    """New idle look target: centered with ``gaze_center_probability``, else
    uniform in yaw and pitch; held for ``gaze_min_steps``..``gaze_max_steps``."""
    cfg, n = self.cfg, len(ids)
    if n == 0:
      return
    dev = self.gaze.device
    r = torch.rand(n, 3, device=dev, generator=self.gen)
    yaw = (2 * r[:, 0] - 1) * cfg.gaze_yaw_max
    pitch = cfg.gaze_pitch_min + (cfg.gaze_pitch_max - cfg.gaze_pitch_min) * r[:, 1]
    target = torch.stack((yaw, pitch), -1)
    center = (r[:, 2] < cfg.gaze_center_probability)[:, None]
    self.gaze_goal[ids] = torch.where(center, 0.0, target)
    self.gaze_remaining[ids] = torch.randint(
      cfg.gaze_min_steps, cfg.gaze_max_steps + 1, (n,), device=dev, generator=self.gen
    )

  def sample(self, ids: torch.Tensor, yaw: torch.Tensor, reset: bool = False) -> None:
    cfg, n = self.cfg, len(ids)
    if n == 0:
      return
    r = torch.rand(n, 5, device=yaw.device, generator=self.gen)
    speed = cfg.speed_min + (cfg.speed_max - cfg.speed_min) * r[:, 0]
    speed = torch.where(r[:, 1] < cfg.stop_probability, 0.0, speed)
    direction = (2 * r[:, 2] - 1) * math.pi
    velocity = speed[:, None] * torch.stack((direction.cos(), direction.sin()), -1)
    previous = torch.zeros_like(self.facing[ids]) if reset else self.facing[ids]
    facing = torch.where(
      r[:, 4] < cfg.facing_change_probability, (2 * r[:, 3] - 1) * math.pi, previous
    )
    self.issue(ids, yaw, velocity, facing, reset=reset)
    walk = torch.randint(
      cfg.command_min_steps, cfg.command_max_steps + 1, (n,), device=yaw.device, generator=self.gen
    )
    idle = torch.randint(
      cfg.idle_min_steps, cfg.idle_max_steps + 1, (n,), device=yaw.device, generator=self.gen
    )
    self.remaining[ids] = torch.where(speed == 0, idle, walk)
    # Every new command turns back to look straight ahead along the facing.
    self.gaze_goal[ids] = 0.0
    if reset:
      self.gaze[ids] = 0.0
    self.gaze_remaining[ids] = torch.randint(
      cfg.gaze_min_steps, cfg.gaze_max_steps + 1, (n,), device=yaw.device, generator=self.gen
    )

  def advance(self, yaw: torch.Tensor) -> None:
    step = self.cfg.gaze_rate / 50.0
    self.gaze += (self.gaze_goal - self.gaze).clamp(-step, step)
    self.remaining -= 1
    self.gaze_remaining -= 1
    ids = torch.where(self.remaining <= 0)[0]
    self.sample(ids, yaw[ids])
    self.sample_gaze(torch.where((self.gaze_remaining <= 0) & self.idle())[0])

  KEYS = ("anchor", "facing", "velocity", "remaining", "gaze", "gaze_goal", "gaze_remaining")

  def state_dict(self) -> dict:
    out = {k: getattr(self, k).clone() for k in self.KEYS}
    return out | {"rng": self.gen.get_state()}

  def load_state_dict(self, state: dict) -> None:
    for k in self.KEYS:
      getattr(self, k).copy_(state[k])
    self.gen.set_state(state["rng"].cpu())


def steering_reward(
  yaw: torch.Tensor,
  velocity: torch.Tensor,
  target_velocity: torch.Tensor,
  target_facing: torch.Tensor,
  target_gaze: torch.Tensor,
  head_forward: torch.Tensor,
  foot_velocity: torch.Tensor,
  target_rate: torch.Tensor,
  joint_pos: torch.Tensor,
  soft_lo: torch.Tensor,
  soft_hi: torch.Tensor,
  contacts: torch.Tensor,
  arm_offset: torch.Tensor,
  cfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """Per-env reward terms, summed without dt scaling or env averaging.

  ``velocity`` is the pelvis world-frame planar velocity; ``target_rate`` is
  the physical joint-target change since the previous executed step (rad);
  ``contacts`` counts non-foot bodies in contact. Direction is disabled when
  commanded or measured speed is below ``near_zero_speed``; stops still
  reward facing and speed. ``head_forward`` is the head link's world x axis
  and ``target_gaze`` the commanded world gaze direction; ``foot_velocity``
  (N, 2, 3) holds the ankle-roll link-origin velocities and ``arm_offset``
  the arm joints' offsets from the default pose (rad).
  Gaze is scored at every step (straight ahead while moving); stillness and
  planted feet only once a stopped robot faces its command within
  ``settle_facing_rad``, so turning in place may still step.
  """
  face = torch.stack((yaw.cos(), yaw.sin()), -1)
  facing_cos = (face * target_facing).sum(-1).clamp(-1, 1)
  speed = velocity.norm(dim=-1)
  desired = target_velocity.norm(dim=-1)
  direction_cos = ((velocity * target_velocity).sum(-1) / (speed * desired).clamp_min(1e-12)).clamp(
    -1, 1
  )
  enabled = (speed >= cfg.near_zero_speed) & (desired >= cfg.near_zero_speed)
  gaze_cos = (head_forward * target_gaze).sum(-1).clamp(-1, 1)
  settled = (desired < cfg.near_zero_speed) & (facing_cos > math.cos(cfg.settle_facing_rad))
  foot_speed = foot_velocity[..., :2].norm(dim=-1).sum(-1)
  over = (joint_pos - soft_hi).clamp_min(0) + (soft_lo - joint_pos).clamp_min(0)
  terms = {
    "facing": cfg.facing_weight * torch.exp((facing_cos - 1) / cfg.facing_width),
    "direction": cfg.direction_weight
    * torch.where(enabled, torch.exp((direction_cos - 1) / cfg.direction_width), 0.0),
    "speed": cfg.speed_weight * torch.exp(-0.5 * ((speed - desired) / cfg.speed_sigma).square()),
    "action_rate": cfg.action_rate_weight * target_rate.norm(dim=-1),
    "joint_limit": cfg.joint_limit_weight * over.sum(-1),
    "contact": cfg.contact_weight * contacts,
    "gaze": cfg.gaze_weight * torch.exp((gaze_cos - 1) / cfg.gaze_width),
    "still": cfg.still_weight * settled * torch.exp(-((speed / cfg.still_sigma).square())),
    "foot_slip": cfg.foot_slip_weight * settled * foot_speed,
    "arm_pose": cfg.arm_pose_weight * arm_offset.square().sum(-1),
  }
  return sum(terms.values()), terms
