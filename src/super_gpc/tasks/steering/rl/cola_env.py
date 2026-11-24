"""MJLab joystick-steering collection with trainer-owned episode bookkeeping.

Keeps the Stage 1 sensing contract the frozen prior and decoder were trained
on: the noisy 87-scalar frame with calibration bias and the executed previous
normalized action, a ten-frame decoder history padded with the first frame at
reset, and the Stage 1 physical randomization. Commands, the G1 gpc steering
reward, the fallen-grace termination and the 20 s timeout are evaluated here
from the clean simulator state. Training and evaluation use separate
instances, so side rollouts never touch training episodes.
"""

from __future__ import annotations

import os

import torch

os.environ.setdefault("MJLAB_WARP_QUIET", "1")

import mujoco  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.nan_guard import NanGuard  # noqa: E402

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1  # noqa: E402
from super_gpc.rl.cuda_graph import CudaGraphed  # noqa: E402
from super_gpc.tasks.steering import mdp as M  # noqa: E402
from super_gpc.tasks.steering.rl.cola_cfg import ColaCfg  # noqa: E402
from super_gpc.tasks.steering.steering_env_cfg import (  # noqa: E402
  CONTACT_SENSOR,
  dollhouse_g1_steering_env_cfg,
)
from super_gpc.tasks.tracking.mdp import observations as O  # noqa: E402
from super_gpc.tasks.tracking.mdp.kinematics import Kinematics  # noqa: E402
from super_gpc.tasks.tracking.mdp.motion_library import POLICY_HZ  # noqa: E402
from super_gpc.tasks.tracking.rl.real_env import capture_set_const  # noqa: E402

PRIVILEGED_DIM = 7
GAZE_BODIES = ("head_tilt_link", "left_ankle_roll_link", "right_ankle_roll_link")
ARM_JOINTS = tuple(
  n for n in G1.ACTION_JOINT_NAMES if "shoulder" in n or "elbow" in n or "wrist" in n
)
OFFCENTER_GAZE_RAD = 0.3  # look targets scored by idle/offcenter_look_error_rad


def forward_axis(q: torch.Tensor) -> torch.Tensor:
  """World x axis of WXYZ body orientations."""
  w, x, y, z = q.unbind(-1)
  return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), -1)


class SteeringEnv:
  def __init__(
    self,
    cfg: ColaCfg,
    num_envs: int,
    device: str,
    seed: int,
  ):
    self.cfg, self.device, self.num_envs = cfg, device, num_envs
    env_cfg = dollhouse_g1_steering_env_cfg(
      num_envs, cfg.episode_steps / POLICY_HZ, cfg.reset_joint_noise
    )
    self.env = ManagerBasedRlEnv(env_cfg, device=device)
    self.njmax = env_cfg.sim.njmax
    self.robot = self.env.scene["robot"]
    self.contact = self.env.scene[CONTACT_SENSOR]
    tree = list(self.robot.joint_names)
    self.a2t = torch.tensor([tree.index(n) for n in G1.ACTION_JOINT_NAMES], device=device)
    self.phys = self.env.event_manager.get_term_cfg("physics").func
    self.default_q = torch.tensor(
      [G1.DEFAULT_JOINT_POS[n] for n in G1.ACTION_JOINT_NAMES], device=device
    )
    self.arm = torch.tensor([G1.ACTION_JOINT_NAMES.index(n) for n in ARM_JOINTS], device=device)
    self.scale = torch.tensor([G1.ACTION_SCALE[n] for n in G1.ACTION_JOINT_NAMES], device=device)
    soft = self.robot.data.soft_joint_pos_limits[0][self.a2t]
    hard = self.robot.data.joint_pos_limits[0][self.a2t]
    self.soft_lo, self.soft_hi = soft[:, 0].clone(), soft[:, 1].clone()
    self.hard_lo, self.hard_hi = hard[:, 0].clone(), hard[:, 1].clone()
    self.noise = O.noise_scale(device)
    self.ctrl = self.robot.indexing.ctrl_ids
    self.commands = M.SteeringCommands(num_envs, device, cfg, seed)
    # Head and ankle poses from the clean generalized state (MJLab body
    # quantities lag one substep behind qpos/qvel).
    mj = mujoco.MjModel.from_xml_path(str(G1.mjcf_path()))
    self.kin = Kinematics(mj, G1.ACTION_JOINT_NAMES, GAZE_BODIES).to(device)
    self.bodies = CudaGraphed(self.kin)  # full-batch calls; eager kinematics launch ~700 kernels
    n = num_envs
    self.history = torch.zeros(n, O.HISTORY, O.OBS_DIM, device=device)
    self.prev_action = torch.zeros(n, O.ACTION_DIM, device=device)
    self.ep_len = torch.zeros(n, dtype=torch.long, device=device)
    self.ep_return = torch.zeros(n, device=device)
    self.fallen_steps = torch.zeros(n, dtype=torch.long, device=device)
    self.episode = torch.zeros(n, dtype=torch.long, device=device)
    self.next_episode_id = 0
    if device.startswith("cuda"):
      capture_set_const(self.env)

  @property
  def rng(self) -> torch.Generator:
    """The environment's own random stream, part of the trainer's native state."""
    return self.commands.gen

  # --- state and observations -------------------------------------------
  def state(self) -> torch.Tensor:
    return O.robot_generalized_state(self.robot, self.env.scene.env_origins, self.a2t)

  def _frame(self, s: torch.Tensor, prev_action: torch.Tensor, ids=None, noisy=True):
    bias = self.phys.context[:, 5 : 5 + G1.NUM_ACTIONS]
    bias = bias if ids is None else bias[ids]
    if not noisy:
      bias = torch.zeros_like(bias)
    # The neck follows the gaze analytically; the CoLA sees it at rest at the
    # pose its previous neck action commanded.
    s = M.mask_neck(s, prev_action, self.default_q, self.scale)
    clean = O.actor_observation(s, prev_action, bias, self.default_q)
    return O.add_sensor_noise(clean, self.noise) if noisy else clean

  def reset(self, ids: torch.Tensor | None = None) -> None:
    if ids is None:
      ids = torch.arange(self.num_envs, device=self.device)
    if len(ids) == 0:
      return
    self.env.reset(env_ids=ids)
    s = self.state()[ids]
    zero = torch.zeros(len(ids), O.ACTION_DIM, device=self.device)
    self.history[ids] = self._frame(s, zero, ids)[:, None].expand(-1, O.HISTORY, -1)
    self.prev_action[ids] = 0.0
    self.ep_len[ids] = 0
    self.ep_return[ids] = 0.0
    self.fallen_steps[ids] = 0
    self.episode[ids] = torch.arange(len(ids), device=self.device) + self.next_episode_id
    self.next_episode_id += len(ids)
    self.commands.sample(ids, M.yaw_wxyz(s[:, 3:7]), reset=True)

  def inputs(self) -> dict[str, torch.Tensor]:
    """Actor inputs (noisy frame, decoder history, command) and the critic's
    clean frame and privileged features, all at the current state."""
    s = self.state()
    yaw = M.yaw_wxyz(s[:, 3:7])
    privileged = torch.cat(
      (
        M.rotate_xy(s[:, 7:9], -yaw),
        s[:, 9:10],
        s[:, 2:3],
        self.fallen_steps[:, None] / self.cfg.fallen_grace_steps,
        self.commands.remaining[:, None] / self.cfg.idle_max_steps,
        self.commands.gaze_remaining[:, None] / self.cfg.gaze_max_steps,
      ),
      -1,
    )
    return dict(
      obs=self.history[:, -1],
      history=self.history.reshape(self.num_envs, -1),
      command=self.commands.local(yaw),
      clean=self._frame(s, self.prev_action, noisy=False),
      privileged=privileged,
    )

  # --- policy step --------------------------------------------------------
  def step(self, action: torch.Tensor) -> dict:
    """Advance one policy step with the executed normalized ``action``.

    Returns the reward, ``terminated`` (physical end: no bootstrap), ``done``
    (terminated or timeout) and diagnostics. Does not reset; the caller
    reads the terminal inputs first and then calls ``finish``.
    """
    cfg = self.cfg
    s = self.state()
    target_velocity, target_facing, target_gaze = self.commands.world()
    idle = self.commands.idle()
    # The executed neck follows the gaze command; ``action`` (the decoder's,
    # including its neck) stays the CoLA's previous action.
    executed = action.clone()
    neck = M.neck_targets(
      s, self.commands.local(M.yaw_wxyz(s[:, 3:7])), cfg.neck_pan_max, cfg.neck_tilt_max
    )
    executed[:, M.NECK] = (neck - self.default_q[M.NECK]) / self.scale[M.NECK]
    tree_action = torch.empty_like(executed)
    tree_action[:, self.a2t] = executed
    self.env.step(tree_action)
    s_next = self.state()
    diverged = NanGuard.detect_nans(self.env.sim.data) | ~torch.isfinite(s_next).all(-1)
    if diverged.any():
      print(f"[sim] non-finite state in {int(diverged.sum())} env(s)", flush=True)
      s_next = torch.where(diverged[:, None], s, s_next)
    yaw = M.yaw_wxyz(s_next[:, 3:7])
    q = s_next[:, 13:40]
    force = self.contact.data.force.norm(dim=-1)
    contacts = (force > cfg.contact_threshold_n).float().sum(-1)
    rate = (action - self.prev_action) * self.scale  # target change (rad)
    bodies = self.bodies(s_next)
    head = forward_axis(bodies.quat[:, 0])
    gaze = self.commands.gaze.clone()
    arm_offset = (q - self.default_q)[:, self.arm]
    reward, terms = M.steering_reward(
      yaw,
      s_next[:, 7:9],
      target_velocity,
      target_facing,
      target_gaze,
      head,
      bodies.lin_vel[:, 1:],
      rate,
      q,
      self.soft_lo,
      self.soft_hi,
      contacts,
      arm_offset,
      cfg,
    )
    fallen = (s_next[:, 2] < cfg.fallen_height) | (
      M.uprightness_wxyz(s_next[:, 3:7]) < cfg.fallen_uprightness
    )
    self.fallen_steps = torch.where(fallen, self.fallen_steps + 1, 0)
    lim = cfg.joint_limit_terminate_rad
    beyond = ((q < self.hard_lo - lim) | (q > self.hard_hi + lim)).any(-1)
    terminated = (self.fallen_steps >= cfg.fallen_grace_steps) | beyond | diverged
    self.ep_len += 1
    self.ep_return += reward
    timeout = self.ep_len >= cfg.episode_steps
    done = terminated | timeout
    vel = s_next[:, 7:9]
    act_force = self.env.sim.data.actuator_force[:, self.ctrl]
    limit = self.env.sim.model.actuator_forcerange[:, self.ctrl, 1]
    if limit.dim() == 1:
      limit = limit[None]
    nan = torch.full_like(reward, float("nan"))
    gaze_error = torch.acos((head * target_gaze).sum(-1).clamp(-1, 1))
    head_yaw = torch.atan2(head[:, 1], head[:, 0]) - yaw
    head_yaw = torch.atan2(head_yaw.sin(), head_yaw.cos())
    idle_info = {  # NaN outside zero-speed commands; averaged over idle steps
      "idle/gaze_error_rad": gaze_error,
      "idle/gaze_target_yaw_rad": gaze[:, 0].abs(),
      "idle/head_yaw_from_pelvis_rad": head_yaw.abs(),
      "idle/offcenter_look_error_rad": torch.where(
        gaze[:, 0].abs() > OFFCENTER_GAZE_RAD, gaze_error, float("nan")
      ),
      "idle/arm_offset_rad": arm_offset.norm(dim=-1),
      "idle/head_pitch_rad": torch.asin(head[:, 2].clamp(-1, 1)),
      "idle/pelvis_speed_mps": s_next[:, 7:9].norm(dim=-1),
      "idle/ankle_speed_mps": bodies.lin_vel[:, 1:, :2].norm(dim=-1).sum(-1),
      "idle/pelvis_yaw_rate_rad_per_s": s_next[:, 12].abs(),
      "idle/pelvis_height_m": s_next[:, 2],
    }
    info = (
      {f"reward/{k}": v for k, v in terms.items()}
      | {k: torch.where(idle, v, nan) for k, v in idle_info.items()}
      | {
        "task/idle_fraction": idle.float(),
        "task/gaze_error_rad": gaze_error,
        "task/speed_error_mps": (vel.norm(dim=-1) - target_velocity.norm(dim=-1)).abs(),
        "task/velocity_error_mps": (vel - target_velocity).norm(dim=-1),
        "task/facing_error_rad": torch.acos(
          (torch.stack((yaw.cos(), yaw.sin()), -1) * target_facing).sum(-1).clamp(-1, 1)
        ),
        "task/fallen": fallen.float(),
        "task/terminated": terminated.float(),
        "task/nonfoot_contacts": contacts,
        "task/target_rate_rad_per_s": rate.norm(dim=-1) * POLICY_HZ,
        "real/actuator_saturation": (act_force.abs() >= 0.99 * limit).float().mean(-1),
        "real/action_abs_mean": action.abs().mean(-1),
        "real/pelvis_height_m": s_next[:, 2],
      }
    )
    self.history = torch.cat((self.history[:, 1:], self._frame(s_next, action)[:, None]), 1)
    self.prev_action = action.clone()
    self.commands.advance(yaw)  # score the old command first, as the source does
    nefc = self.env.sim.data.nefc
    return dict(
      reward=reward,
      terminated=terminated,
      timeout=timeout & ~terminated,
      done=done,
      info=info,
      max_nefc=nefc.max(),
      nefc_overflow=(nefc >= self.njmax).float().mean(),
    )

  def finish(self, done: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reset finished episodes; returns their lengths (s) and returns."""
    ids = done.nonzero(as_tuple=False).squeeze(-1)
    if len(ids) == 0:
      return {}
    out = dict(length_s=self.ep_len[ids].float() / POLICY_HZ, ret=self.ep_return[ids].clone())
    self.reset(ids)
    return out

  def fingerprint_tensors(self) -> dict[str, torch.Tensor]:
    """Everything a side rollout must leave untouched."""
    sim = self.env.sim
    return {
      "qpos": sim.data.qpos,
      "qvel": sim.data.qvel,
      "ctrl": sim.data.ctrl,
      "dr_fields": torch.cat(
        [getattr(sim.model, f).reshape(self.num_envs, -1) for f in self.phys.model_fields[:8]], -1
      ),
      "context": self.phys.context,
      "history": self.history,
      "prev_action": self.prev_action,
      "commands": torch.cat(
        (
          self.commands.anchor[:, None],
          self.commands.facing[:, None],
          self.commands.velocity,
          self.commands.remaining[:, None].float(),
          self.commands.gaze,
          self.commands.gaze_goal,
          self.commands.gaze_remaining[:, None].float(),
        ),
        -1,
      ),
      "command_rng": self.commands.gen.get_state(),
      "counters": torch.stack(
        (
          self.ep_len.float(),
          self.ep_return,
          self.fallen_steps.float(),
          self.episode.float(),
        )
      ),
      "mjlab_episode_length": self.env.episode_length_buf,
    }
