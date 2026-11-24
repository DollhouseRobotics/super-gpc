"""MJLab get-down collection with trainer-owned episode bookkeeping.

Keeps the Stage 1 sensing contract the frozen prior and decoder were trained
on (noisy 87-scalar frame with calibration bias, executed previous action,
ten-frame decoder history padded with the first frame at reset) and the
Stage 1 physical randomization. The CoLA adapter's task input is the newest
five clean task frames (``mdp.task_frame``), padded the same way at reset.
It is returned under the ``command`` key the shared CoLA trainer reads.
Reward, terminations and the timeout are evaluated here from the clean
simulator state: the impact, action-rate and joint-limit terms plus
the pelvis-height reward. ``step`` also returns ``amp``, the (pre-step,
post-step) AMP frames of the same episode; the discriminator and its style
reward belong to the trainer. Training and evaluation use separate instances.
"""

from __future__ import annotations

import os

import torch

os.environ.setdefault("MJLAB_WARP_QUIET", "1")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.nan_guard import NanGuard  # noqa: E402

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1  # noqa: E402
from super_gpc.rl.cuda_graph import CudaGraphed  # noqa: E402
from super_gpc.tasks.get_down import mdp as M  # noqa: E402
from super_gpc.tasks.get_down.get_down_env_cfg import (  # noqa: E402
  FLOOR_SENSOR,
  dollhouse_g1_get_down_env_cfg,
)
from super_gpc.tasks.get_down.rl.get_down_cfg import GetDownCfg  # noqa: E402
from super_gpc.tasks.tracking.mdp import observations as O  # noqa: E402
from super_gpc.tasks.tracking.mdp.motion_library import POLICY_HZ, MotionLibrary  # noqa: E402
from super_gpc.tasks.tracking.rl.real_env import capture_set_const  # noqa: E402

PRIVILEGED_DIM = 9  # episode clock, torso forward-z, torso up-z, estimated features (6)


class GetDownEnv:
  def __init__(
    self,
    cfg: GetDownCfg,
    num_envs: int,
    device: str,
    seed: int,
    library: MotionLibrary,
    bank: torch.Tensor,
  ):
    self.cfg, self.device, self.num_envs = cfg, device, num_envs
    env_cfg = dollhouse_g1_get_down_env_cfg(num_envs, cfg.episode_steps / POLICY_HZ)
    self.env = ManagerBasedRlEnv(env_cfg, device=device)
    self.njmax = env_cfg.sim.njmax
    self.robot = self.env.scene["robot"]
    self.floor = self.env.scene[FLOOR_SENSOR]
    names = list(self.floor.primary_names)
    self.torso_ids = torch.tensor([names.index(b) for b in M.TORSO_CONTACT_BODIES], device=device)
    self.wrist_ids = torch.tensor([names.index(b) for b in M.WRIST_CONTACT_BODIES], device=device)
    tree = list(self.robot.joint_names)
    self.a2t = torch.tensor([tree.index(n) for n in G1.ACTION_JOINT_NAMES], device=device)
    self.phys = self.env.event_manager.get_term_cfg("physics").func
    self.starts: M.GetDownStarts = self.env.event_manager.get_term_cfg("start").func
    self.starts.attach(library, bank, self.a2t, cfg.joystick_fraction, seed)
    self.kin = M.build_kinematics(device)
    self.amp_kin = M.build_kinematics(device, M.AMP_BODIES)
    # Full-batch calls (inputs, step) replay a graph; resets of a few envs run eagerly.
    self.batch_kin = CudaGraphed(self.kin)
    self.batch_amp_kin = CudaGraphed(self.amp_kin)
    self.default_q = torch.tensor(
      [G1.DEFAULT_JOINT_POS[n] for n in G1.ACTION_JOINT_NAMES], device=device
    )
    self.scale = torch.tensor([G1.ACTION_SCALE[n] for n in G1.ACTION_JOINT_NAMES], device=device)
    soft = self.robot.data.soft_joint_pos_limits[0][self.a2t]
    hard = self.robot.data.joint_pos_limits[0][self.a2t]
    self.soft_lo, self.soft_hi = soft[:, 0].clone(), soft[:, 1].clone()
    self.hard_lo, self.hard_hi = hard[:, 0].clone(), hard[:, 1].clone()
    self.noise = O.noise_scale(device)
    self.ctrl = self.robot.indexing.ctrl_ids
    self.impact_progress = 1.0  # set by the trainer before each collection
    n = num_envs
    self.history = torch.zeros(n, O.HISTORY, O.OBS_DIM, device=device)
    self.task = torch.zeros(n, M.TASK_FRAMES, M.TASK_FRAME_DIM, device=device)
    self.amp_prev = torch.zeros(n, M.AMP_FRAME_DIM, device=device)
    self.prev_action = torch.zeros(n, O.ACTION_DIM, device=device)
    self.ep_len = torch.zeros(n, dtype=torch.long, device=device)
    self.fresh = torch.zeros(n, dtype=torch.bool, device=device)  # no step since reset
    self.ep_return = torch.zeros(n, device=device)
    self.episode = torch.zeros(n, dtype=torch.long, device=device)
    self.next_episode_id = 0
    if device.startswith("cuda"):
      capture_set_const(self.env)

  @property
  def rng(self) -> torch.Generator:
    return self.starts.gen

  # --- state and observations -------------------------------------------
  def state(self) -> torch.Tensor:
    return O.robot_generalized_state(self.robot, self.env.scene.env_origins, self.a2t)

  def _frame(self, s: torch.Tensor, prev_action: torch.Tensor, ids=None, noisy=True):
    bias = self.phys.context[:, 5 : 5 + G1.NUM_ACTIONS]
    bias = bias if ids is None else bias[ids]
    if not noisy:
      bias = torch.zeros_like(bias)
    clean = O.actor_observation(s, prev_action, bias, self.default_q)
    return O.add_sensor_noise(clean, self.noise) if noisy else clean

  def floor_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-body peak net floor force over the step's substeps (N, B) and the
    peak total floor force (N): vector sum over bodies, largest substep."""
    f = self.floor.data.force_history  # (N, B, substeps, 3)
    return f.norm(dim=-1).amax(-1), f.sum(1).norm(dim=-1).amax(-1)

  def reset(self, ids: torch.Tensor | None = None) -> None:
    if ids is None:
      ids = torch.arange(self.num_envs, device=self.device)
    if len(ids) == 0:
      return
    self.env.reset(env_ids=ids)
    s = self.state()[ids]
    zero = torch.zeros(len(ids), O.ACTION_DIM, device=self.device)
    self.history[ids] = self._frame(s, zero, ids)[:, None].expand(-1, O.HISTORY, -1)
    frame = M.task_frame(self.kin(s), s)
    self.task[ids] = frame[:, None].expand(-1, M.TASK_FRAMES, -1)
    self.amp_prev[ids] = M.amp_frame(self.amp_kin(s, velocities=False), s)
    self.prev_action[ids] = 0.0
    self.ep_len[ids] = 0
    self.fresh[ids] = True
    self.ep_return[ids] = 0.0
    self.episode[ids] = torch.arange(len(ids), device=self.device) + self.next_episode_id
    self.next_episode_id += len(ids)

  def stagger(self) -> None:
    """Spread the episode clocks uniformly over the episode after a reset of
    every environment, so timeouts (nearly the only episode end) do not reset
    all environments in the same step. The first episodes are shortened."""
    n, steps = self.num_envs, self.cfg.episode_steps
    self.ep_len[:] = torch.randint(0, steps, (n,), device=self.device, generator=self.rng)

  def inputs(self) -> dict[str, torch.Tensor]:
    """Actor inputs (noisy frame, decoder history, task input) and the
    critic's clean frame and privileged features at the current state."""
    s = self.state()
    body = self.batch_kin(s)
    up = 1 - 2 * s[:, 4:6].square().sum(-1)  # pelvis uprightness
    privileged = torch.cat(
      (
        torch.stack((self.ep_len / self.cfg.episode_steps, M.torso_forward_z(body), up), -1),
        M.estimated_features(body, s),
      ),
      -1,
    )
    return dict(
      obs=self.history[:, -1],
      history=self.history.reshape(self.num_envs, -1),
      command=self.task.reshape(self.num_envs, -1),
      clean=self._frame(s, self.prev_action, noisy=False),
      privileged=privileged,
    )

  # --- policy step --------------------------------------------------------
  def step(self, action: torch.Tensor) -> dict:
    """Advance one policy step with the executed normalized ``action``.

    Returns the reward, the AMP transition ``amp``, ``terminated`` (joint far beyond its range or a
    diverged simulation: no bootstrap), ``done`` and diagnostics. Does not
    reset; the caller reads the terminal inputs first and then calls
    ``finish``.
    """
    cfg = self.cfg
    s = self.state()
    tree_action = torch.empty_like(action)
    tree_action[:, self.a2t] = action
    self.env.step(tree_action)
    s_next = self.state()
    diverged = NanGuard.detect_nans(self.env.sim.data) | ~torch.isfinite(s_next).all(-1)
    if diverged.any():
      print(f"[sim] non-finite state in {int(diverged.sum())} env(s)", flush=True)
      s_next = torch.where(diverged[:, None], s, s_next)
    body = self.batch_kin(s_next)
    q = s_next[:, 13:40]
    per_body, total = self.floor_forces()
    rate = (action - self.prev_action) * self.scale  # target change (rad)
    impact = M.impact_weights(cfg, self.impact_progress)
    reward, terms = M.impact_reward(
      per_body,
      self.torso_ids,
      self.wrist_ids,
      rate,
      q,
      self.soft_lo,
      self.soft_hi,
      impact,
      ~self.fresh,
      cfg,
    )
    amp_body = self.batch_amp_kin(s_next, velocities=False)
    up = M.chest_up(amp_body)
    terms["pelvis_height"] = M.pelvis_height_reward(s_next[:, 2], up, cfg)
    reward = reward + terms["pelvis_height"]
    frame = M.amp_frame(amp_body, s_next)
    amp = torch.cat((self.amp_prev, frame), -1)
    self.amp_prev = frame
    lim = cfg.joint_limit_terminate_rad
    beyond = ((q < self.hard_lo - lim) | (q > self.hard_hi + lim)).any(-1)
    terminated = beyond | diverged
    self.ep_len += 1
    self.fresh[:] = False
    self.ep_return += reward
    timeout = self.ep_len >= cfg.episode_steps
    done = terminated | timeout
    act_force = self.env.sim.data.actuator_force[:, self.ctrl]
    limit = self.env.sim.model.actuator_forcerange[:, self.ctrl, 1]
    if limit.dim() == 1:
      limit = limit[None]
    forward_z = M.torso_forward_z(body)
    info = {f"reward/{k}": v for k, v in terms.items()} | {
      "task/torso_height_m": body.pos[:, M.TORSO, 2],
      "task/torso_horizontal": forward_z.abs(),
      "task/lying": M.lying(body, cfg).float(),
      "task/on_back": M.torso_on_back(amp_body, cfg).float(),
      "task/chest_up": up,
      "task/pelvis_down_speed_mps": (-s_next[:, 9]).clamp_min(0),
      "task/pelvis_speed_mps": s_next[:, 7:10].norm(dim=-1),
      "task/torso_spin_rad_s": body.ang_vel[:, M.TORSO].norm(dim=-1),
      "task/peak_floor_force_n": total,
      "task/peak_body_force_n": per_body.amax(-1),
      "task/peak_torso_force_n": per_body[:, self.torso_ids].amax(-1),
      "task/peak_wrist_force_n": per_body[:, self.wrist_ids].amax(-1),
      "task/terminated": terminated.float(),
      "task/target_rate_rad_per_s": rate.norm(dim=-1) * POLICY_HZ,
      "real/actuator_saturation": (act_force.abs() >= 0.99 * limit).float().mean(-1),
      "real/action_abs_mean": action.abs().mean(-1),
      "real/pelvis_height_m": s_next[:, 2],
    }
    self.history = torch.cat((self.history[:, 1:], self._frame(s_next, action)[:, None]), 1)
    frame = M.task_frame(body, s_next)
    self.task = torch.cat((self.task[:, 1:], frame[:, None]), 1)
    self.prev_action = action.clone()
    nefc = self.env.sim.data.nefc
    return dict(
      reward=reward,
      amp=amp,
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
      "task": self.task,
      "amp_prev": self.amp_prev,
      "prev_action": self.prev_action,
      "start_source": self.starts.source,
      "start_rng": self.starts.gen.get_state(),
      "floor_force": self.floor.data.force_history,
      "counters": torch.stack(
        (self.ep_len.float(), self.ep_return, self.episode.float(), self.fresh.float())
      ),
      "mjlab_episode_length": self.env.episode_length_buf,
    }
