"""Real MJLab simulator collection with trainer-owned episode bookkeeping.

``RealEnv`` wraps one ``ManagerBasedRlEnv`` and keeps what the selected
contract needs outside MJLab's managers: the clean generalized state, noisy
decoder observation history with reset padding, the executed previous
action, reference windows, rewards, SuperMimic failures and the distinct
timeout and clip-completion boundaries. Training and evaluation use separate
instances, so side rollouts never touch training episodes.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Sequence

import torch

os.environ.setdefault("MJLAB_WARP_QUIET", "1")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.nan_guard import NanGuard  # noqa: E402

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1  # noqa: E402
from super_gpc.tasks.tracking.config.dollhouse_g1.env_cfgs import (  # noqa: E402
  dollhouse_g1_tracking_env_cfg,
)
from super_gpc.tasks.tracking.mdp import observations as O  # noqa: E402
from super_gpc.tasks.tracking.mdp import rewards as R  # noqa: E402
from super_gpc.tasks.tracking.mdp.commands import (  # noqa: E402
  MotionCommand,
  ReturnSampler,
  max_discounted_return,
)
from super_gpc.tasks.tracking.mdp.events import (  # noqa: E402
  PhysicsRandomization,
)
from super_gpc.tasks.tracking.mdp.kinematics import Kinematics  # noqa: E402
from super_gpc.tasks.tracking.mdp.motion_library import MotionLibrary  # noqa: E402

FUTURE_OFFSETS = O.DEFAULT_REFERENCE_OFFSETS


class GraphedFn:
  """Replay a pure, RNG-free tensor function from CUDA graphs, one per input
  shape. The captured kernels are the eager ones, so results are bitwise
  equal; only the per-kernel host launch cost disappears. Outputs are cloned
  from the graph's static buffers."""

  def __init__(self, fn):
    self.fn = fn
    self.graphs: dict = {}

  def __call__(self, *args: torch.Tensor):
    key = tuple((a.shape, a.dtype) for a in args)
    entry = self.graphs.get(key)
    if entry is None:
      static = [a.clone() for a in args]
      stream = torch.cuda.Stream()
      stream.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(stream):
        for _ in range(2):  # warm-up outside capture, as torch.cuda.graph requires
          self.fn(*static)
      torch.cuda.current_stream().wait_stream(stream)
      graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):
        out = self.fn(*static)
      entry = self.graphs[key] = (graph, static, out)
    graph, static, out = entry
    for s, a in zip(static, args, strict=True):
      s.copy_(a)
    graph.replay()
    return torch.utils._pytree.tree_map(torch.clone, out)


def capture_set_const(env: ManagerBasedRlEnv) -> None:
  """Replay MuJoCo Warp's ``set_const`` from one CUDA graph.

  Physics randomization recomputes derived model constants on every reset,
  which issues about 900 un-graphed Warp launches. The graph holds the same
  kernels over the same (never replaced) model and data arrays, mirroring how
  MJLab captures ``step``/``forward``/``reset``. Other levels run as before.
  """
  import mujoco_warp as mjwarp
  import warp as wp
  from mjlab.managers.event_manager import RecomputeLevel

  sim = env.sim
  if not sim.use_cuda_graph:
    return
  gc_enabled = gc.isenabled()
  gc.disable()  # as MJLab: no stale-graph destructors recorded into the capture
  try:
    with wp.ScopedDevice(sim.wp_device), wp.ScopedCapture() as capture:
      mjwarp.set_const(sim.wp_model, sim.wp_data)
  finally:
    if gc_enabled:
      gc.enable()
  graph, eager = capture.graph, sim.recompute_constants

  def recompute(level: RecomputeLevel) -> None:
    if level != RecomputeLevel.set_const:
      return eager(level)
    with wp.ScopedDevice(sim.wp_device):
      wp.capture_launch(graph)

  sim.recompute_constants = recompute


_OFFSETS: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}


def reference_window(
  lib: MotionLibrary,
  clip: torch.Tensor,
  frame: torch.Tensor,
  reference_offsets: Sequence[int] = FUTURE_OFFSETS,
):
  """Current alignment anchor and ten conditioning samples, padded past clip end.

  The physics reward/failure target is always frame +1, not the first sample.
  """
  reference_offsets = O.validate_reference_offsets(reference_offsets)
  key = (str(frame.device), reference_offsets)
  if key not in _OFFSETS:  # built once: no host copy inside captured updates
    _OFFSETS[key] = torch.tensor(reference_offsets, device=frame.device)
  offsets = _OFFSETS[key]
  return lib.state(clip, frame), lib.state(clip[..., None], frame[..., None] + offsets)


class RealEnv:
  def __init__(
    self,
    num_envs: int,
    device: str,
    library: MotionLibrary,
    sampler: ReturnSampler | None,
    kin: Kinematics,
    kin_torso: Kinematics,
    gamma: float,
    episode_steps: int,
    reset_perturbation: bool = True,
    nominal_physics: bool = False,
    cuda_graphs: bool = False,
    compile_rewards: bool = False,
    reference_offsets: Sequence[int] = FUTURE_OFFSETS,
    reward_scale: float = 1.0,
    reward_anchor: int = R.ANCHOR,
    sensor_noise: bool = True,
  ):
    """``cuda_graphs`` replays the pure per-step torch calculations (encoder
    features, forward kinematics and rewards) and the reset-time ``set_const`` from CUDA
    graphs: bitwise-equal results with far fewer host kernel launches.
    ``compile_rewards`` also fuses the reward calculation with
    ``torch.compile`` (equal to eager up to FP32 rounding). Use it only where
    rewards are diagnostics, as in Stage 2; failures and observations never
    pass through it."""
    self.reference_offsets = O.validate_reference_offsets(reference_offsets)
    cfg = dollhouse_g1_tracking_env_cfg(
      num_envs, play=nominal_physics, reset_perturbation=reset_perturbation
    )
    self.reward_scale, self.reward_anchor = reward_scale, reward_anchor
    self.env = ManagerBasedRlEnv(cfg, device=device)
    self.njmax = cfg.sim.njmax
    self.device, self.num_envs = device, num_envs
    self.robot = self.env.scene["robot"]
    tree = list(self.robot.joint_names)
    self.a2t = torch.tensor([tree.index(n) for n in G1.ACTION_JOINT_NAMES], device=device)
    self.cmd: MotionCommand = self.env.command_manager.get_term("motion")  # type: ignore[assignment]
    self.cmd.attach(library, sampler, self.a2t)
    phys = self.env.event_manager.get_term_cfg("physics").func
    assert isinstance(phys, PhysicsRandomization)
    self.phys = phys
    self.lib, self.sampler = library, sampler
    self.kin, self.kin_torso = kin, kin_torso
    self.gamma, self.episode_steps = gamma, episode_steps
    self.default_q = torch.tensor(
      [G1.DEFAULT_JOINT_POS[n] for n in G1.ACTION_JOINT_NAMES], device=device
    )
    soft = self.robot.data.soft_joint_pos_limits[0][self.a2t]
    self.soft_lo, self.soft_hi = soft[:, 0].clone(), soft[:, 1].clone()
    self.noise = O.noise_scale(device)
    if not sensor_noise:
      self.noise.zero_()
    ctrl = self.robot.indexing.ctrl_ids
    self.ctrl = ctrl
    n = num_envs
    self.history = torch.zeros(n, O.HISTORY, O.OBS_DIM, device=device)
    self.prev_action = torch.zeros(n, O.ACTION_DIM, device=device)
    self.ep_len = torch.zeros(n, dtype=torch.long, device=device)
    self.ep_return = torch.zeros(n, device=device)
    self.ep_disc_return = torch.zeros(n, device=device)
    self.ep_diagnostic_return = torch.zeros(n, device=device)
    self.episode = torch.zeros(n, dtype=torch.long, device=device)
    self.next_episode_id = 0
    self.done = torch.zeros(n, dtype=torch.bool, device=device)
    self._features = lambda s, now, fut: O.encoder_features(self.kin_torso, s, now, fut)
    self._outcome = self._step_outcome
    if compile_rewards:
      self._outcome = torch.compile(self._step_outcome, dynamic=False)
    if cuda_graphs and device.startswith("cuda"):
      self._features = GraphedFn(self._features)
      self._outcome = GraphedFn(self._outcome)
      capture_set_const(self.env)

  # --- state and observations -------------------------------------------
  def state(self) -> torch.Tensor:
    return O.robot_generalized_state(self.robot, self.env.scene.env_origins, self.a2t)

  @property
  def context(self) -> torch.Tensor:
    return self.phys.context

  def observe(self, state: torch.Tensor, prev_action: torch.Tensor, ids=None) -> torch.Tensor:
    ctx = self.context if ids is None else self.context[ids]
    clean = O.actor_observation(state, prev_action, ctx[:, 5 : 5 + G1.NUM_ACTIONS], self.default_q)
    return O.add_sensor_noise(clean, self.noise)

  def reset(self, env_ids: torch.Tensor | None = None, assignment=None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if len(env_ids) == 0:
      return
    self.cmd.assignment = assignment
    self.env.reset(env_ids=env_ids)
    s = self.state()[env_ids]
    zero = torch.zeros(len(env_ids), O.ACTION_DIM, device=self.device)
    obs = self.observe(s, zero, env_ids)
    self.history[env_ids] = obs[:, None].expand(-1, O.HISTORY, -1)
    self.prev_action[env_ids] = 0.0
    self.ep_len[env_ids] = 0
    self.ep_return[env_ids] = 0.0
    self.ep_disc_return[env_ids] = 0.0
    self.ep_diagnostic_return[env_ids] = 0.0
    self.done[env_ids] = False
    ids = torch.arange(len(env_ids), device=self.device) + self.next_episode_id
    self.episode[env_ids] = ids
    self.next_episode_id += len(env_ids)

  # --- policy step --------------------------------------------------------
  def policy_inputs(self):
    s = self.state()
    ref_now, ref_future = reference_window(
      self.lib, self.cmd.clip, self.cmd.frame, self.reference_offsets
    )
    features = self._features(s, ref_now, ref_future)
    return s, features, self.history.reshape(self.num_envs, -1)

  def _step_outcome(self, s_next, ref_next, action, prev_action):
    """Pure post-physics reward calculation (forward kinematics and terms)."""
    robot_b, ref_b = self.kin(s_next), self.kin(ref_next)
    terms = R.tracking_reward(
      robot_b,
      ref_b,
      action,
      prev_action,
      s_next[:, 13:40],
      self.soft_lo,
      self.soft_hi,
      anchor=self.reward_anchor,
    )
    for key in (*R.TERMS, "total"):
      terms[key] = terms[key] * self.reward_scale
    if self.reward_anchor == R.ANCHOR:
      terms["diagnostic"] = terms["total"] / self.reward_scale
    else:
      terms["diagnostic"] = R.tracking_reward(
        robot_b, ref_b, action, prev_action, s_next[:, 13:40], self.soft_lo, self.soft_hi
      )["total"]
    return terms

  def step(self, action: torch.Tensor, auto_reset: bool = True) -> dict[str, torch.Tensor]:
    """Advance one policy step with the executed normalized ``action``.

    Returns the replay transition and per-env step diagnostics. With
    ``auto_reset`` finished episodes update the sampler and are reset.
    """
    s = self.state()
    clip, frame = self.cmd.clip.clone(), self.cmd.frame.clone()
    tree_action = torch.empty_like(action)
    tree_action[:, self.a2t] = action
    self.env.step(tree_action)
    s_next = self.state()
    # A diverged world (non-finite physics state) ends its episode as a
    # failure. Its replay transition keeps the finite pre-step state so NaN
    # never reaches replay, rewards or the sampler; the reset clears the world.
    diverged = NanGuard.detect_nans(self.env.sim.data) | ~torch.isfinite(s_next).all(-1)
    if diverged.any():
      ids = diverged.nonzero().squeeze(-1)
      print(
        f"[sim] non-finite state in {len(ids)} env(s): clips {self.cmd.clip[ids].tolist()[:8]} "
        f"frames {frame[ids].tolist()[:8]}",
        flush=True,
      )
      s_next = torch.where(diverged[:, None], s, s_next)
    frame_next = self.cmd.frame
    # MotionCommand advances exactly one policy frame, independently of the
    # encoder preview cadence (whose first conditioning sample may be zero).
    ref_next = self.lib.state(clip, frame_next)
    terms = self._outcome(s_next, ref_next, action, self.prev_action)
    failure, dz, dg = R.physical_failure(s_next, ref_next)
    failure = failure | diverged
    clip_end = frame_next >= self.lib.lengths[clip] - 1
    timeout = self.ep_len + 1 >= self.episode_steps
    reward = terms["total"]
    force = self.env.sim.data.actuator_force[:, self.ctrl]
    limit = self.env.sim.model.actuator_forcerange[:, self.ctrl, 1]
    if limit.dim() == 1:
      limit = limit[None]
    saturated = (force.abs() >= 0.99 * limit).float().mean(-1)
    transition = dict(
      state=s,
      next_state=s_next,
      obs=self.history[:, -1].clone(),
      action=action,
      context=self.context.clone(),
      clip=clip.int(),
      frame=frame.int(),
      episode=self.episode.int(),
      ep_step=self.ep_len.int(),
      failure=failure,
      clip_end=clip_end & ~failure,
      timeout=timeout & ~failure & ~clip_end,
    )
    self.ep_disc_return += (self.gamma ** self.ep_len.float()) * reward
    self.ep_diagnostic_return += (self.gamma ** self.ep_len.float()) * terms["diagnostic"]
    self.ep_return += reward
    self.ep_len += 1
    done = failure | clip_end | timeout
    nefc = self.env.sim.data.nefc
    info = dict(
      reward=reward,
      terms=terms,
      dz=dz,
      dg=dg,
      saturation=saturated,
      done=done,
      max_nefc=nefc.max(),
      nefc_overflow=(nefc >= self.njmax).float().mean(),
      contacts_per_world=self.env.sim.data.nacon[0] / self.num_envs,
      diverged=diverged.float().sum(),
    )
    obs = self.observe(s_next, action)
    self.history = torch.cat((self.history[:, 1:], obs[:, None]), 1)
    self.prev_action = action.clone()
    if auto_reset:
      info["episodes"] = self._finish(done, failure)
    else:
      self.done |= done
    return dict(transition=transition, info=info)

  def _finish(self, done: torch.Tensor, failure: torch.Tensor) -> dict[str, torch.Tensor]:
    ids = done.nonzero(as_tuple=False).squeeze(-1)
    out = {}
    if len(ids):
      start = self.cmd.start_frame[ids]
      available = torch.minimum(
        torch.full_like(start, self.episode_steps), self.lib.lengths[self.cmd.clip[ids]] - 1 - start
      ).clamp_min(1)
      normalized = self.ep_disc_return[ids] / (
        self.reward_scale * max_discounted_return(available, self.gamma)
      )
      out = dict(
        length=self.ep_len[ids].float(),
        ret=self.ep_return[ids],
        normalized=normalized,
        diagnostic_normalized=self.ep_diagnostic_return[ids]
        / max_discounted_return(available, self.gamma),
        available=available.float(),
      )
      if self.sampler is not None:
        if isinstance(self.sampler, ReturnSampler) and self.sampler.recent_share:
          self.sampler.record_failures(self.cmd.clip[ids], self.cmd.frame[ids], failure[ids])
        self.sampler.update(self.cmd.start_bin[ids], normalized)
      self.reset(ids)
    return out
