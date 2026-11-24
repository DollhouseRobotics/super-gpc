"""Motion command and SuperMimic return-based reset sampler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg

from super_gpc.tasks.tracking.mdp import kinematics as K
from super_gpc.tasks.tracking.mdp.motion_library import POLICY_HZ, MotionLibrary

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class ReturnSampler:
  """SuperMimic Appendix D.1 bin sampler.

  Engineering choices (unpublished in SM): a bin estimate is the normalized
  discounted return of episodes that started in it, divided by the largest
  return attainable over that episode's available horizon; estimates start at
  0 (unvisited bins count as hardest) and update by EMA with rate ``alpha``.
  The historical component accumulates the current-difficulty probabilities
  over updates. Optional Blink-style allocation mixes recent live-policy
  failure rates into that base, then mixes EMA-policy probe focus into the
  result. Shares 0.30/0.15 yield 0.595 base, 0.255 recent, 0.15 probe once
  both have failures. Empty focus distributions fall back to their input.

  Focus needs the library's ordered ``bin_clip`` identities. Frames supplied
  to record methods are clip-local 50 Hz frames, not packed-corpus offsets.
  The caller owns collection/probe timing and must exclude side rollouts from
  live statistics. None of these statistics change return units or history.
  """

  def __init__(
    self,
    num_bins: int,
    device: str,
    alpha: float = 0.25,
    temperature: float = 25.0,
    mix: tuple[float, float, float] = (0.5, 0.4, 0.1),
    *,
    recent_share: float = 0.0,
    probe_share: float = 0.0,
    bin_clip: torch.Tensor | None = None,
    focus_seed: int = 0,
  ):
    if not (0 <= recent_share <= 1 and 0 <= probe_share <= 1):
      raise ValueError("Focus shares must be in [0, 1]")
    if (recent_share or probe_share) and bin_clip is None:
      raise ValueError("Focus allocation requires MotionLibrary.bin_clip")
    self.estimate = torch.zeros(num_bins, device=device)
    self.visits = torch.zeros(num_bins, device=device)
    self.history = torch.zeros(num_bins, device=device)
    self.alpha, self.temperature, self.mix = alpha, temperature, mix
    self.device = device
    self.recent_share, self.probe_share = recent_share, probe_share
    self.recent_failure = torch.zeros_like(self.estimate)
    self.probe_counts = torch.zeros_like(self.estimate)
    self.focus_rng = torch.Generator(device=device).manual_seed(focus_seed)
    self._set_bin_clips(bin_clip)

  def _set_bin_clips(self, bin_clip: torch.Tensor | None) -> None:
    self.bin_clip = None
    if bin_clip is None:
      return
    clips = bin_clip.to(device=self.device, dtype=torch.long).clone()
    if clips.shape != self.estimate.shape or clips.numel() == 0:
      raise ValueError("bin_clip must have one identity per bin")
    ids, counts = torch.unique_consecutive(clips, return_counts=True)
    if not torch.equal(ids, torch.arange(ids.numel(), device=self.device)):
      raise ValueError("bin_clip must be ordered contiguous clip IDs starting at zero")
    self.bin_clip = clips
    self.clip_last_bin = counts.cumsum(0) - 1
    self.clip_first_bin = self.clip_last_bin - counts + 1
    bins = torch.arange(clips.numel(), device=self.device)
    self.previous_bin = torch.maximum(bins - 1, self.clip_first_bin[clips])
    self.next_bin = torch.minimum(bins + 1, self.clip_last_bin[clips])
    self.has_next_bin = self.next_bin != bins

  def _end_bins(self, clips: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    if self.bin_clip is None:
      raise ValueError("Failure recording requires MotionLibrary.bin_clip")
    if clips.ndim != 1 or frames.shape != clips.shape:
      raise ValueError("clips and frames must be matching 1-D tensors")
    clips = clips.to(device=self.device, dtype=torch.long)
    frames = frames.to(device=self.device, dtype=torch.long)
    if ((clips < 0) | (clips >= self.clip_first_bin.numel()) | (frames < 0)).any():
      raise ValueError("Invalid clip ID or negative clip-local frame")
    # A completion frame can lie just beyond the last valid start bin.
    return torch.minimum(
      self.clip_first_bin[clips] + frames // POLICY_HZ, self.clip_last_bin[clips]
    )

  def record_failures(
    self, clips: torch.Tensor, frames: torch.Tensor, failed: torch.Tensor
  ) -> None:
    """Record ALL live episode endings from one vector step, before resetting.

    ``failed`` is physical/nonfinite failure, false for clip ends/timeouts.
    Empty calls do not fade. Nonempty calls fade all rates by .999, then
    update hit bins with .9 old + .1 mean(failed), including successes.
    Duplicate bin hits are averaged (Blink's bin_fix semantics).
    """
    if not self.recent_share:
      return
    bins = self._end_bins(clips, frames)
    if failed.shape != bins.shape:
      raise ValueError("failed must match clips/frames")
    if bins.numel() == 0:
      return
    values = failed.to(device=self.device, dtype=self.estimate.dtype)
    total = torch.zeros_like(self.recent_failure).index_add_(0, bins, values)
    count = torch.zeros_like(total).index_add_(0, bins, torch.ones_like(values))
    hit = count > 0
    self.recent_failure.mul_(0.999)
    self.recent_failure[hit] = 0.9 * self.recent_failure[hit] + 0.1 * total[hit] / count[hit]

  def record_probes(self, clips: torch.Tensor, frames: torch.Tensor, failed: torch.Tensor) -> None:
    """Apply ONE completed EMA-policy sweep, with at most its first failure per probe.

    Counts fade by .7 per call, including failure-free/empty sweeps, then
    receive unit counts at failure bins. Successful probe frames may be -1;
    they are ignored. The sampler spreads counts over the failure bin and
    ten preceding bins of the same clip. The caller must supply actual EMA
    deterministic, noise/randomization-free probe outcomes, not live endings.
    """
    if not self.probe_share:
      return
    if clips.ndim != 1 or frames.shape != clips.shape or failed.shape != clips.shape:
      raise ValueError("clips, frames and failed must be matching 1-D tensors")
    clips = clips.to(device=self.device)
    frames = frames.to(device=self.device)
    failed = failed.to(device=self.device, dtype=torch.bool)
    bins = self._end_bins(clips[failed], frames[failed])
    self.probe_counts.mul_(0.7)
    self.probe_counts.index_add_(0, bins, torch.ones_like(bins, dtype=self.probe_counts.dtype))

  def _allocate_focus(
    self, bins: torch.Tensor, weights: torch.Tensor, share: float
  ) -> torch.Tensor:
    if weights.sum() <= 0:
      return bins
    focus = torch.multinomial(weights, bins.numel(), replacement=True, generator=self.focus_rng)
    choose = torch.rand(bins.shape, device=self.device, generator=self.focus_rng) < share
    return torch.where(choose, focus, bins)

  def current(self) -> torch.Tensor:
    return torch.softmax(-self.temperature * self.estimate, 0)

  def accumulate_history(self) -> None:
    self.history += self.current()

  def sample(self, n: int) -> torch.Tensor:
    u = torch.rand(n, device=self.device)
    uniform = torch.randint(0, self.estimate.numel(), (n,), device=self.device)
    hard = torch.multinomial(self.current(), n, replacement=True)
    hist_p = self.history if self.history.sum() > 0 else torch.ones_like(self.history)
    hist = torch.multinomial(hist_p / hist_p.sum(), n, replacement=True)
    a, b, _ = self.mix
    bins = torch.where(u < a, uniform, torch.where(u < a + b, hard, hist))
    if self.recent_share:
      recent = torch.zeros_like(self.recent_failure)
      recent.index_add_(0, self.previous_bin, self.recent_failure)
      bins = self._allocate_focus(bins, recent, self.recent_share)
    if self.probe_share:
      weights = self.probe_counts.clone()
      window = weights.clone()
      for _ in range(10):
        window = window[self.next_bin] * self.has_next_bin
        weights += window
      bins = self._allocate_focus(bins, weights, self.probe_share)
    return bins

  def update(self, bins: torch.Tensor, normalized_return: torch.Tensor) -> None:
    if bins.numel() == 0:
      return
    total = torch.zeros_like(self.estimate).index_add_(0, bins, normalized_return)
    count = torch.zeros_like(self.estimate).index_add_(0, bins, torch.ones_like(normalized_return))
    hit = count > 0
    mean = total[hit] / count[hit]
    self.estimate[hit] = torch.where(
      self.visits[hit] > 0, (1 - self.alpha) * self.estimate[hit] + self.alpha * mean, mean
    )
    self.visits += count

  def state_dict(self) -> dict:
    # Keep flat, same-device, 1-D tensors: side-rollout signatures concatenate
    # these values and existing checkpoint writers call .cpu() on each one.
    state = dict(estimate=self.estimate, visits=self.visits, history=self.history)
    if self.bin_clip is not None:
      state.update(
        focus_shares=torch.tensor(
          [self.recent_share, self.probe_share], device=self.device, dtype=torch.float64
        ),
        focus_bin_clip=self.bin_clip,
        recent_failure=self.recent_failure,
        probe_counts=self.probe_counts,
        focus_rng=self.focus_rng.get_state().to(self.device),
      )
    return state

  def load_state_dict(self, state: dict) -> None:
    if "focus_bin_clip" in state:
      clips = state["focus_bin_clip"]
      if self.bin_clip is not None and not torch.equal(self.bin_clip, clips.to(self.device)):
        raise ValueError("Checkpoint focus bin identities differ from the motion library")
      self._set_bin_clips(clips)
      self.recent_share, self.probe_share = state["focus_shares"].tolist()
      self.recent_failure.copy_(state["recent_failure"])
      self.probe_counts.copy_(state["probe_counts"])
      self.focus_rng.set_state(state["focus_rng"].cpu())
    for k in ("estimate", "visits", "history"):
      getattr(self, k).copy_(state[k])


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (0.0, 0.0)
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)


class MotionCommand(CommandTerm):
  """Owns per-env clip, frame clock and episode start bin.

  The library, sampler and action-order joint map are attached after
  environment construction (``attach``). On reset the term samples a start
  from the sampler (or an explicit assignment), writes the perturbed
  reference state and leaves the frame at the sampled start; per-step updates
  advance the frame by one. Clip completion and timeouts are decided by the
  trainer, which never lets a frame pass the clip end.
  """

  cfg: MotionCommandCfg

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.clip = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.frame = torch.zeros_like(self.clip)
    self.start_bin = torch.full_like(self.clip, -1)
    self.start_frame = torch.zeros_like(self.clip)
    self.library: MotionLibrary | None = None
    self.sampler: ReturnSampler | None = None
    self.assignment: tuple[torch.Tensor, torch.Tensor] | None = None
    self.action_to_tree: torch.Tensor | None = None

  def attach(
    self, library: MotionLibrary, sampler: ReturnSampler | None, action_to_tree: torch.Tensor
  ) -> None:
    self.library, self.sampler, self.action_to_tree = library, sampler, action_to_tree

  @property
  def command(self) -> torch.Tensor:
    return self.frame[:, None].float()

  def _update_metrics(self) -> None:
    pass

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:
      self.frame += 1

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.library is None:
      return  # startup reset before the library is attached
    lib = self.library
    n = len(env_ids)
    if self.assignment is not None:
      clip, frame = self.assignment[0][env_ids], self.assignment[1][env_ids]
      bins = torch.full_like(clip, -1)
    else:
      assert self.sampler is not None
      bins = self.sampler.sample(n)
      clip, frame = lib.sample_start(bins)
    self.clip[env_ids], self.frame[env_ids] = clip, frame
    self.start_bin[env_ids], self.start_frame[env_ids] = bins, frame
    ref = lib.state(clip, frame)
    write_start_state(
      self.robot,
      self._env.scene.env_origins,
      env_ids,
      ref,
      self.cfg.pose_range,
      self.cfg.velocity_range,
      self.cfg.joint_position_range,
      self.action_to_tree,
    )


def write_start_state(
  robot: Entity,
  env_origins: torch.Tensor,
  env_ids: torch.Tensor,
  ref: torch.Tensor,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]],
  joint_position_range: tuple[float, float],
  action_to_tree: torch.Tensor,
) -> None:
  """Write env-local generalized states ``ref`` (N, 67), perturbed by uniform
  root pose, root velocity (world linear, world angular) and joint offsets,
  with joints clamped to their limits."""
  n, device = len(env_ids), ref.device

  def ranges(spec, keys):
    r = torch.tensor([spec.get(k, (0.0, 0.0)) for k in keys], device=device)
    return r[:, 0] + torch.rand(n, len(keys), device=device) * (r[:, 1] - r[:, 0])

  keys = ("x", "y", "z", "roll", "pitch", "yaw")
  dp = ranges(pose_range, keys)
  dv = ranges(velocity_range, keys)
  pos = ref[:, 0:3] + dp[:, :3] + env_origins[env_ids]
  roll, pitch, yaw = dp[:, 3], dp[:, 4], dp[:, 5]
  delta = K.quat_mul(
    K.quat_from_rotvec(torch.stack((torch.zeros_like(yaw),) * 2 + (yaw,), -1)),
    K.quat_mul(
      K.quat_from_rotvec(
        torch.stack((torch.zeros_like(pitch), pitch, torch.zeros_like(pitch)), -1)
      ),
      K.quat_from_rotvec(torch.stack((roll, torch.zeros_like(roll), torch.zeros_like(roll)), -1)),
    ),
  )
  quat = K.normalize(K.quat_mul(delta, ref[:, 3:7]))
  lin = ref[:, 7:10] + dv[:, :3]
  ang_w = K.quat_rotate(ref[:, 3:7], ref[:, 10:13]) + dv[:, 3:]
  joint = ref[:, 13:40] + torch.empty_like(ref[:, 13:40]).uniform_(*joint_position_range)
  limits = robot.data.joint_pos_limits[env_ids][:, action_to_tree]
  joint = torch.clamp(joint, limits[..., 0], limits[..., 1])
  q_tree = torch.empty_like(joint)
  v_tree = torch.empty_like(joint)
  q_tree[:, action_to_tree] = joint
  v_tree[:, action_to_tree] = ref[:, 40:67]
  robot.write_joint_state_to_sim(q_tree, v_tree, env_ids=env_ids)
  robot.write_root_state_to_sim(torch.cat((pos, quat, lin, ang_w), -1), env_ids=env_ids)
  robot.reset(env_ids=env_ids)


RESET_POSE_RANGE = {
  "x": (-0.05, 0.05),
  "y": (-0.05, 0.05),
  "z": (-0.01, 0.01),
  "roll": (-0.1, 0.1),
  "pitch": (-0.1, 0.1),
  "yaw": (-0.2, 0.2),
}
RESET_VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def max_discounted_return(
  steps: torch.Tensor, gamma: float, max_reward: float = 5.0
) -> torch.Tensor:
  return max_reward * (1 - gamma ** steps.float()) / (1 - gamma)
