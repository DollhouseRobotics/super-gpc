"""Stage 1 tracker-training configuration.

Values in the first block are SuperMimic Table A.5 settings and decisions
recorded in the spec; the others are engineering choices recorded in
docs/spec.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Stage1Cfg:
  # --- Specification (SuperMimic Table A.5 and decisions) ---
  num_envs: int = 4096  # per GPU
  replay_capacity: int = 1000  # steps per env, FIFO
  initial_fill_steps: int = 64  # collection-only vector steps, no model-only warm-up
  model_window: int = 8
  model_batches: int = 10
  model_batch_size: int = 2048
  actor_batch: int = 4096
  horizon: int = 32
  critic_epochs: int = 5
  critic_minibatches: int = 4
  action_noise_std: float = 0.1
  utd: float = 0.4  # iterations per vector step -> 2 and 3 steps alternately
  lr: float = 4e-4  # CQ base rate; actor initially uses 4x this rate
  lr_floor_ratio: float = 0.03
  lr_decay_iterations: int = 250_000
  actor_betas: tuple[float, float] = (0.9, 0.95)
  grad_clip_norm: float = 1.0
  actor_grad_clip_value: float = 0.1
  gamma: float = 0.99
  lam: float = 0.95
  max_iterations: int = 320_000
  episode_seconds: float = 10.0
  # --- Engineering choices ---
  action_clip: float = 5.0  # normalized units, collection and imagination alike
  value_horizon_frames: int = 500  # critic clip-remaining feature saturation
  sampler_alpha: float = 0.25
  eval_interval: int = 500  # iterations
  eval_envs: int = 512  # half held-out validation clips, half training clips
  eval_steps: int = 500
  checkpoint_interval: int = 1000  # policy checkpoints, each published to Miniverse
  resume_interval: int = 1000  # native resumable state, overwritten in place
  log_interval: int = 10
  compile: bool = True  # torch.compile the imagined step (CPU smoke runs may disable)
  # Replay each model update and each actor+critic update from one CUDA graph
  # (needs compile and CUDA; same calculation without per-kernel host launches).
  cuda_graphs: bool = True
  bf16: bool = True  # bf16 autocast for the decoder and dynamics MLPs inside training updates
  export_workers: int = 2  # background checkpoint export/check processes; 0 exports inline
  seed: int = 1
  eval_seed: int = 12345
  example_clips: tuple[str, ...] = field(default_factory=tuple)  # Miniverse switcher
  # --- CQ recipe ---
  reference_offsets: tuple[int, ...] = tuple(range(0, 46, 5))
  reward_scale: float = 0.02
  reward_anchor: int = 0  # pelvis; encoder alignment stays torso
  contact_weight: float = 0.25  # imagination only, before reward_scale
  policy_ema_decay: float = 0.995
  actor_lr_multiplier: float = 4.0
  actor_lr_trigger: float = 0.75
  actor_lr_decay_steps: int = 300
  actor_lr_decay_by: int = 1400
  recent_failure_share: float = 0.30
  probe_share: float = 0.15
  probe_start: int = 1000
  probe_every: int = 125

  def steps_for_iteration(self, iteration: int) -> int:
    return int((iteration + 1) / self.utd + 1e-9) - int(iteration / self.utd + 1e-9)

  def lr_at(self, iteration: int, base: float | None = None) -> float:
    import math

    base = self.lr if base is None else base
    floor = self.lr_floor_ratio * base
    x = min(iteration, self.lr_decay_iterations) / self.lr_decay_iterations
    return floor + (base - floor) * 0.5 * (1 + math.cos(math.pi * x))

  def to_dict(self) -> dict:
    return asdict(self)
