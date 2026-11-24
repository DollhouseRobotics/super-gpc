"""Stage 2 prior-training configuration.

Values marked Spec are accepted decisions in the specification's Stage 2
section (our decisions, then pinned original-GPC code defaults); values marked
Choice are engineering choices recorded in docs/spec.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from super_gpc.rl.prior import D_MODEL, FF_SIZE


@dataclass
class Stage2Cfg:
  # --- Spec: architecture -------------------------------------------------
  d_model: int = D_MODEL  # the only accepted width (feed-forward FF_SIZE)
  num_layers: int = 3
  num_heads: int = 8
  dropout: float = 0.1
  norm_ema_decay: float = 0.999  # observation statistics, not model weights
  norm_clamp: float = 5.0
  # --- Spec: optimization ---------------------------------------------------
  lr: float = 1e-4  # constant, no warm-up, not scaled with GPU count
  betas: tuple[float, float] = (0.9, 0.999)
  eps: float = 1e-8
  weight_decay: float = 0.01
  label_smoothing: float = 0.01
  grad_clip_norm: float = 100.0
  # --- Spec: collection and limits --------------------------------------------
  num_envs: int = 1024  # per GPU
  batch_size: int = 1024  # examples per GPU minibatch
  rollout_steps: int = 64  # fresh real expert rollout per iteration
  mini_epochs: int = 1  # one shuffled pass
  action_noise_std: float = 1e-4  # executed collection noise, normalized units
  eval_interval: int = 200  # iterations
  latest_interval: int = 1000  # resumable + latest inference state (GPC-code 10)
  checkpoint_interval: int = 1000  # numbered policy checkpoint, exported and published
  max_transitions: int = 10_000_000_000_000  # GPC launcher ceiling, never a budget
  top_p: float = 0.9
  temperature: float = 1.0
  episode_seconds: float = 10.0  # Stage 1 task contract
  gamma: float = 0.99  # Stage 1 episode-return bookkeeping only (no RL loss)
  # --- Choices --------------------------------------------------------------
  action_clip: float = 5.0  # must equal the teacher's
  eval_envs: int = 64  # half held-out validation clips, half training clips
  eval_steps: int = 500
  fall_height_m: float = 0.4  # generation failure: pelvis below this height
  log_interval: int = 1
  seed: int = 1
  eval_seed: int = 12345
  quality_interval: int = 1000  # unconditional quality evaluation (stage2_quality); 0 disables
  quality_seeds: int = 2  # seeds 0..n-1, same starts and push directions at every checkpoint
  example_clips: tuple[str, ...] = field(default_factory=tuple)  # Miniverse start poses
  demo_seeds: int = 8  # Miniverse sampler-seed choices
  # --- Choices: throughput (recipe unchanged; TF32/BF16 change only rounding) -
  cuda_graphs: bool = True  # bitwise-equal CUDA-graph replay of per-step env calculations
  compile_rewards: bool = True  # fuse the diagnostic-only real reward terms (torch.compile)
  tf32_prior: bool = True  # TF32 matmuls in prior optimization only; teacher, eval, export FP32
  bf16_linears: bool = True  # BF16 feed-forward, head and obs-encoder Linears in optimization only
  attention_math: bool = True  # direct exact-FP32 4-position attention instead of the SDPA kernel
  compile_prior: bool = True  # torch.compile the prior's training loss

  def validate(self) -> None:
    if self.d_model != D_MODEL:
      raise ValueError(f"the accepted Stage 2 prior has width {D_MODEL}")
    if (self.num_layers, self.num_heads) != (3, 8):
      raise ValueError("the accepted Stage 2 prior uses 3 layers and 8 heads")
    if self.temperature <= 0 or not 0 < self.top_p <= 1:
      raise ValueError("invalid nucleus settings")

  @property
  def ff_size(self) -> int:
    return FF_SIZE

  @classmethod
  def from_dict(cls, d: dict) -> Stage2Cfg:
    d = {k: v for k, v in d.items() if k != "ff_size"}
    d["betas"] = tuple(d["betas"])
    d["example_clips"] = tuple(d["example_clips"])
    return cls(**d)

  def to_dict(self) -> dict:
    return asdict(self) | {"ff_size": FF_SIZE}
