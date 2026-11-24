"""CoLA RLFT configuration.

``RlftCfg`` holds the adapter, PPO and run settings shared by every CoLA task;
``ColaCfg`` adds joystick steering. Values marked Source are the G1 gpc
steering settings (``SteeringConfig``, single-GPU recipe); values marked
Choice or Decision are ours, recorded in docs/spec.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RlftCfg:
  # --- Source: adapter and PPO ---------------------------------------------
  rank: int = 32
  alpha: float = 64.0
  kl_beta: float = 0.1
  num_envs: int = 4096
  rollout_steps: int = 16  # Decision: fast recipe; G1 gpc 32
  minibatch_size: int = 2048  # fast recipe; G1 gpc 4096
  mini_epochs: int = 2
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip_ratio: float = 0.2
  entropy_coefficient: float = 0.0
  actor_lr: float = 1e-3  # fast recipe; G1 gpc 1e-4
  critic_lr: float = 1e-4
  gradient_clip: float = 25.0
  target_update_kl: float = 0.02  # actor updates stop after a minibatch above 1.5x
  critic_width: int = 1024
  critic_layers: int = 4
  target_transitions: int = 500_000_000  # source budget; --iterations is the run budget
  eval_interval: int = 100
  eval_envs: int = 128
  eval_steps: int = 400
  # --- Choices --------------------------------------------------------------
  action_clip: float = 5.0  # the frozen decoder's Stage 1 clip
  checkpoint_interval: int = 100  # resume state + exported, published policy
  numbered_interval: int = 500  # retained native policy files
  seed: int = 1729
  eval_seed: int = 99173
  tf32: bool = True  # transformer matmuls in rollout and PPO alike; decoder/sim FP32
  export_workers: int = 1  # one export (~1 CPU min) per checkpoint interval
  # Throughput settings of the fast recipe (docs/spec.md).
  bf16_prior: bool = True  # frozen transformer/head Linears in BF16, rollout and PPO alike
  fused_base: bool = True  # base logits in the rollout's cached pass; PPO reuses its first block
  compile: bool = True  # torch.compile PPO's teacher-forced logits and the token statistics

  def validate(self) -> None:
    if (self.num_envs * self.rollout_steps) % self.minibatch_size:
      raise ValueError("rollout must split into whole PPO minibatches")

  @classmethod
  def from_dict(cls, d: dict):
    return cls(**d)

  def to_dict(self) -> dict:
    return asdict(self)


@dataclass
class ColaCfg(RlftCfg):
  # --- Source: commands ---------------------------------------------------
  speed_min: float = 0.2
  speed_max: float = 1.5
  stop_probability: float = 0.3  # Decision (idle gaze): G1 gpc uses 0.1
  command_min_steps: int = 100
  command_max_steps: int = 250
  facing_change_probability: float = 0.5
  # --- Source: reward -------------------------------------------------------
  facing_width: float = 1.0
  direction_width: float = 1.0
  speed_sigma: float = 0.5
  near_zero_speed: float = 0.1
  facing_weight: float = 1.0
  direction_weight: float = 1.0
  speed_weight: float = 1.0
  action_rate_weight: float = -0.1
  joint_limit_weight: float = -10.0
  contact_weight: float = -0.1
  contact_threshold_n: float = 1.0
  # --- Decision: idle gaze ----------------------------------------------------
  idle_min_steps: int = 150  # zero-speed commands last 3-10 s
  idle_max_steps: int = 500
  gaze_min_steps: int = 75  # a look target is held 1.5-4 s
  gaze_max_steps: int = 200
  gaze_center_probability: float = 0.25
  gaze_yaw_max: float = 1.0  # rad from the commanded facing
  gaze_pitch_min: float = -0.3  # rad, positive up
  gaze_pitch_max: float = 0.2
  gaze_weight: float = 1.0
  gaze_width: float = 0.25  # exp((cos(gaze error) - 1) / width)
  gaze_rate: float = 1.5  # rad/s the gaze command slews toward a new look target
  neck_pan_max: float = 1.5  # rad; analytic neck targets (joint range 2.62)
  neck_tilt_max: float = 0.6  # rad (joint range 0.785)
  still_weight: float = 1.0
  still_sigma: float = 0.1  # m/s
  foot_slip_weight: float = -1.0  # per m/s of summed planar ankle speed
  settle_facing_rad: float = 0.35
  arm_pose_weight: float = -0.05  # per rad^2 of arm offset from the default pose
  # --- Source: episodes -----------------------------------------------------
  episode_steps: int = 1000  # 20 s
  fallen_height: float = 0.45  # pelvis height
  fallen_uprightness: float = 0.6
  fallen_grace_steps: int = 150
  joint_limit_terminate_rad: float = 1.0  # beyond the physical range
  reset_joint_noise: float = 0.02

  @property
  def command_dim(self) -> int:
    """Local command width: velocity and facing, plus the gaze direction and pitch."""
    return 7
