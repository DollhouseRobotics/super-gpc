"""Get-down CoLA RLFT configuration.

Adapter and PPO settings are the shared CoLA settings (``RlftCfg``) except
where overridden here. All values here are our choices, recorded in
docs/spec.md, except the AMP gradient-penalty weight.
"""

from __future__ import annotations

from dataclasses import dataclass

from super_gpc.tasks.steering.rl.cola_cfg import RlftCfg

# bones-floor "spread_out_R_001" takes (stand, squat, sit, roll onto the back, lie spread
# out) in the prepared corpus that get down smoothly: A432 (1.36 m/s peak downward pelvis
# speed) and A431_M (tips back with the legs kicking up) are left out.
REFERENCE_CLIPS = (
  "spread_out_R_001__A428_M",
  "spread_out_R_001__A429",
  "spread_out_R_001__A429_M",
  "spread_out_R_001__A430",
  "spread_out_R_001__A430_M",
)


@dataclass
class GetDownCfg(RlftCfg):
  # --- Choice: KL and starts ----------------------------------------------
  kl_beta: float = 0.05  # relative to the unconditioned prior
  joystick_fraction: float = 0.5  # resets replaced by joystick-rollout states
  # --- Choice: impact and regularizer weights (penalty weights are magnitudes) --
  body_impact_weight: float = 0.007
  body_impact_n: float = 327.0
  torso_impact_weight: float = 0.014
  torso_impact_n: float = 150.0
  wrist_impact_weight: float = 0.014
  wrist_impact_n: float = 100.0
  body_impact_initial: float = 0.0002
  torso_impact_initial: float = 0.0004
  wrist_impact_initial: float = 0.0004
  action_rate_weight: float = 0.1
  joint_limit_weight: float = 10.0
  # --- Choice: lying gate and impact schedule --------------------------------
  lying_height: float = 0.2  # torso_link height below which the robot can be lying (m)
  lying_horizontal: float = 0.8  # |torso x . world z| above this (37 deg band)
  impact_ramp_iterations: int = 500
  # --- Choice: pelvis height ----------------------------------------------------
  pelvis_weight: float = 0.5
  pelvis_low: float = 0.10  # reference lying pelvis height (m)
  pelvis_stand: float = 0.78  # frozen-prior standing pelvis height (m)
  # --- Choice: AMP style reward -------------------------------------------------
  amp_weight: float = 1.0
  reference_clips: tuple[str, ...] = REFERENCE_CLIPS
  reference_tail_s: float = 1.5  # dropped before each clip leaves lying on its back
  disc_hidden: tuple[int, ...] = (1024, 512)
  disc_lr: float = 1e-4
  disc_weight_decay: float = 1e-4
  disc_minibatch: int = 4096
  disc_epochs: int = 1
  disc_grad_penalty: float = 10.0  # AMP paper value
  disc_logit_reg: float = 0.05
  disc_replay: int = 1_048_576  # recent policy transitions the discriminator samples
  # --- Choice: episodes ---------------------------------------------------------
  episode_steps: int = 400  # 8 s, bootstrapped timeout; the reference lies down in about 3 s
  joint_limit_terminate_rad: float = 1.0  # beyond the physical range, as steering
  # --- Choice: budget and evaluation ----------------------------------------
  target_transitions: int = 2000 * 4096 * 32
  eval_envs: int = 144  # 9 push conditions x 16 starts
  eval_steps: int = 400
  eval_push_mps: tuple[float, ...] = (0.6, 1.0)
  eval_lying_s: float = 1.0  # final window over which settled metrics are averaged
