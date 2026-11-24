"""Register the Dollhouse G1 Stage 1 tracking task with MJLab's registry.

MJLab's default RSL-RL PPO runner is not the Stage 1 algorithm, so no PPO
runner configuration is registered; train with ``gpc-train-stage1``.
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import dollhouse_g1_tracking_env_cfg

TASK_ID = "Gpc-Tracking-Flat-Dollhouse-G1-Body25-Head2"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=dollhouse_g1_tracking_env_cfg(),
  play_env_cfg=dollhouse_g1_tracking_env_cfg(play=True),
  rl_cfg=None,  # type: ignore[arg-type]
  runner_cls=None,
)
