"""Dollhouse G1 (g1_body25_head2) Stage 1 tracking environment variants."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1
from super_gpc.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def dollhouse_g1_tracking_env_cfg(
  num_envs: int = 4096,
  play: bool = False,
  reset_perturbation: bool | None = None,
) -> ManagerBasedRlEnvCfg:
  """Training variant, or ``play``: nominal physics and no reset perturbation.

  ``reset_perturbation`` overrides the reset-pose perturbation independently
  (evaluation keeps training physics randomization but starts on reference).
  """
  perturb = (not play) if reset_perturbation is None else reset_perturbation
  cfg = make_tracking_env_cfg(num_envs, nominal_physics=play, reset_perturbation=perturb)
  cfg.scene.entities = {"robot": G1.get_dollhouse_g1_robot_cfg()}
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = {f"^{n}$": s for n, s in G1.ACTION_SCALE.items()}
  return cfg
