"""Get-down environment for the Dollhouse G1.

Same scene, robot, actuation, physical randomization, solver and 50 Hz
control as the tracking task, without a motion command. The ``start`` reset
event places the robot in a perturbed motion-archive state or a recorded
joystick-rollout state (``mdp.GetDownStarts``). A floor contact sensor
reports every robot body's net force against the terrain at each of the four
physics substeps, so brief impacts are not missed between policy steps.
Rewards and terminations are evaluated by the trainer.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from super_gpc.tasks.get_down.mdp import GetDownStarts
from super_gpc.tasks.tracking.config.dollhouse_g1.env_cfgs import dollhouse_g1_tracking_env_cfg
from super_gpc.tasks.tracking.mdp.events import PhysicsRandomization
from super_gpc.tasks.tracking.tracking_env_cfg import DECIMATION

FLOOR_SENSOR = "floor_contact"


def dollhouse_g1_get_down_env_cfg(num_envs: int, episode_s: float) -> ManagerBasedRlEnvCfg:
  cfg = dollhouse_g1_tracking_env_cfg(num_envs, reset_perturbation=False)
  cfg.commands = {}
  cfg.events = {
    "physics": EventTermCfg(func=PhysicsRandomization, mode="reset"),
    "start": EventTermCfg(func=GetDownStarts, mode="reset"),
  }
  cfg.scene.sensors = (
    ContactSensorCfg(
      name=FLOOR_SENSOR,
      primary=ContactMatch(mode="body", pattern=(".*",), entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      history_length=DECIMATION,
    ),
  )
  cfg.episode_length_s = episode_s
  return cfg
