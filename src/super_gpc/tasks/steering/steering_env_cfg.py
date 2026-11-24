"""Reference-free joystick steering environment for the Dollhouse G1.

Same scene, robot, actuation, physical randomization, solver and 50 Hz
control as the tracking task, without a motion command. Resets place the
robot in its default standing pose with a uniformly random heading and
+-0.02 rad joint noise (G1 gpc steering reset); a contact sensor reports the
net contact force on every non-foot body for the undesired-contact penalty.
Rewards, commands and terminations are evaluated by the trainer.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import events as mjlab_events
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from super_gpc.tasks.tracking.config.dollhouse_g1.env_cfgs import dollhouse_g1_tracking_env_cfg
from super_gpc.tasks.tracking.mdp.events import PhysicsRandomization

CONTACT_SENSOR = "nonfoot_contact"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def dollhouse_g1_steering_env_cfg(
  num_envs: int, episode_s: float, joint_noise: float, nominal_physics: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = dollhouse_g1_tracking_env_cfg(num_envs, play=nominal_physics, reset_perturbation=False)
  cfg.commands = {}
  cfg.events = {
    "physics": EventTermCfg(
      func=PhysicsRandomization, mode="reset", params={"nominal": nominal_physics}
    ),
    "base": EventTermCfg(
      func=mjlab_events.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-math.pi, math.pi)}, "velocity_range": {}},
    ),
    "joints": EventTermCfg(
      func=mjlab_events.reset_joints_by_offset,
      mode="reset",
      params={"position_range": (-joint_noise, joint_noise), "velocity_range": (0.0, 0.0)},
    ),
  }
  cfg.scene.sensors = (
    ContactSensorCfg(
      name=CONTACT_SENSOR,
      primary=ContactMatch(mode="body", pattern=(".*",), entity="robot", exclude=FOOT_BODIES),
      fields=("found", "force"),
      reduce="netforce",
    ),
  )
  cfg.episode_length_s = episode_s
  return cfg
