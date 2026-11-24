"""Stage 1 tracking environment composed from MJLab managers.

MJLab owns the scene, MuJoCo Warp simulation, actuators, action application,
reset-mode randomization events and the motion-command clock. Observations,
rewards and failures are the stateless functions in ``mdp`` evaluated by the
trainer on the clean generalized state after each step, so the simulator and
imagined rollouts share one implementation and reward units (no dt scaling).
The environment therefore declares no reward, observation or termination
terms and never auto-resets; the trainer resets done environments explicitly.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from super_gpc.tasks.tracking.mdp.commands import (
  RESET_POSE_RANGE,
  RESET_VELOCITY_RANGE,
  MotionCommandCfg,
)
from super_gpc.tasks.tracking.mdp.events import PhysicsRandomization

PHYSICS_DT = 0.005
DECIMATION = 4
EPISODE_LENGTH_S = 10.0
# Contact (pooled per world) and per-world constraint budgets. 27 joint limits
# and 27 friction-loss rows are always present; pyramidal foot contacts add 4
# rows each. Peak use is logged as sim/max_nefc and sim/contacts_per_world.
NCONMAX = 64
NJMAX = 512


def make_tracking_env_cfg(
  num_envs: int = 4096,
  nominal_physics: bool = False,
  reset_perturbation: bool = True,
) -> ManagerBasedRlEnvCfg:
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=num_envs),
    observations={},
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0, use_default_offset=True
      )
    },
    commands={
      "motion": MotionCommandCfg(
        pose_range=RESET_POSE_RANGE if reset_perturbation else {},
        velocity_range=RESET_VELOCITY_RANGE if reset_perturbation else {},
        joint_position_range=(-0.1, 0.1) if reset_perturbation else (0.0, 0.0),
      )
    },
    events={
      "physics": EventTermCfg(
        func=PhysicsRandomization, mode="reset", params={"nominal": nominal_physics}
      ),
    },
    rewards={},
    terminations={},
    sim=SimulationCfg(
      nconmax=NCONMAX,
      njmax=NJMAX,
      mujoco=MujocoCfg(timestep=PHYSICS_DT, iterations=10, ls_iterations=20),
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link"
    ),
    decimation=DECIMATION,
    episode_length_s=EPISODE_LENGTH_S,
    scale_rewards_by_dt=False,
    auto_reset=False,
  )
