"""Dollhouse G1 ``g1_body25_head2`` robot configuration for MJLab.

The canonical MJCF and meshes are resolved through the pinned
``dollhouse-robots`` package. This module owns the project-specific actuation:
BeyondMimic nominal body-joint PD gains, armatures, effort limits and action
scales, provisional neck values and the knees-bent default pose. The joint-name
mapping is explicit; no positional order is inherited from a source library.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
from dollhouse_robots import get_robot
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

ROBOT_ID = "g1_body25_head2"
ROBOTS_REVISION = "62aebbf1bb3f8b2518aceef45ed39e8b475744cb"
MJCF_SHA256 = "d70db45b3937013cb8c1162516ee4b66304019bafae0380276a40daacb9cda9f"

# Canonical 27-actuator action order: the canonical MJCF <actuator> order.
ACTION_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "head_pan_joint",
  "head_tilt_joint",
)
NUM_ACTIONS = len(ACTION_JOINT_NAMES)

# BeyondMimic G1 motor-group armatures and PD design rule (BM-actuation,
# whole_body_tracking@9db9158 robots/g1.py). Source constants, not rounded
# table values.
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
  return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
  return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


@dataclass(frozen=True)
class JointActuation:
  kp: float
  kd: float
  effort_limit: float
  armature: float

  @property
  def action_scale(self) -> float:
    """BeyondMimic rule: 0.25 x nominal effort limit / nominal Kp."""
    return 0.25 * self.effort_limit / self.kp


_HIP_PITCH_YAW_WAIST_YAW = JointActuation(
  _stiffness(ARMATURE_7520_14), _damping(ARMATURE_7520_14), 88.0, ARMATURE_7520_14
)
_HIP_ROLL_KNEE = JointActuation(
  _stiffness(ARMATURE_7520_22), _damping(ARMATURE_7520_22), 139.0, ARMATURE_7520_22
)
_ANKLE_WAIST_PITCH_ROLL = JointActuation(
  2.0 * _stiffness(ARMATURE_5020),
  2.0 * _damping(ARMATURE_5020),
  50.0,
  2.0 * ARMATURE_5020,
)
_ARM = JointActuation(_stiffness(ARMATURE_5020), _damping(ARMATURE_5020), 25.0, ARMATURE_5020)
# Provisional Dollhouse neck servo values (spec): Kp 20, Kd = 4010-class
# damping at 10 Hz (1.068142), effort 8 N m. Armature uses the same 4010-class
# value that produced Kd. Requires hardware identification.
_NECK = JointActuation(20.0, _damping(ARMATURE_4010), 8.0, ARMATURE_4010)


def _group(name: str) -> JointActuation:
  if name.endswith(("hip_pitch_joint", "hip_yaw_joint")) or name == "waist_yaw_joint":
    return _HIP_PITCH_YAW_WAIST_YAW
  if name.endswith(("hip_roll_joint", "knee_joint")):
    return _HIP_ROLL_KNEE
  if name.endswith(("ankle_pitch_joint", "ankle_roll_joint")) or name in (
    "waist_roll_joint",
    "waist_pitch_joint",
  ):
    return _ANKLE_WAIST_PITCH_ROLL
  if name.startswith("head_"):
    return _NECK
  if "shoulder" in name or "elbow" in name or name.endswith("wrist_roll_joint"):
    return _ARM
  raise KeyError(name)


ACTUATION: dict[str, JointActuation] = {n: _group(n) for n in ACTION_JOINT_NAMES}
ACTION_SCALE: dict[str, float] = {n: a.action_scale for n, a in ACTUATION.items()}

# BeyondMimic knees-bent default pose on retained joints; zero elsewhere,
# including both neck axes.
DEFAULT_JOINT_POS: dict[str, float] = {n: 0.0 for n in ACTION_JOINT_NAMES}
DEFAULT_JOINT_POS.update(
  {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  }
)

# Body names used by the task.
PELVIS_BODY = "pelvis"
ANCHOR_BODY = "torso_link"
TRACKED_BODIES: tuple[str, ...] = (
  "pelvis",
  "torso_link",
  "head_tilt_link",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_hand_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_hand_link",
)
FOOT_GEOMS = r"^(left|right)_foot[1-7]_collision$"
NOMINAL_FOOT_FRICTION = 0.6  # MJLab G1 default; training randomizes it.

# MuJoCo's default, critically damped foot-ground contact. The earlier
# restitution-0.5 mapping (damping ratio 0.2764) made standing feet chatter
# and slide; see the spec's randomization section.
FOOT_SOLREF_TIMECONST = 0.02
FOOT_SOLREF_DAMPRATIO = 1.0


def mjcf_path() -> Path:
  return get_robot(ROBOT_ID).mjcf_path


def verify_asset() -> str:
  digest = hashlib.sha256(mjcf_path().read_bytes()).hexdigest()
  if digest != MJCF_SHA256:
    raise RuntimeError(f"{ROBOT_ID} MJCF hash {digest} != pinned {MJCF_SHA256}")
  return digest


def get_spec() -> mujoco.MjSpec:
  """Canonical MJCF without its placeholder actuators.

  MJLab adds the selected position actuators below. Canonical passive joint
  damping (0.05) and friction loss (0.2) are retained; armature is replaced by
  the BeyondMimic motor-group value through the actuator configuration.
  """
  verify_asset()
  spec = mujoco.MjSpec.from_file(str(mjcf_path()))
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  return spec


def _actuator_cfgs() -> tuple[BuiltinPositionActuatorCfg, ...]:
  # One config per joint, in canonical action order, so ctrl order equals the
  # canonical action order.
  return tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(f"^{name}$",),
      stiffness=a.kp,
      damping=a.kd,
      effort_limit=a.effort_limit,
      armature=a.armature,
    )
    for name, a in ACTUATION.items()
  )


FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={FOOT_GEOMS: 3, ".*_collision": 1},
  priority={FOOT_GEOMS: 1, ".*": 0},
  friction={FOOT_GEOMS: (NOMINAL_FOOT_FRICTION,)},
  solref={FOOT_GEOMS: (FOOT_SOLREF_TIMECONST, FOOT_SOLREF_DAMPRATIO)},
)

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.76),
  joint_pos=dict(DEFAULT_JOINT_POS),
  joint_vel={".*": 0.0},
)


def get_dollhouse_g1_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=_actuator_cfgs(),
      soft_joint_pos_limit_factor=0.9,
    ),
  )


assert math.isclose(ACTION_SCALE["head_pan_joint"], 0.1)
