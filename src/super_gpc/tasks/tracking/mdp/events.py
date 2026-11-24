"""Physical randomization and its realized simulator context.

Selected ranges (spec "Physical domain randomization"), drawn independently
and uniformly. Foot friction and torso COM are drawn once per world when the
environment is constructed; all other parameters are drawn at every episode
reset. All parameters stay fixed within an episode:

  foot friction            per env       [0.3, 1.2] (absolute, all foot geoms)
  joint calibration bias   per actuator  [-0.01, 0.01] rad
  torso COM offset         per env       x [-0.025, 0.025], y/z [-0.05, 0.05] m
  torso added mass         per env       [-3, 6] kg, inertia scaled by mass ratio
  motor strength scale     per actuator  [0.8, 1.2]
  PD gain scale            per actuator  [0.8, 1.2], common to Kp and Kd
  armature scale           per actuator  [1.0, 1.05]

MuJoCo position actuators compute ``clamp(kp (u - q) - kd qd, +-F)``. Writing
``s g kp``, ``s g kd`` and ``+-s F`` yields ``s clamp(g kp (u - q) - g kd qd,
+-F)``: the randomized-gain PD torque clipped to the nominal effort limit and
then multiplied by strength, exactly as specified. Action scales stay nominal.
Calibration bias enters the PD error through MJLab's JointPositionAction
(target - bias) and the sensed joint positions in the observation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

NJ = G1.NUM_ACTIONS
CONTEXT_DIM = 1 + 1 + 3 + 4 * NJ
# Context slices: friction | added mass | COM offset | bias | strength | gain | armature
RANGES = dict(
  friction=((0.3, 1.2),),
  added_mass=((-3.0, 6.0),),
  com=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
  bias=((-0.01, 0.01),) * NJ,
  strength=((0.8, 1.2),) * NJ,
  gain=((0.8, 1.2),) * NJ,
  armature=((1.0, 1.05),) * NJ,
)
NOMINAL = dict(
  friction=1.0, added_mass=0.0, com=0.0, bias=0.0, strength=1.0, gain=1.0, armature=1.0
)

_BOUNDS: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _bounds(device) -> tuple[torch.Tensor, torch.Tensor]:
  """Context ranges, built once per device: no host-to-device copy per call,
  so CUDA-graph-captured training updates can normalize context."""
  key = str(device)
  if key not in _BOUNDS:
    ranges = [r for v in RANGES.values() for r in v]
    lo = torch.tensor([r[0] for r in ranges], device=device)
    hi = torch.tensor([r[1] for r in ranges], device=device)
    _BOUNDS[key] = (lo, hi)
  return _BOUNDS[key]


def normalize_context(context: torch.Tensor) -> torch.Tensor:
  """Map each context entry to [-1, 1] over its selected range."""
  lo, hi = _bounds(context.device)
  return (2 * context - lo - hi) / (hi - lo)


class PhysicsRandomization:
  """Reset-mode event term; ``context`` holds each env's realized parameters."""

  model_fields = (
    "geom_friction",
    "body_mass",
    "body_inertia",
    "body_ipos",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "dof_armature",
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
    "actuator_acc0",
  )
  recompute = RecomputeLevel.set_const

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self.env = env
    self.robot = env.scene["robot"]
    self.nominal = bool(cfg.params.get("nominal", False))
    lo, hi = _bounds(env.device)
    self.lo, self.hi = lo, hi
    self.context = torch.zeros(env.num_envs, CONTEXT_DIM, device=env.device)
    self._nominal_context = torch.cat(
      [torch.full((len(RANGES[k]),), float(NOMINAL[k]), device=env.device) for k in RANGES]
    )
    self.context[:] = self._nominal_context
    if not self.nominal:
      startup = [0, 2, 3, 4]  # friction and torso COM, fixed for this world's lifetime
      self.context[:, startup] = lo[startup] + torch.rand(
        env.num_envs, len(startup), device=env.device
      ) * (hi[startup] - lo[startup])
    self._resolved = False

  def _resolve(self) -> None:
    robot, sim = self.robot, self.env.sim
    geom_local, _ = robot.find_geoms(G1.FOOT_GEOMS)
    self.foot_geoms = robot.indexing.geom_ids[torch.tensor(geom_local, device=self.env.device)]
    self.torso = int(robot.indexing.body_ids[robot.find_bodies("^torso_link$")[0][0]])
    tree = list(robot.joint_names)
    self.action_to_tree = torch.tensor(
      [tree.index(n) for n in G1.ACTION_JOINT_NAMES], device=self.env.device
    )
    self.ctrl = robot.indexing.ctrl_ids  # actuator configs are in action order
    self.dof = robot.indexing.joint_v_adr[self.action_to_tree]
    self.default = {
      f: sim.get_default_field(f)
      for f in (
        "geom_friction",
        "body_mass",
        "body_inertia",
        "body_ipos",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "dof_armature",
      )
    }
    self._resolved = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    nominal: bool = False,
  ) -> None:
    if not self._resolved:
      self._resolve()
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if self.nominal:
      ctx = self._nominal_context.expand(n, -1).clone()
    else:
      ctx = self.lo + torch.rand(n, CONTEXT_DIM, device=env.device) * (self.hi - self.lo)
      ctx[:, 0] = self.context[env_ids, 0]
      ctx[:, 2:5] = self.context[env_ids, 2:5]
    self.context[env_ids] = ctx
    m, d = env.sim.model, self.default
    e = env_ids[:, None]
    friction = ctx[:, 0:1]
    m.geom_friction[e, self.foot_geoms[None], 0] = friction.expand(n, len(self.foot_geoms))
    t = self.torso
    m0 = d["body_mass"][t]
    mass = m0 + ctx[:, 1]
    m.body_mass[env_ids, t] = mass
    m.body_inertia[env_ids, t] = d["body_inertia"][t] * (mass / m0)[:, None]
    m.body_ipos[env_ids, t] = d["body_ipos"][t] + ctx[:, 2:5]
    o = 5 + NJ
    strength, gain, arm = ctx[:, o : o + NJ], ctx[:, o + NJ : o + 2 * NJ], ctx[:, o + 2 * NJ :]
    c = self.ctrl[None]
    sg = strength * gain
    m.actuator_gainprm[e, c, 0] = d["actuator_gainprm"][self.ctrl, 0] * sg
    m.actuator_biasprm[e, c, 1] = d["actuator_biasprm"][self.ctrl, 1] * sg
    m.actuator_biasprm[e, c, 2] = d["actuator_biasprm"][self.ctrl, 2] * sg
    m.actuator_forcerange[e, c] = d["actuator_forcerange"][self.ctrl] * strength[..., None]
    m.dof_armature[e, self.dof[None]] = d["dof_armature"][self.dof] * arm
    bias_tree = torch.zeros(n, NJ, device=env.device)
    bias_tree[:, self.action_to_tree] = ctx[:, 5 : 5 + NJ]
    self.robot.data.encoder_bias[env_ids] = bias_tree
