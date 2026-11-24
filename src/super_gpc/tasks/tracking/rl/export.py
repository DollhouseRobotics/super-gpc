"""Stage 1 tracker ONNX export, Miniverse source bundle and parity checks.

The exported graph reuses the training functions (``encoder_features``,
``actor_observation``, FSQ, encoder/decoder and torso kinematics) on a small
bank of named reference clips, so deployment and training share one numeric
path. Physical targets are ``default + action_scale * action`` in canonical
actuator order, which equals the canonical action order.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.multiprocessing as mp
from torch import nn

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1
from super_gpc.rl.networks import Tracker
from super_gpc.tasks.tracking.mdp import jitter as J
from super_gpc.tasks.tracking.mdp import kinematics as K
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.mdp.motion_library import POLICY_HZ, MotionLibrary
from super_gpc.tasks.tracking.tracking_env_cfg import DECIMATION, PHYSICS_DT

CONTRACT_KEY = "com.dollhouserobotics.miniverse.simulation_contract"
PAST_DIM = (O.HISTORY - 1) * O.OBS_DIM
FREE_JOINT = "floating_base_joint"
CTX = mp.get_context("spawn")  # background export processes


def export_worker_init(threads: int) -> None:
  """Export-process setup: exit with the parent (Linux), FP32 only, a small
  thread budget. Batch-1 CPU inference is fastest on a couple of threads;
  throughput comes from several concurrent exports."""
  if sys.platform.startswith("linux"):
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.set_num_threads(threads)
  os.environ.setdefault("OMP_NUM_THREADS", str(threads))


def interface() -> tuple[tuple, tuple]:
  """ONNX inputs and outputs. The robot state, reference and position targets
  stay 27-joint; ``prev_action``, ``action`` and the history have the policy width."""
  inputs = (
    ("robot_state", "float32", [1, 67]),
    ("past_obs", "float32", [1, (O.HISTORY - 1) * O.OBS_DIM]),
    ("prev_action", "float32", [1, O.ACTION_DIM]),
    ("reset", "float32", [1, 1]),
    ("motion", "int64", [1]),
    ("frame", "int64", [1]),
  )
  outputs = (
    ("position_targets", 27),
    ("action", O.ACTION_DIM),
    ("history", O.HISTORY_DIM),
    ("fsq_codes", 16),
    ("reference", 67),
  )
  return inputs, outputs


INPUTS, OUTPUTS = interface()


def sha256(path: Path) -> str:
  with Path(path).open("rb") as f:
    return hashlib.file_digest(f, "sha256").hexdigest()


def reference_bank(
  lib: MotionLibrary, reference_offsets: Sequence[int] = O.DEFAULT_REFERENCE_OFFSETS
) -> tuple[torch.Tensor, torch.Tensor]:
  """(M, max_len + max_offset + 1, 67) states, padded past each clip end exactly as
  training encoder lookups pad (final pose, zero velocity)."""
  reference_offsets = O.validate_reference_offsets(reference_offsets)
  width = int(lib.lengths.max()) + max(reference_offsets) + 1
  frames = torch.arange(width, device=lib.device)
  clips = torch.arange(lib.num_clips, device=lib.device)
  bank = lib.state(clips[:, None].expand(-1, width), frames[None].expand(lib.num_clips, -1))
  return bank.float().cpu(), lib.lengths.cpu()


class InferenceGraph(nn.Module):
  def __init__(
    self,
    tracker: Tracker,
    kin_torso: K.Kinematics,
    lib: MotionLibrary,
    action_clip: float,
    reference_offsets: Sequence[int] = O.DEFAULT_REFERENCE_OFFSETS,
  ):
    super().__init__()
    self.reference_offsets = O.validate_reference_offsets(reference_offsets)
    self.tracker = copy.deepcopy(tracker).cpu().eval()
    self.kin_torso = copy.deepcopy(kin_torso).cpu()
    bank, lengths = reference_bank(lib, self.reference_offsets)
    self.register_buffer("bank", bank)
    self.register_buffer("lengths", lengths)
    names = G1.ACTION_JOINT_NAMES
    self.register_buffer("default_q", torch.tensor([G1.DEFAULT_JOINT_POS[n] for n in names]))
    self.register_buffer("scale", torch.tensor([G1.ACTION_SCALE[n] for n in names]))
    # Separate the current alignment anchor from the configured samples;
    # offset zero intentionally appears twice for the SuperMimic cadence.
    self.register_buffer("offsets", torch.tensor((0, *self.reference_offsets)))
    self.action_clip = action_clip

  def forward(self, robot_state, past_obs, prev_action, reset, motion, frame):
    keep = 1.0 - reset
    obs = O.actor_observation(
      robot_state, prev_action * keep, torch.zeros_like(prev_action), self.default_q
    )
    past = keep * past_obs + reset * obs.repeat(1, O.HISTORY - 1)
    history = torch.cat((past, obs), -1)
    f = torch.minimum(frame, self.lengths[motion] - 1)
    rows = self.bank[motion[0]][f[0] + self.offsets]  # (11, 67)
    ref_now, ref_future = rows[None, 0], rows[None, 1:]
    features = O.encoder_features(self.kin_torso, robot_state, ref_now, ref_future)
    latent, codes, _ = self.tracker.encode(features)
    action = self.tracker.decode(latent, history).clamp(-self.action_clip, self.action_clip)
    targets = self.default_q + self.scale * action
    return targets, action, history, codes, ref_now


def canonical(value) -> bytes:
  return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def export_onnx(
  graph: InferenceGraph, path: Path, bundle_id: str, provenance: dict, actuator_names: list[str]
) -> str:
  graph = graph.cpu().eval()
  inputs, outputs = interface()
  example = (
    torch.zeros(1, 67),
    torch.zeros(1, (O.HISTORY - 1) * O.OBS_DIM),
    torch.zeros(1, O.ACTION_DIM),
    torch.ones(1, 1),
    torch.zeros(1, dtype=torch.int64),
    torch.zeros(1, dtype=torch.int64),
  )
  example[0][0, 3] = 1.0
  path.parent.mkdir(parents=True, exist_ok=True)
  with torch.no_grad():
    torch.onnx.export(
      graph,
      example,
      str(path),
      input_names=[n for n, _, _ in inputs],
      output_names=[n for n, _ in outputs],
      opset_version=18,
      dynamo=False,
    )
  model = onnx.load(str(path))
  physics_hz = round(1 / PHYSICS_DT)
  contract = dict(
    schemaVersion="0.3",
    precision="fp32",
    contractHash="",
    modelSha256=hashlib.sha256(model.SerializeToString(deterministic=True)).hexdigest(),
    skeletonId=bundle_id,
    compatibleSceneContractHashes=[hashlib.sha256(bundle_id.encode()).hexdigest()],
    backends=[dict(id="mujoco-cpu", versionRange=">=3.3,<4", providers=["CPUExecutionProvider"])],
    opset=18,
    execution=dict(kind="singlePolicyStep"),
    rates=dict(
      physicsHz=physics_hz,
      policyHz=POLICY_HZ,
      publishHz=POLICY_HZ,
      actionHold="zero-order-hold",
      commandBoundary="policy",
      controlLoop="policy-then-decimation",
    ),
    inputs=[
      dict(
        name=n, dtype=t, shape=s, slices=[dict(start=0, length=math.prod(s), provider="constant")]
      )
      for n, t, s in inputs
    ],
    outputs=[dict(name=n, dtype="float32", shape=[1, w], role="auxiliary") for n, w in outputs],
    commands=[],
    stateEstimation=dict(mode="simulator-ground-truth"),
    provenance=provenance,
  )
  default = [G1.DEFAULT_JOINT_POS[n] for n in G1.ACTION_JOINT_NAMES]
  contract["outputs"][0].update(
    role="actuatorTargets",
    controlMode="position",
    clip=[-1000000.0, 1000000.0],
    actuators=actuator_names,
    failsafe=default,
  )
  contract["contractHash"] = hashlib.sha256(
    canonical({k: v for k, v in contract.items() if k != "contractHash"})
  ).hexdigest()
  for suffix, value in (
    ("", canonical(contract).decode()),
    ("_schema_version", "0.3"),
    ("_hash", contract["contractHash"]),
  ):
    prop = model.metadata_props.add()
    prop.key, prop.value = CONTRACT_KEY + suffix, value
  onnx.save(model, str(path))
  return sha256(path)


VELOCITY_LIMIT_KEY = "dhr.miniverse.actuator_velocity_limit."


def write_embodiment(directory: Path, velocity_limit: float | None = None) -> list[str]:
  """Canonical ``g1_body25_head2`` MJCF edited to the training dynamics.

  Only attributes the training entity changes are edited in the canonical XML:
  solver options, BeyondMimic armature, position-actuator gains and effort
  limits (unclipped targets, as in training), foot friction/solref/priority
  and the condim of non-foot collision geoms. Every compiled dynamics array is
  then checked against MJLab's compiled training entity. Returns the canonical
  actuator names, which are in canonical action order.
  """
  import xml.etree.ElementTree as ET

  import mujoco
  import numpy as np
  from mjlab.entity import Entity

  source = G1.mjcf_path()
  G1.verify_asset()
  tree = ET.parse(source)
  root = tree.getroot()
  option = root.find("option")
  if option is None:
    option = ET.Element("option")
    root.insert(1, option)
  option.attrib.update(
    timestep=repr(PHYSICS_DT),
    integrator="implicitfast",
    cone="pyramidal",
    iterations="10",
    ls_iterations="20",
  )
  for joint in root.iter("joint"):
    name = joint.get("name")
    if name in G1.ACTUATION:
      joint.set("armature", repr(G1.ACTUATION[name].armature))
  foot = re.compile(G1.FOOT_GEOMS)
  for geom in root.iter("geom"):
    name = geom.get("name") or ""
    if foot.match(name):
      geom.attrib.update(
        friction=repr(G1.NOMINAL_FOOT_FRICTION),
        priority="1",
        condim="3",
        solref=f"{G1.FOOT_SOLREF_TIMECONST} {G1.FOOT_SOLREF_DAMPRATIO}",
      )
    elif re.match(r".*_collision", name):  # MJLab CollisionCfg prefix-match semantics
      geom.set("condim", "1")
  actuators = root.find("actuator")
  assert actuators is not None
  names = []
  for act in actuators:
    a = G1.ACTUATION[act.get("joint")]
    act.attrib.update(
      kp=repr(a.kp),
      kv=repr(a.kd),
      forcerange=f"{-a.effort_limit} {a.effort_limit}",
      forcelimited="true",
      ctrlrange="-1000000 1000000",
      ctrllimited="false",
    )
    names.append(act.get("name"))
  if velocity_limit is not None:
    custom = root.find("custom")
    if custom is None:
      custom = ET.SubElement(root, "custom")
    for n in names:
      ET.SubElement(custom, "numeric", name=VELOCITY_LIMIT_KEY + n, data=repr(velocity_limit))
  directory.mkdir(parents=True)
  meshdir = (root.find("compiler").get("meshdir") or "").strip("/")
  (directory / meshdir).mkdir(parents=True, exist_ok=True)
  for mesh in root.iter("mesh"):
    shutil.copy2(source.parent / meshdir / mesh.get("file"), directory / meshdir / mesh.get("file"))
  tree.write(directory / "robot.xml", encoding="utf-8", xml_declaration=True)

  got = mujoco.MjModel.from_xml_path(str(directory / "robot.xml"))
  ref = Entity(G1.get_dollhouse_g1_robot_cfg()).spec.compile()
  assert [got.geom(i).name for i in range(got.ngeom)] == [
    ref.geom(i).name for i in range(ref.ngeom)
  ]
  for field in (
    "geom_contype",
    "geom_conaffinity",
    "geom_condim",
    "geom_priority",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_margin",
    "geom_gap",
    "geom_size",
    "geom_pos",
    "geom_quat",
  ):
    assert np.allclose(getattr(got, field), getattr(ref, field), atol=1e-9), field
  for field in (
    "dof_armature",
    "dof_damping",
    "dof_frictionloss",
    "jnt_range",
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_pos",
    "body_quat",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "actuator_forcelimited",
    "actuator_ctrllimited",
    "actuator_gaintype",
    "actuator_biastype",
  ):
    assert np.allclose(getattr(got, field), getattr(ref, field), atol=1e-9), field
  joints = [got.joint(int(got.actuator_trnid[i, 0])).name for i in range(got.nu)]
  assert joints == list(G1.ACTION_JOINT_NAMES), joints
  assert got.opt.timestep == PHYSICS_DT and got.opt.iterations == 10
  return names


def select_examples(
  motions: Path,
  per_category: int = 2,
  min_s: float = 4.0,
  max_s: float = 12.0,
  max_total: int = 16,
) -> list[str]:
  """Deterministic switcher references: per category, the first non-mirrored
  clips of 4-12 s by name hash, from the validation split when the corpus has one.

  At most ``max_total`` (the controller source must stay under 64 KiB): every
  category's first clip comes before any category's second."""
  clips = json.loads((Path(motions) / "clips.json").read_text())
  split = "val" if any(c["split"] == "val" for c in clips) else "train"
  ranked = []
  for category in sorted({c["category"] for c in clips}):
    pool = [
      c["name"]
      for c in clips
      if c["category"] == category
      and c["split"] == split
      and not c["mirror"]
      and min_s * POLICY_HZ <= c["frames"] <= max_s * POLICY_HZ
    ]
    pool.sort(key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    ranked += [(rank, category, name) for rank, name in enumerate(pool[:per_category])]
  kept = sorted(ranked)[:max_total]
  return [name for _, _, name in sorted(kept, key=lambda x: (x[1], x[0]))]


def reset_states(lib: MotionLibrary) -> list[dict]:
  out = []
  for c in range(lib.num_clips):
    s = lib.state(
      torch.tensor([c], device=lib.device), torch.zeros(1, dtype=torch.long, device=lib.device)
    )[0]
    s = s.double().cpu().numpy()
    q = s[3:7]
    out.append(
      dict(
        root_position=s[0:3].tolist(),
        root_rotation=[q[1], q[2], q[3], q[0]],  # XYZW
        root_linear_velocity=s[7:10].tolist(),
        root_angular_velocity=s[10:13].tolist(),  # pelvis body frame
        joint_positions={
          n: [float(v)] for n, v in zip(G1.ACTION_JOINT_NAMES, s[13:40], strict=True)
        },
        joint_velocities={
          n: [float(v)] for n, v in zip(G1.ACTION_JOINT_NAMES, s[40:67], strict=True)
        },
      )
    )
  return out


def build_bundle(
  out: Path,
  graph: InferenceGraph,
  lib: MotionLibrary,
  bundle_id: str,
  name: str,
  provenance: dict,
  metadata: dict,
  embodiment_from: Path | None = None,
) -> dict:
  """Write the source bundle directory and its .mini archive.

  ``embodiment_from`` copies an embodiment already written by this function
  (the robot is identical for every checkpoint of a run) instead of
  recompiling MJLab's entity spec.
  """
  if out.exists():
    shutil.rmtree(out)
  if embodiment_from is None:
    actuators = write_embodiment(out / "embodiment")
  else:
    shutil.copytree(embodiment_from, out / "embodiment")
    actuators = [f"{n}_actuator" for n in G1.ACTION_JOINT_NAMES]
  model_path = out / "models" / "tracker.onnx"
  onnx_sha = export_onnx(graph, model_path, bundle_id, provenance, actuators)
  labels = [c["name"] for c in lib.clips]
  constants = dict(
    MOTION_LABELS=labels,
    RESET_STATES=reset_states(lib),
    JOINT_NAMES=list(G1.ACTION_JOINT_NAMES),
    FREE_JOINT=FREE_JOINT,
    LENGTHS=[int(n) for n in lib.lengths.cpu()],
    OBS_DIM=O.OBS_DIM,
    ACTION_DIM=O.ACTION_DIM,
    HISTORY=O.HISTORY,
    HIGHPASS=J.highpass_coefficients(POLICY_HZ),
  )
  template = (Path(__file__).parent / "miniverse_policy.py").read_text()
  source = "\n".join(f"{k} = {v!r}" for k, v in constants.items()) + "\n\n" + template
  assert len(source.encode()) < 64 * 1024, len(source.encode())
  (out / "policy.py").write_text(source)
  assert round(1 / PHYSICS_DT) // DECIMATION == POLICY_HZ
  bundle = dict(
    version="v1",
    id=bundle_id,
    name=name,
    description="Reference-motion tracker: select a named motion to replay it on the robot.",
    primarySimulator="mujoco",
    embodiment=dict(kind="mjcf", path="embodiment/robot.xml"),
    models=[dict(id="tracker")],
    program=dict(apiVersion="dhr.python-policy/v1", entrypoint="policy:Policy"),
    commands=[
      dict(
        id="motion",
        kind="scalar",
        label="Motion",
        unit="clip",
        default=[0],
        range=[0, len(labels) - 1],
        step=1,
        frame="selection",
        sliceLength=1,
        update="on-release",
        restart="episode",
      )
    ],
    ui=dict(
      components=[
        dict(
          id="motion-selector",
          renderer="builtin/switcher",
          commandId="motion",
          options=dict(items=[dict(value=i, label=lbl) for i, lbl in enumerate(labels)], wrap=True),
        )
      ]
    ),
    viewer=dict(camera=dict(framingScale=1.15)),
    metadata=metadata,
  )
  (out / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
  archive = out.with_suffix(".mini")
  with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for file in sorted(out.rglob("*")):
      if file.is_file():
        z.write(file, file.relative_to(out))
  return dict(
    archive=str(archive),
    archive_sha256=sha256(archive),
    onnx=str(model_path),
    onnx_sha256=onnx_sha,
    policy_sha256=sha256(out / "policy.py"),
  )


def ort_session(onnx_path: Path):
  """CPU session sized to the caller's torch thread budget (two in export workers).

  ONNX Runtime's default spawns one intra-op thread per host core, which on a
  64-core host oversubscribes the container's CPU quota and slows training."""
  import onnxruntime as ort

  opts = ort.SessionOptions()
  opts.intra_op_num_threads = opts.inter_op_num_threads = torch.get_num_threads()
  return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


@torch.no_grad()
def parity(graph: InferenceGraph, onnx_path: Path, samples: list[tuple]) -> dict[str, float]:
  """Max absolute native/ONNX Runtime differences per output over ``samples``."""
  sess = ort_session(onnx_path)
  graph = graph.cpu().eval()
  inputs, outputs = interface()
  worst = {n: 0.0 for n, _ in outputs}
  code_mismatch = 0
  for sample in samples:
    native = graph(*sample)
    feeds = {n: t.numpy() for (n, _, _), t in zip(inputs, sample, strict=True)}
    got = sess.run([n for n, _ in outputs], feeds)
    for (n, _), a, b in zip(outputs, native, got, strict=True):
      worst[n] = max(worst[n], float(np.abs(a.numpy() - b).max()))
    code_mismatch += int((native[3].numpy() != got[3]).any())
  return {f"export/parity_max_abs_{n}": v for n, v in worst.items()} | {
    "export/parity_code_mismatch_samples": float(code_mismatch),
    "export/parity_samples": float(len(samples)),
  }


class _Slices:
  def __init__(self, model):
    self.m = model

  def joint_slice(self, name, velocity=False):
    import mujoco

    j = self.m.joint(name).id
    width = {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6), int(mujoco.mjtJoint.mjJNT_BALL): (4, 3)}.get(
      int(self.m.jnt_type[j]), (1, 1)
    )
    start = self.m.jnt_dofadr[j] if velocity else self.m.jnt_qposadr[j]
    return slice(int(start), int(start) + width[1 if velocity else 0])


class _Step:
  def __init__(self, data, model_view, commands, policy_index):
    self.sim_data = type(
      "SimData",
      (),
      dict(
        joint_positions=data.qpos.copy(),
        joint_velocities=data.qvel.copy(),
        simulation_time=float(data.time),
        model=model_view,
      ),
    )()
    self.commands, self.policy_index = commands, policy_index


@torch.no_grad()
def closed_loop_check(
  bundle_dir: Path,
  lib: MotionLibrary,
  tracker: Tracker,
  kin: K.Kinematics,
  kin_torso: K.Kinematics,
  seconds: float = 10.0,
  reference_offsets: Sequence[int] = O.DEFAULT_REFERENCE_OFFSETS,
) -> dict:
  """Run the bundle's own policy.py and ONNX model on its MJCF in native MuJoCo.

  For every motion of the switcher, reset through ``reset_state``, step the
  controller at 50 Hz with 4 physics substeps and measure tracking against the
  reference library and SuperMimic failures. At each step the training-code
  action (``Tracker`` on ``encoder_features`` and the controller's clean
  history) is recomputed and compared with the exported action.
  """
  import mujoco

  from super_gpc.tasks.tracking.mdp import rewards as R

  reference_offsets = O.validate_reference_offsets(reference_offsets)
  # The embodiment has no ground (Miniverse supplies it), so add the training
  # plane; without it the robot free-falls and every reference "fails" at 0.24 s.
  spec = mujoco.MjSpec.from_file(str(bundle_dir / "embodiment" / "robot.xml"))
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05])
  model = spec.compile()
  data = mujoco.MjData(model)
  sess = ort_session(bundle_dir / "models" / "tracker.onnx")

  class Model:
    def run(self, feeds):
      names = [o.name for o in sess.get_outputs()]
      return dict(zip(names, sess.run(names, feeds), strict=True))

  class InitialState(dict):
    def __init__(self, **kw):
      super().__init__(**kw)

  class StepResult:
    def __init__(self, actuation, metrics=None, **_):
      self.actuation, self.metrics = actuation, metrics or {}

  scope = dict(np=np, ControllerInitialState=InitialState, StepResult=StepResult)
  exec(compile((bundle_dir / "policy.py").read_text(), "policy.py", "exec"), scope)
  view = _Slices(model)
  tracker = copy.deepcopy(tracker).cpu().eval()
  kin, kin_torso = copy.deepcopy(kin).cpu(), copy.deepcopy(kin_torso).cpu()
  soft = model.jnt_range[[model.joint(n).id for n in G1.ACTION_JOINT_NAMES]]
  mid, half = soft.mean(1), 0.45 * (soft[:, 1] - soft[:, 0])
  soft_lo, soft_hi = torch.tensor(mid - half).float(), torch.tensor(mid + half).float()
  default_q = torch.tensor([G1.DEFAULT_JOINT_POS[n] for n in G1.ACTION_JOINT_NAMES])
  results = []
  for motion in range(lib.num_clips):
    policy = scope["Policy"]()
    policy.initialize(type("Ctx", (), dict(models={"tracker": Model()}))())
    init = policy.reset_state({"motion": [float(motion)]})
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = init["root_position"]
    x, y, z, w = init["root_rotation"]
    data.qpos[3:7] = (w, x, y, z)
    data.qvel[0:3] = init["root_linear_velocity"]
    data.qvel[3:6] = init["root_angular_velocity"]
    for n in G1.ACTION_JOINT_NAMES:
      data.qpos[view.joint_slice(n)] = init["joint_positions"][n]
      data.qvel[view.joint_slice(n, True)] = init["joint_velocities"][n]
    mujoco.mj_forward(model, data)
    length = int(lib.lengths[motion])
    steps = min(round(seconds * POLICY_HZ), length - 1)
    clip_t = torch.tensor([motion], device=lib.device)
    errs, max_action_diff, failed_at, prev_action = [], 0.0, -1, torch.zeros(1, O.ACTION_DIM)
    hist = None
    for i in range(steps):
      result = policy.step(_Step(data, view, {"motion": [float(motion)]}, i))
      # training-code path on the same state and clean history
      s = policy_state = torch.from_numpy(policy.robot_state(_Step(data, view, {}, i).sim_data))
      obs = O.actor_observation(s, prev_action, torch.zeros(1, 27), default_q)
      hist = (
        obs.repeat(1, O.HISTORY) if hist is None else torch.cat((hist[:, O.OBS_DIM :], obs), -1)
      )
      f = torch.tensor([min(i, length - 1)], device=lib.device)
      offs = torch.tensor(reference_offsets, device=lib.device)
      ref_now = lib.state(clip_t, f).float().cpu()
      fut = lib.state(clip_t[:, None], f[:, None] + offs).float().cpu()
      action = tracker(O.encoder_features(kin_torso, s, ref_now, fut), hist)
      action = action.clamp(-5.0, 5.0)
      max_action_diff = max(
        max_action_diff, float((action - torch.from_numpy(policy.prev)).abs().max())
      )
      prev_action = torch.from_numpy(policy.prev.copy())
      data.ctrl[:] = result.actuation
      for _ in range(DECIMATION):
        mujoco.mj_step(model, data)
      s1 = torch.from_numpy(policy.robot_state(_Step(data, view, {}, i).sim_data))
      ref1 = lib.state(clip_t, f + 1).float().cpu()
      terms = R.tracking_reward(
        kin(s1), kin(ref1), prev_action, prev_action, s1[:, 13:40], soft_lo, soft_hi
      )
      fail, dz, dg = R.physical_failure(s1, ref1)
      errs.append(
        [
          float(terms["err_body_pos_m"]),
          float(terms["err_anchor_pos_m"]),
          float((s1[0, 13:40] - ref1[0, 13:40]).pow(2).mean().sqrt()),
          float(terms["total"] - terms["action_rate"]),
        ]
      )
      if bool(fail) and failed_at < 0:
        failed_at = i + 1
        break
      del policy_state
    e = np.asarray(errs)
    results.append(
      dict(
        motion=lib.clips[motion]["name"],
        steps=len(errs),
        clip_steps=length - 1,
        failed_at_step=failed_at,
        mean_body_pos_err_m=float(e[:, 0].mean()),
        mean_anchor_pos_err_m=float(e[:, 1].mean()),
        mean_joint_rmse_rad=float(e[:, 2].mean()),
        mean_reward_without_action_rate=float(e[:, 3].mean()),
        max_abs_action_diff_vs_training_code=max_action_diff,
      )
    )
  return dict(
    motions=results,
    success_rate=float(np.mean([r["failed_at_step"] < 0 for r in results])),
    max_abs_action_diff=max(r["max_abs_action_diff_vs_training_code"] for r in results),
  )
