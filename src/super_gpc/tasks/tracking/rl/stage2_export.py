"""Stage 2 ONNX export, Miniverse source bundle and parity checks.

One exported ``prior`` model is a single 50 Hz policy step: the current clean
sensor frame (the training ``actor_observation`` of the robot state and the
executed previous action) -> four nucleus-sampled tokens (top-p, temperature)
-> 16 integer FSQ codes -> the frozen teacher decoder and its fixed
normalizer over the ten-frame history (reset padding) and ``codes / 3`` ->
joint targets. Miniverse requires each model to emit exactly one actuator
target output, so the frozen decoder is part of the same graph; its weight
digest is recorded as the companion identity. The controller supplies the
four uniforms and therefore owns the randomness. The nucleus mask uses an
exact ``TopK`` form instead of a full sort (see ``prior.nucleus_keep``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1
from super_gpc.rl import prior as P
from super_gpc.rl.networks import FSQ_HALF_LEVELS, Tracker
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.mdp.motion_library import POLICY_HZ, MotionLibrary
from super_gpc.tasks.tracking.rl import export as E
from super_gpc.tasks.tracking.tracking_env_cfg import DECIMATION, PHYSICS_DT

PAST_DIM = (O.HISTORY - 1) * O.OBS_DIM


def interface() -> tuple[tuple, tuple]:
  """ONNX inputs and outputs: the robot state and position targets stay 27-joint;
  ``prev_action``, ``action`` and the history have the policy width."""
  inputs = (
    ("robot_state", "float32", [1, 67]),
    ("past_obs", "float32", [1, (O.HISTORY - 1) * O.OBS_DIM]),
    ("prev_action", "float32", [1, O.ACTION_DIM]),
    ("reset", "float32", [1, 1]),
    ("uniform", "float32", [1, P.NUM_TOKENS]),
  )
  outputs = (
    ("position_targets", 27),
    ("action", O.ACTION_DIM),
    ("history", O.HISTORY_DIM),
    ("fsq_codes", 16),
    ("tokens", P.NUM_TOKENS),
  )
  return inputs, outputs


INPUTS, OUTPUTS = interface()


def _frame(robot_state, prev_action, reset, default_q):
  keep = 1.0 - reset
  return O.actor_observation(
    robot_state, prev_action * keep, torch.zeros_like(prev_action), default_q
  ), keep


def _session(onnx_path: Path):
  """CPU ONNX Runtime session limited to torch's thread budget. ORT's default
  pool spans every core and spin-waits, which oversubscribes the host when
  export runs beside training."""
  import onnxruntime as ort

  opts = ort.SessionOptions()
  opts.intra_op_num_threads = opts.inter_op_num_threads = torch.get_num_threads()
  return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def _default_q() -> torch.Tensor:
  return torch.tensor([G1.DEFAULT_JOINT_POS[n] for n in G1.ACTION_JOINT_NAMES])


class PriorPolicyGraph(nn.Module):
  def __init__(
    self, prior: P.CodePrior, tracker: Tracker, top_p: float, temperature: float, action_clip: float
  ):
    super().__init__()
    self.prior = copy.deepcopy(prior).cpu().eval()
    self.tracker = copy.deepcopy(tracker).cpu().eval()
    self.top_p, self.temperature, self.action_clip = top_p, temperature, action_clip
    names = G1.ACTION_JOINT_NAMES
    self.register_buffer("default_q", torch.tensor([G1.DEFAULT_JOINT_POS[n] for n in names]))
    self.register_buffer("scale", torch.tensor([G1.ACTION_SCALE[n] for n in names]))

  def forward(self, robot_state, past_obs, prev_action, reset, uniform):
    keep = 1.0 - reset
    obs = O.actor_observation(
      robot_state, prev_action * keep, torch.zeros_like(prev_action), self.default_q
    )
    tokens, _ = self.prior.generate(obs, uniform, self.top_p, self.temperature, export=True)
    codes = P.tokens_to_codes(tokens)
    past = keep * past_obs + reset * obs.repeat(1, O.HISTORY - 1)
    history = torch.cat((past, obs), -1)
    action = self.tracker.decode(codes / FSQ_HALF_LEVELS, history)
    action = action.clamp(-self.action_clip, self.action_clip)
    targets = self.default_q + self.scale * action
    return targets, action, history, codes, tokens.float()


def _example(inputs) -> tuple[torch.Tensor, ...]:
  out = []
  for name, dtype, shape in inputs:
    t = torch.zeros(shape, dtype=torch.float32 if dtype == "float32" else torch.int64)
    if name == "robot_state":
      t[0, 3] = 1.0
    if name == "reset":
      t.fill_(1.0)
    if name == "uniform":
      t.fill_(0.5)
    out.append(t)
  return tuple(out)


def export_model(
  graph: nn.Module,
  path: Path,
  bundle_id: str,
  provenance: dict,
  actuator_names: list[str],
  inputs=None,
  outputs=None,
) -> str:
  """ONNX export with the embedded Miniverse 0.3 simulation contract (the
  Stage 1 contract layout with this model's tensors). The first output is the
  actuator targets. ``inputs``/``outputs`` default to ``interface()``."""
  if inputs is None:
    inputs, outputs = interface()
  graph = graph.cpu().eval()
  path.parent.mkdir(parents=True, exist_ok=True)
  fast = torch.backends.mha.get_fastpath_enabled()
  torch.backends.mha.set_fastpath_enabled(False)  # fused MHA has no ONNX lowering
  try:
    with torch.no_grad():
      torch.onnx.export(
        graph,
        _example(inputs),
        str(path),
        input_names=[n for n, _, _ in inputs],
        output_names=[n for n, _ in outputs],
        opset_version=18,
        dynamo=False,
      )
  finally:
    torch.backends.mha.set_fastpath_enabled(fast)
  model = onnx.load(str(path))
  contract = dict(
    schemaVersion="0.3",
    precision="fp32",  # in-graph sampler CDF must stay fp32
    contractHash="",
    modelSha256=hashlib.sha256(model.SerializeToString(deterministic=True)).hexdigest(),
    skeletonId=bundle_id,
    compatibleSceneContractHashes=[hashlib.sha256(bundle_id.encode()).hexdigest()],
    backends=[dict(id="mujoco-cpu", versionRange=">=3.3,<4", providers=["CPUExecutionProvider"])],
    opset=18,
    execution=dict(kind="singlePolicyStep"),
    rates=dict(
      physicsHz=round(1 / PHYSICS_DT),
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
  contract["outputs"][0].update(
    role="actuatorTargets",
    controlMode="position",
    clip=[-1000000.0, 1000000.0],
    actuators=actuator_names,
    failsafe=[G1.DEFAULT_JOINT_POS[n] for n in G1.ACTION_JOINT_NAMES],
  )
  contract["contractHash"] = hashlib.sha256(
    E.canonical({k: v for k, v in contract.items() if k != "contractHash"})
  ).hexdigest()
  for suffix, value in (
    ("", E.canonical(contract).decode()),
    ("_schema_version", "0.3"),
    ("_hash", contract["contractHash"]),
  ):
    prop = model.metadata_props.add()
    prop.key, prop.value = E.CONTRACT_KEY + suffix, value
  onnx.save(model, str(path))
  return E.sha256(path)


def build_bundle(
  out: Path,
  graph: PriorPolicyGraph,
  starts: MotionLibrary,
  num_seeds: int,
  bundle_id: str,
  name: str,
  provenance: dict,
  metadata: dict,
  embodiment_from: Path | None = None,
) -> dict:
  """Write the source bundle directory and its .mini archive."""
  if out.exists():
    shutil.rmtree(out)
  if embodiment_from is None:
    actuators = E.write_embodiment(out / "embodiment")
  else:
    shutil.copytree(embodiment_from, out / "embodiment")
    actuators = [f"{n}_actuator" for n in G1.ACTION_JOINT_NAMES]
  model_path = out / "models" / "prior.onnx"
  onnx_sha = export_model(graph, model_path, bundle_id, provenance, actuators)
  labels = [c["name"] for c in starts.clips]
  constants = dict(
    START_LABELS=labels,
    RESET_STATES=E.reset_states(starts),
    JOINT_NAMES=list(G1.ACTION_JOINT_NAMES),
    FREE_JOINT=E.FREE_JOINT,
    OBS_DIM=O.OBS_DIM,
    ACTION_DIM=O.ACTION_DIM,
    HISTORY=O.HISTORY,
    NUM_TOKENS=P.NUM_TOKENS,
  )
  template = (Path(__file__).parent / "miniverse_prior_policy.py").read_text()
  source = "\n".join(f"{k} = {v!r}" for k, v in constants.items()) + "\n\n" + template
  assert len(source.encode()) < 64 * 1024, len(source.encode())
  (out / "policy.py").write_text(source)
  assert round(1 / PHYSICS_DT) // DECIMATION == POLICY_HZ

  def switcher(cid, label, items):
    command = dict(
      id=cid,
      kind="scalar",
      label=label,
      unit="choice",
      default=[0],
      range=[0, len(items) - 1],
      step=1,
      frame="selection",
      sliceLength=1,
      update="on-release",
      restart="episode",
    )
    ui = dict(
      id=f"{cid}-selector",
      renderer="builtin/switcher",
      commandId=cid,
      options=dict(items=[dict(value=i, label=v) for i, v in enumerate(items)], wrap=True),
    )
    return command, ui

  start_cmd, start_ui = switcher("start", "Start pose", labels)
  seed_cmd, seed_ui = switcher("seed", "Sampler seed", [f"seed {i}" for i in range(num_seeds)])
  bundle = dict(
    version="v1",
    id=bundle_id,
    name=name,
    description=(
      "Generative motion prior: samples motion codes from the robot's current sensing and "
      "drives the frozen tracker decoder. Choose a start pose and a sampler seed."
    ),
    primarySimulator="mujoco",
    embodiment=dict(kind="mjcf", path="embodiment/robot.xml"),
    models=[dict(id="prior")],
    program=dict(apiVersion="dhr.python-policy/v1", entrypoint="policy:Policy"),
    commands=[start_cmd, seed_cmd],
    ui=dict(components=[start_ui, seed_ui]),
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
    archive_sha256=E.sha256(archive),
    onnx=str(model_path),
    onnx_sha256=onnx_sha,
    policy_sha256=E.sha256(out / "policy.py"),
  )


@torch.no_grad()
def parity(graph: PriorPolicyGraph, onnx_path: Path, samples: list[tuple]) -> dict[str, float]:
  """Max native/ONNX Runtime differences per output over ``samples`` of
  (robot_state, past_obs, prev_action, reset, uniform)."""
  sess = _session(onnx_path)
  graph = graph.cpu().eval()
  inputs, outputs = interface()
  worst = {n: 0.0 for n, _ in outputs}
  token_mismatch = 0
  for sample in samples:
    native = graph(*sample)
    feeds = {n: t.numpy() for (n, _, _), t in zip(inputs, sample, strict=True)}
    got = sess.run([n for n, _ in outputs], feeds)
    for (n, _), a, b in zip(outputs, native, got, strict=True):
      worst[n] = max(worst[n], float(np.abs(a.numpy() - b).max()))
    token_mismatch += int((native[4].numpy() != got[4]).any())
  return {f"export/parity_max_abs_{n}": v for n, v in worst.items()} | {
    "export/parity_token_mismatch_samples": float(token_mismatch),
    "export/parity_samples": float(len(samples)),
  }


def parity_samples(starts: MotionLibrary, seed: int, count: int = 32) -> list[tuple]:
  gen = torch.Generator().manual_seed(seed)
  out = []
  for i in range(count):
    m = i % starts.num_clips
    f = int(torch.randint(0, int(starts.lengths[m]), (1,), generator=gen))
    s = starts.state(torch.tensor([m]), torch.tensor([f])).float().cpu()
    s = s + 0.02 * torch.randn(s.shape, generator=gen)
    out.append(
      (
        s,
        torch.randn(1, (O.HISTORY - 1) * O.OBS_DIM, generator=gen),
        torch.randn(1, O.ACTION_DIM, generator=gen),
        torch.tensor([[float(i % 4 == 0)]]),
        torch.rand(1, P.NUM_TOKENS, generator=gen),
      )
    )
  return out


def latency(onnx_path: Path, repeats: int = 50) -> dict[str, float]:
  """Batch-1 ONNX Runtime CPU latency of one control step on this host."""
  import time

  sess = _session(onnx_path)
  inputs = interface()[0]
  feeds = {n: t.numpy() for (n, _, _), t in zip(inputs, _example(inputs), strict=True)}
  for _ in range(3):
    sess.run(None, feeds)
  t0 = time.perf_counter()
  for _ in range(repeats):
    sess.run(None, feeds)
  return {"export/onnx_step_latency_ms": 1000 * (time.perf_counter() - t0) / repeats}


@torch.no_grad()
def closed_loop_check(
  bundle_dir: Path,
  starts: MotionLibrary,
  prior: P.CodePrior,
  tracker: Tracker,
  cfg,
  seconds: float = 5.0,
  seeds: tuple[int, ...] = (0, 1),
) -> dict:
  """Run the bundle's own policy.py and ONNX model in native MuJoCo.

  For every start pose and the given sampler seeds, reset through
  ``reset_state`` and step at 50 Hz with 4 physics substeps. At each step the
  training-code path (sort-based nucleus sampling, ``Tracker.decode``) is
  recomputed from the same state, previous action, history and the
  controller's uniforms, and compared with the exported result. A rollout
  fails when the pelvis falls below ``cfg.fall_height_m``.
  """
  import mujoco

  # The embodiment has no ground (Miniverse supplies it): add the training
  # plane, as the Stage 1 check does, or the robot free-falls.
  spec = mujoco.MjSpec.from_file(str(bundle_dir / "embodiment" / "robot.xml"))
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05])
  model = spec.compile()
  data = mujoco.MjData(model)
  sess = _session(bundle_dir / "models" / "prior.onnx")

  class Model:
    def run(self, feeds):
      names = [o.name for o in sess.get_outputs()]
      return dict(zip(names, sess.run(names, feeds), strict=True))

  class StepResult:
    def __init__(self, actuation, metrics=None, **_):
      self.actuation, self.metrics = actuation, metrics or {}

  scope = dict(np=np, ControllerInitialState=dict, StepResult=StepResult)
  exec(compile((bundle_dir / "policy.py").read_text(), "policy.py", "exec"), scope)
  view = E._Slices(model)
  prior = copy.deepcopy(prior).cpu().eval()
  tracker = copy.deepcopy(tracker).cpu().eval()
  default_q = _default_q()
  results = []
  for start in range(starts.num_clips):
    for seed in seeds:
      policy = scope["Policy"]()
      ctx = type("Ctx", (), dict(models={"prior": Model()}))()
      policy.initialize(ctx)
      commands = {"start": [float(start)], "seed": [float(seed)]}
      init = policy.reset_state(commands)
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
      hist, prev = None, torch.zeros(1, O.ACTION_DIM)
      token_mismatch, max_diff, fell_at, steps = 0, 0.0, -1, 0
      min_height = float("inf")
      for i in range(round(seconds * POLICY_HZ)):
        result = policy.step(E._Step(data, view, commands, i))
        s = torch.from_numpy(policy.robot_state(E._Step(data, view, {}, i).sim_data))
        obs = O.actor_observation(s, prev, torch.zeros(1, 27), default_q)
        hist = (
          obs.repeat(1, O.HISTORY) if hist is None else torch.cat((hist[:, O.OBS_DIM :], obs), -1)
        )
        uniform = torch.from_numpy(policy.last_uniform)
        tokens, _ = prior.generate(obs, uniform, cfg.top_p, cfg.temperature)
        action = tracker.decode(P.tokens_to_codes(tokens) / FSQ_HALF_LEVELS, hist)
        action = action.clamp(-cfg.action_clip, cfg.action_clip)
        exported_tokens = torch.from_numpy(policy.last_tokens).long()
        if not torch.equal(tokens, exported_tokens):
          token_mismatch += 1
        else:
          max_diff = max(max_diff, float((action - torch.from_numpy(policy.prev)).abs().max()))
        prev = torch.from_numpy(policy.prev.copy())
        data.ctrl[:] = result.actuation
        for _ in range(DECIMATION):
          mujoco.mj_step(model, data)
        steps += 1
        height = float(data.qpos[2])
        min_height = min(min_height, height)
        if height < cfg.fall_height_m:
          fell_at = steps
          break
      results.append(
        dict(
          start=starts.clips[start]["name"],
          seed=seed,
          steps=steps,
          fell_at_step=fell_at,
          min_pelvis_height_m=min_height,
          token_mismatch_steps=token_mismatch,
          max_abs_action_diff_vs_training_code=max_diff,
        )
      )
  return dict(
    rollouts=results,
    survival_rate=float(np.mean([r["fell_at_step"] < 0 for r in results])),
    mean_survival_s=float(np.mean([r["steps"] for r in results]) / POLICY_HZ),
    token_mismatch_steps=int(sum(r["token_mismatch_steps"] for r in results)),
    total_steps=int(sum(r["steps"] for r in results)),
    max_abs_action_diff=max(r["max_abs_action_diff_vs_training_code"] for r in results),
  )
