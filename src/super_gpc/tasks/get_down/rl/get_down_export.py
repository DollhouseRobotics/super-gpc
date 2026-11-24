"""Get-down CoLA ONNX export, Miniverse source bundle and parity checks.

One exported ``policy`` model is a single 50 Hz step: robot state -> the
clean sensor frame (no calibration bias) and the clean task frame
(``mdp.task_frame`` through in-graph forward kinematics) -> the five-frame
task input (four carried frames, reset padding) -> four full-support
categorical tokens from the task-adapted prior (uniforms supplied by the
controller) -> 16 FSQ codes -> the frozen decoder over the ten-frame history
-> joint targets. Inputs and outputs extend the Stage 2 prior export with
``past_task`` and ``task_history``.

The Miniverse demonstration starts from named joystick-rollout states with a
selectable push (the evaluation's push conditions) and a controller switch
between the get-down policy and an approximate damping mode.
"""

from __future__ import annotations

import copy
import json
import shutil
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1
from super_gpc.rl import prior as P
from super_gpc.rl.cola import ColaPolicy
from super_gpc.tasks.get_down import mdp as M
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.rl import export as E
from super_gpc.tasks.tracking.rl import stage2_export as X
from super_gpc.tasks.tracking.tracking_env_cfg import DECIMATION

PAST_TASK_DIM = (M.TASK_FRAMES - 1) * M.TASK_FRAME_DIM


def interface() -> tuple[tuple, tuple]:
  """The Stage 2 prior contract for ``joints`` plus the task input and output."""
  inputs, outputs = X.interface()
  return (
    inputs + (("past_task", "float32", [1, PAST_TASK_DIM]),),
    outputs + (("task_history", M.TASK_DIM),),
  )


INPUTS, OUTPUTS = interface()
CONTROLLERS = ("Get down", "Damping (approximate)")
NAME = "GPC get down (CoLA)"
DESCRIPTION = (
  "Task-adapted generative prior that lowers the robot onto its back slowly and "
  "naturally from wherever it is: standing, walking or just pushed. Trained with an "
  "adversarial motion prior on a smooth get-down-on-back reference, a pelvis-height "
  "reward and floor-impact penalties. Switch to damping mode to compare."
)


class GetDownPolicyGraph(nn.Module):
  def __init__(self, policy: ColaPolicy):
    super().__init__()
    self.policy = copy.deepcopy(policy).cpu().eval()
    self.kin = M.build_kinematics("cpu")
    names = G1.ACTION_JOINT_NAMES
    self.register_buffer("default_q", torch.tensor([G1.DEFAULT_JOINT_POS[n] for n in names]))
    self.register_buffer("scale", torch.tensor([G1.ACTION_SCALE[n] for n in names]))

  def forward(self, robot_state, past_obs, prev_action, reset, uniform, past_task):
    keep = 1.0 - reset
    obs = O.actor_observation(
      robot_state, prev_action * keep, torch.zeros_like(prev_action), self.default_q
    )
    frame = M.task_frame(self.kin(robot_state), robot_state)
    task = torch.cat((keep * past_task + reset * frame.repeat(1, M.TASK_FRAMES - 1), frame), -1)
    tokens, _ = self.policy.sample(obs, task, uniform)
    codes = P.tokens_to_codes(tokens)
    past = keep * past_obs + reset * obs.repeat(1, O.HISTORY - 1)
    history = torch.cat((past, obs), -1)
    action = self.policy.decode(tokens, history)
    return (
      self.default_q + self.scale * action,
      action,
      history,
      codes,
      tokens.float(),
      task,
    )


def named_starts(bank: dict) -> tuple[list[str], list[dict]]:
  """Miniverse start states from the bank's named examples, at heading zero."""
  labels, states = [], []
  for name, i in bank["examples"].items():
    state = bank["states"][i : i + 1]
    s = M.rotate_start(state, -M.heading(state))[0].double().tolist()
    q = s[3:7]
    labels.append(name)
    states.append(
      dict(
        root_position=s[0:3],
        root_rotation=[q[1], q[2], q[3], q[0]],  # XYZW
        root_linear_velocity=s[7:10],
        root_angular_velocity=s[10:13],  # pelvis body frame
        joint_positions={
          n: [float(v)] for n, v in zip(G1.ACTION_JOINT_NAMES, s[13:40], strict=True)
        },
        joint_velocities={
          n: [float(v)] for n, v in zip(G1.ACTION_JOINT_NAMES, s[40:67], strict=True)
        },
      )
    )
  return labels, states


def build_bundle(
  out: Path,
  graph: GetDownPolicyGraph,
  bundle_id: str,
  provenance: dict,
  metadata: dict,
  bank: dict,
  pushes: list[tuple[str, float, float]],
  embodiment_from: Path | None = None,
) -> dict:
  if out.exists():
    shutil.rmtree(out)
  if embodiment_from is None:
    actuators = E.write_embodiment(out / "embodiment")
  else:
    shutil.copytree(embodiment_from, out / "embodiment")
    actuators = [f"{n}_actuator" for n in G1.ACTION_JOINT_NAMES]
  model_path = out / "models" / "policy.onnx"
  inputs, outputs = interface()
  onnx_sha = X.export_model(graph, model_path, bundle_id, provenance, actuators, inputs, outputs)
  start_labels, start_states = named_starts(bank)
  constants = dict(
    START_LABELS=start_labels,
    RESET_STATES=start_states,
    PUSH_LABELS=[p[0] for p in pushes],
    PUSHES=[[p[1], p[2]] for p in pushes],
    CONTROLLERS=list(CONTROLLERS),
    JOINT_NAMES=list(G1.ACTION_JOINT_NAMES),
    FREE_JOINT=E.FREE_JOINT,
    OBS_DIM=O.OBS_DIM,
    ACTION_DIM=O.ACTION_DIM,
    HISTORY=O.HISTORY,
    NUM_TOKENS=P.NUM_TOKENS,
    PAST_TASK_DIM=PAST_TASK_DIM,
    TASK_FRAME_DIM=M.TASK_FRAME_DIM,
  )
  template = (Path(__file__).parent / "miniverse_get_down_policy.py").read_text()
  source = "\n".join(f"{k} = {v!r}" for k, v in constants.items()) + "\n\n" + template
  assert len(source.encode()) < 64 * 1024, len(source.encode())
  (out / "policy.py").write_text(source)

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

  push_labels = [p[0].replace("_", " ") + ("" if p[0] == "stand" else " m/s") for p in pushes]
  push_labels[0] = "no push"
  cmds = [
    switcher("start", "Start state", start_labels),
    switcher("push", "Push", push_labels),
    switcher("controller", "Controller", list(CONTROLLERS)),
  ]
  bundle = dict(
    version="v1",
    id=bundle_id,
    name=NAME,
    description=DESCRIPTION,
    primarySimulator="mujoco",
    embodiment=dict(kind="mjcf", path="embodiment/robot.xml"),
    models=[dict(id="policy")],
    program=dict(apiVersion="dhr.python-policy/v1", entrypoint="policy:Policy"),
    commands=[c for c, _ in cmds],
    ui=dict(components=[u for _, u in cmds]),
    viewer=dict(camera=dict(framingScale=1.3)),
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


def parity_samples(bank: dict, seed: int, count: int = 32) -> list[tuple]:
  gen = torch.Generator().manual_seed(seed)
  states = bank["states"]
  out = []
  for i in range(count):
    j = int(torch.randint(0, len(states), (1,), generator=gen))
    s = M.rotate_start(states[j : j + 1], torch.rand(1, generator=gen) * 6.2832)
    s = s + 0.02 * torch.randn(s.shape, generator=gen)
    out.append(
      (
        s.float(),
        torch.randn(1, (O.HISTORY - 1) * O.OBS_DIM, generator=gen),
        torch.randn(1, O.ACTION_DIM, generator=gen),
        torch.tensor([[float(i % 4 == 0)]]),
        torch.rand(1, P.NUM_TOKENS, generator=gen),
        torch.randn(1, PAST_TASK_DIM, generator=gen),
      )
    )
  return out


@torch.no_grad()
def parity(graph: GetDownPolicyGraph, onnx_path: Path, samples: list[tuple]) -> dict[str, float]:
  sess = X._session(onnx_path)
  graph = graph.cpu().eval()
  worst = {n: 0.0 for n, _ in OUTPUTS}
  mismatch = 0
  for sample in samples:
    native = graph(*sample)
    feeds = {n: t.numpy() for (n, _, _), t in zip(INPUTS, sample, strict=True)}
    got = sess.run([n for n, _ in OUTPUTS], feeds)
    if (native[4].numpy() != got[4]).any():
      mismatch += 1
      continue  # a different token makes the downstream outputs incomparable
    for (n, _), a, b in zip(OUTPUTS, native, got, strict=True):
      worst[n] = max(worst[n], float(np.abs(a.numpy() - b).max()))
  return {f"export/parity_max_abs_{n}": v for n, v in worst.items()} | {
    "export/parity_token_mismatch_samples": float(mismatch),
    "export/parity_samples": float(len(samples)),
  }


def latency(onnx_path: Path, sample: tuple, repeats: int = 50) -> dict[str, float]:
  import time

  sess = X._session(onnx_path)
  feeds = {n: t.numpy() for (n, _, _), t in zip(INPUTS, sample, strict=True)}
  for _ in range(3):
    sess.run(None, feeds)
  t0 = time.perf_counter()
  for _ in range(repeats):
    sess.run(None, feeds)
  return {"export/onnx_step_latency_ms": 1000 * (time.perf_counter() - t0) / repeats}


def _floor_force(model, data, floor: int) -> float:
  """Magnitude of the vector sum of all floor contact forces (world frame)."""
  total = np.zeros(3)
  f = np.zeros(6)
  for i in range(data.ncon):
    c = data.contact[i]
    if floor not in (c.geom1, c.geom2):
      continue
    mujoco.mj_contactForce(model, data, i, f)
    world = c.frame.reshape(3, 3).T @ f[:3]  # on geom2, along the geom1 -> geom2 normal
    total += world if c.geom1 == floor else -world
  return float(np.linalg.norm(total))


CHECKS = ((0, "stand", 0), (0, "fwd_1", 0), (0, "back_1", 0), (0, "fwd_1", 1), (0, "back_1", 1))


@torch.no_grad()
def closed_loop_check(
  bundle_dir: Path,
  policy: ColaPolicy,
  pushes: list,
  seconds: float = 4.0,
) -> dict:
  """Run the bundle's own policy.py and ONNX model in native MuJoCo from the
  first named start under several pushes and both controllers. In get-down
  mode the training-code path recomputes tokens and actions at every step
  from the same state, history, task input and uniforms. Reports the peak
  floor contact force over all physics substeps and the settled pose."""
  spec = mujoco.MjSpec.from_file(str(bundle_dir / "embodiment" / "robot.xml"))
  spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05])
  model = spec.compile()
  floor = model.geom("floor").id
  data = mujoco.MjData(model)
  torso = model.body("torso_link").id
  sess = X._session(bundle_dir / "models" / "policy.onnx")

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
  graph = GetDownPolicyGraph(policy)
  results = []
  for start, push_name, controller in CHECKS:
    push = [p[0] for p in pushes].index(push_name)
    ctrl = scope["Policy"]()
    ctrl.initialize(type("Ctx", (), dict(models={"policy": Model()}))())
    commands = {"start": [float(start)], "push": [float(push)], "controller": [float(controller)]}
    init = ctrl.reset_state(commands)
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
    past_obs = torch.zeros(1, (O.HISTORY - 1) * O.OBS_DIM)
    past_task = torch.zeros(1, PAST_TASK_DIM)
    prev = torch.zeros(1, O.ACTION_DIM)
    mismatch, max_diff, peak, heights = 0, 0.0, 0.0, []
    steps = round(seconds * 50)
    for i in range(steps):
      result = ctrl.step(E._Step(data, view, commands, i))
      if controller == 0:
        s = torch.from_numpy(ctrl.robot_state(E._Step(data, view, {}, i).sim_data))
        reset = torch.tensor([[1.0 if i == 0 else 0.0]])
        ref = graph(s, past_obs, prev, reset, torch.from_numpy(ctrl.last_uniform), past_task)
        if not torch.equal(ref[4], torch.from_numpy(ctrl.last_tokens)):
          mismatch += 1
        else:
          max_diff = max(max_diff, float((ref[1] - torch.from_numpy(ctrl.prev)).abs().max()))
        past_obs = torch.from_numpy(ctrl.past.copy())
        past_task = torch.from_numpy(ctrl.past_task.copy())
        prev = torch.from_numpy(ctrl.prev.copy())
      data.ctrl[:] = result.actuation
      for _ in range(DECIMATION):
        mujoco.mj_step(model, data)
        peak = max(peak, _floor_force(model, data, floor))
      heights.append(float(data.xpos[torso, 2]))
    tail = heights[-50:]
    results.append(
      dict(
        start=int(start),
        push=pushes[push][0],
        controller=CONTROLLERS[controller],
        peak_floor_force_n=peak,
        final_torso_height_m=float(np.mean(tail)),
        token_mismatch_steps=mismatch,
        max_abs_action_diff=max_diff,
      )
    )
  return dict(rollouts=results, steps=steps)
