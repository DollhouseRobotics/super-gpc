"""Joystick CoLA ONNX export, Miniverse source bundle and parity checks.

One exported ``policy`` model is a single 50 Hz step: the clean sensor frame
(training ``actor_observation`` of the robot state and the executed previous
action, no calibration bias) and the 7-D local command -> four full-support
categorical tokens from the command-adapted prior (uniforms supplied by the
controller) -> 16 FSQ codes -> the frozen decoder over the ten-frame history
-> joint targets. The CoLA sees the neck at the pose its own neck action
commanded; the executed neck targets point the head along the commanded gaze
(``mdp.neck_targets``). Inputs and outputs extend the Stage 2 prior export
with the ``command`` input. Bundles are built for idle-gaze policies; without a
gaze config (``cfg`` None) the graph leaves the neck to
the decoder.
"""

from __future__ import annotations

import copy
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1
from super_gpc.rl import prior as P
from super_gpc.rl.cola import ColaPolicy
from super_gpc.tasks.steering import mdp as M
from super_gpc.tasks.steering.rl.cola_cfg import ColaCfg
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.rl import export as E
from super_gpc.tasks.tracking.rl import stage2_export as X
from super_gpc.tasks.tracking.tracking_env_cfg import DECIMATION


def interface() -> tuple[tuple, tuple]:
  """The Stage 2 prior contract for ``joints`` plus the ``command`` input."""
  inputs, outputs = X.interface()
  return inputs + (("command", "float32", [1, ColaCfg().command_dim]),), outputs


INPUTS, OUTPUTS = interface()


class ColaPolicyGraph(nn.Module):
  def __init__(self, policy: ColaPolicy, cfg: ColaCfg):
    super().__init__()
    self.policy = copy.deepcopy(policy).cpu().eval()
    self.neck_max = (cfg.neck_pan_max, cfg.neck_tilt_max)
    names = G1.ACTION_JOINT_NAMES
    self.register_buffer("default_q", torch.tensor([G1.DEFAULT_JOINT_POS[n] for n in names]))
    self.register_buffer("scale", torch.tensor([G1.ACTION_SCALE[n] for n in names]))

  def forward(self, robot_state, past_obs, prev_action, reset, uniform, command):
    keep = 1.0 - reset
    prev = prev_action * keep
    seen = M.mask_neck(robot_state, prev, self.default_q, self.scale)
    bias = torch.zeros_like(prev_action)
    obs = O.actor_observation(seen, prev, bias, self.default_q)
    tokens, _ = self.policy.sample(obs, command, uniform)
    codes = P.tokens_to_codes(tokens)
    past = keep * past_obs + reset * obs.repeat(1, O.HISTORY - 1)
    history = torch.cat((past, obs), -1)
    action = self.policy.decode(tokens, history)
    target = self.default_q + self.scale * action
    neck = M.neck_targets(robot_state, command, *self.neck_max)  # analytic neck
    target = torch.cat((target[:, : M.NECK.start], neck), -1)
    return target, action, history, codes, tokens.float()


def initial_state() -> dict:
  """Default standing pose at the training reset height, zero yaw."""
  return dict(
    root_position=list(G1.INIT_STATE.pos),
    root_rotation=[0.0, 0.0, 0.0, 1.0],  # XYZW
    root_linear_velocity=[0.0, 0.0, 0.0],
    root_angular_velocity=[0.0, 0.0, 0.0],
    joint_positions={n: [G1.DEFAULT_JOINT_POS[n]] for n in G1.ACTION_JOINT_NAMES},
    joint_velocities={n: [0.0] for n in G1.ACTION_JOINT_NAMES},
  )


def build_bundle(
  out: Path,
  graph: ColaPolicyGraph,
  bundle_id: str,
  provenance: dict,
  metadata: dict,
  speed_max: float,
  embodiment_from: Path | None = None,
  gaze: dict | None = None,
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
  constants = dict(
    INITIAL_STATE=initial_state(),
    JOINT_NAMES=list(G1.ACTION_JOINT_NAMES),
    FREE_JOINT=E.FREE_JOINT,
    OBS_DIM=O.OBS_DIM,
    ACTION_DIM=O.ACTION_DIM,
    HISTORY=O.HISTORY,
    NUM_TOKENS=P.NUM_TOKENS,
    SPEED_MAX=speed_max,
    GAZE=gaze or gaze_constants(ColaCfg()),
  )
  template = (Path(__file__).parent / "miniverse_cola_policy.py").read_text()
  source = "\n".join(f"{k} = {v!r}" for k, v in constants.items()) + "\n\n" + template
  (out / "policy.py").write_text(source)
  bundle = dict(
    version="v1",
    id=bundle_id,
    name="GPC joystick steering (CoLA)",
    description=(
      "Command-adapted generative prior: the left stick sets walking velocity in the "
      "commanded facing frame, the right stick or slider sets facing. Standing still, "
      "the robot looks around by itself (auto look) or where the look stick points."
    ),
    primarySimulator="mujoco",
    embodiment=dict(kind="mjcf", path="embodiment/robot.xml"),
    models=[dict(id="policy")],
    program=dict(apiVersion="dhr.python-policy/v1", entrypoint="policy:Policy"),
    commands=[
      dict(
        id="movement",
        kind="joystick2d",
        default=[0, 0],
        range=[-speed_max, speed_max],
        step=0.05,
        frame="base",
        sliceLength=2,
        update="continuous",
      ),
      dict(
        id="facing",
        kind="scalar",
        label="Facing",
        unit="rad",
        default=[0],
        range=[-float(np.pi), float(np.pi)],
        step=0.01,
        frame="base",
        sliceLength=1,
        update="continuous",
      ),
      dict(
        id="look",
        kind="joystick2d",
        label="Look (standing)",
        default=[0, 0],
        range=[-1, 1],
        step=0.05,
        frame="base",
        sliceLength=2,
        update="continuous",
      ),
      dict(
        id="auto_look",
        kind="scalar",
        label="Auto look",
        default=[1],
        range=[0, 1],
        step=1,
        frame="base",
        sliceLength=1,
        update="continuous",
      ),
    ],
    ui=dict(
      components=[
        dict(id="movement-control", renderer="builtin/joystick2d", commandId="movement"),
        dict(id="facing-control", renderer="builtin/scalar", commandId="facing"),
        dict(id="look-control", renderer="builtin/joystick2d", commandId="look"),
        dict(id="auto-look-control", renderer="builtin/scalar", commandId="auto_look"),
        dict(
          id="movement-gamepad",
          renderer="builtin/gamepad",
          commandId="movement",
          options=dict(
            bindings=[
              dict(component=0, source="axis", input="left-stick-x"),
              dict(component=1, source="axis", input="left-stick-y", invert=True),
            ]
          ),
        ),
        dict(
          id="facing-gamepad",
          renderer="builtin/gamepad",
          commandId="facing",
          options=dict(bindings=[dict(source="angle", input="right-stick", invert=True)]),
        ),
      ]
    ),
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


def parity_samples(seed: int, count: int = 32) -> list[tuple]:
  gen = torch.Generator().manual_seed(seed)
  base = torch.zeros(1, 67)
  base[0, 2], base[0, 3] = 0.76, 1.0
  base[0, 13:40] = torch.tensor([G1.DEFAULT_JOINT_POS[n] for n in G1.ACTION_JOINT_NAMES])
  out = []
  for i in range(count):
    s = base + 0.05 * torch.randn(1, 67, generator=gen)
    angle = torch.rand(1, generator=gen) * 6.2832
    v = 1.5 * torch.rand(1, 2, generator=gen) * float(i % 4 != 0)
    gaze = angle + torch.rand(1, generator=gen) - 0.5
    cmd = torch.cat(
      (
        v,
        angle.cos()[:, None],
        angle.sin()[:, None],
        gaze.cos()[:, None],
        gaze.sin()[:, None],
        0.5 * torch.rand(1, 1, generator=gen) - 0.3,
      ),
      -1,
    )
    out.append(
      (
        s,
        torch.randn(1, (O.HISTORY - 1) * O.OBS_DIM, generator=gen),
        torch.randn(1, O.ACTION_DIM, generator=gen),
        torch.tensor([[float(i % 4 == 0)]]),
        torch.rand(1, P.NUM_TOKENS, generator=gen),
        cmd,
      )
    )
  return out


@torch.no_grad()
def parity(graph: ColaPolicyGraph, onnx_path: Path, samples: list[tuple]) -> dict[str, float]:
  sess = X._session(onnx_path)
  graph = graph.cpu().eval()
  inputs, outputs = interface()
  worst = {n: 0.0 for n, _ in outputs}
  mismatch = 0
  for sample in samples:
    native = graph(*sample)
    feeds = {n: t.numpy() for (n, _, _), t in zip(inputs, sample, strict=True)}
    got = sess.run([n for n, _ in outputs], feeds)
    if (native[4].numpy() != got[4]).any():
      mismatch += 1
      continue  # a different token makes the downstream outputs incomparable
    for (n, _), a, b in zip(outputs, native, got, strict=True):
      worst[n] = max(worst[n], float(np.abs(a.numpy() - b).max()))
  return {f"export/parity_max_abs_{n}": v for n, v in worst.items()} | {
    "export/parity_token_mismatch_samples": float(mismatch),
    "export/parity_samples": float(len(samples)),
  }


def latency(onnx_path: Path, repeats: int = 50) -> dict[str, float]:
  import time

  sess = X._session(onnx_path)
  sample = parity_samples(0, 1)[0]
  feeds = {n: t.numpy() for (n, _, _), t in zip(interface()[0], sample, strict=True)}
  for _ in range(3):
    sess.run(None, feeds)
  t0 = time.perf_counter()
  for _ in range(repeats):
    sess.run(None, feeds)
  return {"export/onnx_step_latency_ms": 1000 * (time.perf_counter() - t0) / repeats}


def gaze_constants(cfg: ColaCfg) -> dict:
  """The training look-target distribution, for the controller's auto look."""
  return dict(
    yaw_max=cfg.gaze_yaw_max,
    pitch_min=cfg.gaze_pitch_min,
    pitch_max=cfg.gaze_pitch_max,
    center_probability=cfg.gaze_center_probability,
    min_steps=cfg.gaze_min_steps,
    rate=cfg.gaze_rate,
    max_steps=cfg.gaze_max_steps,
    near_zero_speed=cfg.near_zero_speed,
  )


SCRIPT = (  # (seconds, movement [right, forward], facing rad); stops auto-look
  (6.0, (0.0, 0.0), 0.0),
  (3.0, (0.0, 0.8), 0.0),
  (3.0, (0.6, 0.0), 0.0),
  (2.0, (0.0, 0.0), 1.5),
)


@torch.no_grad()
def closed_loop_check(
  bundle_dir: Path,
  policy: ColaPolicy,
  fall_height: float = 0.45,
  script=SCRIPT,
  on_step=None,
) -> dict:
  """Run the bundle's own policy.py and ONNX model in native MuJoCo through a
  scripted joystick sequence (auto look on), recomputing tokens and actions
  with the training-code path at every step from the same state, history,
  command and uniforms. ``on_step(model, data, controller, metrics)`` is
  called after every policy step (rendering)."""
  import mujoco

  spec = mujoco.MjSpec.from_file(str(bundle_dir / "embodiment" / "robot.xml"))
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05])
  model = spec.compile()
  data = mujoco.MjData(model)
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
  policy = copy.deepcopy(policy).cpu().eval()
  default_q = X._default_q()
  scale = torch.tensor([G1.ACTION_SCALE[n] for n in G1.ACTION_JOINT_NAMES])
  ctrl = scope["Policy"]()
  ctrl.initialize(type("Ctx", (), dict(models={"policy": Model()}))())
  init = ctrl.initial_state()
  mujoco.mj_resetData(model, data)
  data.qpos[0:3] = init["root_position"]
  x, y, z, w = init["root_rotation"]
  data.qpos[3:7] = (w, x, y, z)
  for n in G1.ACTION_JOINT_NAMES:
    data.qpos[view.joint_slice(n)] = init["joint_positions"][n]
  mujoco.mj_forward(model, data)
  hist, prev = None, torch.zeros(1, O.ACTION_DIM)
  mismatch, max_diff, fell_at, i = 0, 0.0, -1, 0
  segments = []
  for seconds, stick, facing in script:
    commands = {"movement": list(stick), "facing": [facing], "look": [0.0, 0.0], "auto_look": [1.0]}
    vel, look = [], []
    for _ in range(round(seconds * 50)):
      result = ctrl.step(E._Step(data, view, commands, i))
      s = torch.from_numpy(ctrl.robot_state(E._Step(data, view, {}, i).sim_data))
      s = M.mask_neck(s, prev, default_q, scale)
      obs = O.actor_observation(s, prev, torch.zeros(1, 27), default_q)
      hist = (
        obs.repeat(1, O.HISTORY) if hist is None else torch.cat((hist[:, O.OBS_DIM :], obs), -1)
      )
      cmd = torch.from_numpy(ctrl.last_command)[:, : policy.adapters[0].gamma.in_features]
      tokens, _ = policy.sample(obs, cmd, torch.from_numpy(ctrl.last_uniform))
      action = policy.decode(tokens, hist)
      if not torch.equal(tokens, torch.from_numpy(ctrl.last_tokens).long()):
        mismatch += 1
      else:
        max_diff = max(max_diff, float((action - torch.from_numpy(ctrl.prev)).abs().max()))
      prev = torch.from_numpy(ctrl.prev.copy())
      data.ctrl[:] = result.actuation
      for _ in range(DECIMATION):
        mujoco.mj_step(model, data)
      i += 1
      vel.append([result.metrics["velocity_forward_mps"], result.metrics["velocity_left_mps"]])
      look.append(result.metrics["look_yaw_rad"])
      if on_step is not None:
        on_step(model, data, ctrl, result.metrics)
      if data.qpos[2] < fall_height and fell_at < 0:
        fell_at = i
    v = np.asarray(vel[len(vel) // 2 :]).mean(0)
    segments.append(
      dict(
        movement=list(stick),
        facing=facing,
        mean_body_velocity_second_half=v.tolist(),
        look_yaw_range_rad=float(np.ptp(look)),
      )
    )
  return dict(
    segments=segments,
    steps=i,
    fell_at_step=fell_at,
    token_mismatch_steps=mismatch,
    max_abs_action_diff=max_diff,
  )
