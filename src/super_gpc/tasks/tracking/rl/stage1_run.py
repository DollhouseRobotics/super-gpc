"""Stage 1 run lifecycle: provenance, Miniverse run, loop, checkpoints, resume.

A run directory is the durable identity of one training job and is also its
Miniverse run directory (see ``telemetry``). It holds the run binding
(``miniverse.json``: source bundle and provenance), Miniverse and local
metrics, evaluation and preservation records, native policy checkpoints, the
resumable state (``resume.pt``, overwritten atomically), published
checkpoints and their ledger. Restarting with the same directory resumes the
same Miniverse run; it never creates a second run.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata as md
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

import torch

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1
from super_gpc.rl.networks import Tracker, tracker_weights
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.mdp.motion_library import MotionLibrary
from super_gpc.tasks.tracking.rl import export as E
from super_gpc.tasks.tracking.rl import standing as S
from super_gpc.tasks.tracking.rl.stage1 import Stage1Trainer, build_kinematics
from super_gpc.tasks.tracking.rl.stage1_cfg import Stage1Cfg
from super_gpc.tasks.tracking.rl.telemetry import CheckpointPublisher, Telemetry

REPO_ROOT = Path(__file__).resolve().parents[5]


def _git(*args: str) -> str:
  try:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
  except Exception:
    return "unknown"


def provenance(cfg: Stage1Cfg, motions: Path | None, device: str, examples: list[str]) -> dict:
  """``motions`` is None for training that uses no motion corpus (joystick RLFT)."""
  manifest = json.loads((motions / "manifest.json").read_text()) if motions else {}
  hw: dict = {"host": platform.node(), "python": sys.version.split()[0], "device": device}
  if device.startswith("cuda"):
    p = torch.cuda.get_device_properties(device)
    hw |= {
      "gpu": p.name,
      "gpu_memory_gb": round(p.total_memory / 2**30, 1),
      "cuda": torch.version.cuda,
    }
  lock = REPO_ROOT / "uv.lock"
  source = hashlib.sha256()
  for path in sorted((REPO_ROOT / "src").rglob("*.py")):
    source.update(str(path.relative_to(REPO_ROOT)).encode() + b"\0")
    source.update(path.read_bytes() + b"\0")
  return {
    "stage": "1",
    "cfg": cfg.to_dict(),
    "seeds": {"train": cfg.seed, "eval": cfg.eval_seed},
    "code": {
      "git_commit": _git("rev-parse", "HEAD"),
      "git_dirty": bool(_git("status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock")),
      "python_source_sha256": source.hexdigest(),
      "uv_lock_sha256": E.sha256(lock) if lock.exists() else None,
    },
    "dependencies": {
      p: md.version(p)
      for p in (
        "mjlab",
        "mujoco",
        "mujoco-warp",
        "torch",
        "miniverse-sdk",
        "dollhouse-robots",
        "onnx",
        "onnxruntime",
      )
      if _has(p)
    },
    "dataset": {
      k: manifest.get(k)
      for k in (
        "dataset",
        "revision",
        "sourceChecksums",
        "sourceRate",
        "policyRate",
        "splitRule",
        "clips",
        "hours",
        "sourceGroups",
        "excludedClips",
        "qposSha256",
        "qvelSha256",
        "clipsSha256",
      )
    }
    | {"prepared_manifest_sha256": E.sha256(motions / "manifest.json") if motions else None},
    "robot": {
      "id": G1.ROBOT_ID,
      "robots_revision": G1.ROBOTS_REVISION,
      "mjcf_sha256": G1.MJCF_SHA256,
    },
    "hardware": hw,
    "examples": examples,
  }


def _has(package: str) -> bool:
  try:
    md.version(package)
    return True
  except md.PackageNotFoundError:
    return False


def export_policy_checkpoint(
  run_dir: str,
  motions: str,
  example_names: list[str],
  standing_poses: list[dict],
  action_clip: float,
  bundle_id: str,
  job_id: str,
  tracker_state: dict,
  iteration: int,
  native: str,
  native_sha256: str,
  reference_offsets: tuple[int, ...] = O.DEFAULT_REFERENCE_OFFSETS,
) -> tuple[dict, dict]:
  """Export, parity-, closed-loop- and standing-check one saved tracker.

  A pure function of the saved native policy state, so it runs inline or in a
  background CPU process while training continues. Returns the publication
  ledger record and the export metrics for the checkpoint's own step."""
  run_dir_p = Path(run_dir)
  tracker = Tracker(O.ENCODER_DIM, O.HISTORY_DIM, O.ACTION_DIM)
  tracker.load_state_dict(tracker_state)
  tracker.eval().requires_grad_(False)
  kin, kin_torso = build_kinematics("cpu")
  for k in (kin, kin_torso):
    k.requires_grad_(False)
  examples = MotionLibrary(Path(motions), "cpu", clip_names=example_names)
  graph = E.InferenceGraph(tracker, kin_torso, examples, action_clip, reference_offsets)
  it = iteration
  out = run_dir_p / "exports" / f"iter_{it:07d}"
  info = E.build_bundle(
    out,
    graph,
    examples,
    bundle_id,
    "GPC Stage 1 motion tracker",
    {"iteration": it, "job": job_id, "native_sha256": native_sha256},
    {"iteration": it},
    embodiment_from=run_dir_p / "bundle" / "source" / "embodiment",
  )
  samples = []
  gen = torch.Generator().manual_seed(it)
  for i in range(32):
    m = i % examples.num_clips
    f = int(torch.randint(0, int(examples.lengths[m]), (1,), generator=gen))
    s = examples.state(torch.tensor([m]), torch.tensor([f])).float()
    s = s + 0.02 * torch.randn(s.shape, generator=gen)
    samples.append(
      (
        s,
        torch.randn(1, (O.HISTORY - 1) * O.OBS_DIM, generator=gen),
        torch.randn(1, O.ACTION_DIM, generator=gen),
        torch.tensor([[float(i % 4 == 0)]]),
        torch.tensor([m]),
        torch.tensor([f]),
      )
    )
  parity = E.parity(graph, Path(info["onnx"]), samples)
  loop = E.closed_loop_check(
    out, examples, tracker, kin, kin_torso, reference_offsets=reference_offsets
  )
  standing = [
    S.standing_check(
      out / "embodiment" / "robot.xml", standing_poses, tracker, kin_torso, action_clip, noisy
    )
    for noisy in (False, True)
  ]
  record = {
    "step": it,
    "label": f"iter-{it}",
    "onnx": info["onnx"],
    "onnx_sha256": info["onnx_sha256"],
    "native": native,
    "native_sha256": native_sha256,
    "parity": parity,
    "closed_loop": loop,
    "standing": standing,
  }
  (out.parent / f"iter_{it:07d}.json").write_text(json.dumps(record, indent=2))
  metrics = (
    parity
    | {
      "export/closed_loop_success_rate": loop["success_rate"],
      "export/closed_loop_max_action_diff": loop["max_abs_action_diff"],
    }
    | {k: v for s in standing for k, v in s["metrics"].items()}
  )
  # The closed-loop check's local controller class forms a reference cycle that
  # holds its ONNX Runtime session and threads; long-lived export workers rarely
  # run a full collection, so free them here (the four-dataset worker reached 775).
  gc.collect()
  return record, metrics


class Stage1Run:
  def __init__(
    self,
    cfg: Stage1Cfg,
    motions: Path,
    run_dir: Path,
    device: str,
    name: str,
  ):
    self.cfg, self.motions, self.run_dir, self.device = cfg, Path(motions), Path(run_dir), device
    self.run_dir.mkdir(parents=True, exist_ok=True)
    self.job_id = name
    self.trainer = Stage1Trainer(cfg, self.motions, device)
    self.examples_names = list(cfg.example_clips) or E.select_examples(self.motions)
    self.examples = MotionLibrary(self.motions, "cpu", clip_names=self.examples_names)
    self.standing_poses = self._standing_poses()
    self.binding_path = self.run_dir / "miniverse.json"
    self.binding = self._bind_miniverse()
    self.telemetry = Telemetry(self.run_dir, f"super-gpc stage1 {self.job_id}")
    self.publisher = CheckpointPublisher(self.run_dir, self.telemetry.run)
    # Exports, parity, the native closed loop and the standing check take CPU
    # minutes; background processes run them while training continues.
    self.exporter: ProcessPoolExecutor | None = None
    if cfg.export_workers > 0:
      self.exporter = ProcessPoolExecutor(
        cfg.export_workers, mp_context=E.CTX, initializer=E.export_worker_init, initargs=(2,)
      )
    self.exports: list[Future] = []

  def _standing_poses(self) -> list[dict]:
    """Standing-evaluation poses, chosen once from the corpus and fixed in the run dir."""
    path = self.run_dir / "standing_poses.json"
    if not path.exists():
      path.write_text(json.dumps(S.quiet_poses(self.motions), indent=2))
    return json.loads(path.read_text())

  # --- Miniverse binding --------------------------------------------------
  def _graph(self) -> E.InferenceGraph:
    t = self.trainer
    return E.InferenceGraph(
      t.ema if t.ema is not None else t.tracker,
      t.kin_torso,
      self.examples,
      self.cfg.action_clip,
      self.cfg.reference_offsets,
    )

  def _bind_miniverse(self) -> dict:
    if self.binding_path.exists():
      return json.loads(self.binding_path.read_text())
    prov = provenance(self.cfg, self.motions, self.device, self.examples_names)
    bundle_id = f"super-gpc-stage1-{self.job_id}"
    info = E.build_bundle(
      self.run_dir / "bundle" / "source",
      self._graph(),
      self.examples,
      bundle_id,
      "GPC Stage 1 motion tracker",
      {"iteration": 0, "job": self.job_id},
      {
        "stage": "super-gpc Stage 1 tracker",
        "job": self.job_id,
        "references": self.examples_names,
        "code": prov["code"],
        "dataset_revision": prov["dataset"].get("revision"),
        "robot": prov["robot"],
        "visibility": "private",
      },
    )
    binding = {
      "bundle_id": bundle_id,
      "source_bundle": info,
      "provenance": prov,
      "created": time.time(),
    }
    self.binding_path.write_text(json.dumps(binding, indent=2))
    return binding

  # --- checkpoints ------------------------------------------------------------
  def save_resumable(self) -> None:
    # Finish, rather than discard/restart, a separate-env EMA sweep. This adds
    # checkpoint-boundary time but avoids serializing partial probe simulator state.
    while getattr(self.trainer, "probe_state", None) is not None:
      self.trainer.probe_tick()
    path = self.run_dir / "resume.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(self.trainer.resumable_state(), tmp)
    tmp.replace(path)

  def policy_checkpoint(self) -> None:
    """Save the native tracker, then export, check and publish it; metrics
    are logged at the checkpoint's own step when its export finishes."""
    t, it = self.trainer, self.trainer.iteration
    native = self.run_dir / "checkpoints" / f"policy_{it:07d}.pt"
    native.parent.mkdir(exist_ok=True)
    torch.save(t.policy_state(), native)
    self._submit_export(native, (t.ema if t.ema is not None else t.tracker).state_dict(), it)

  def resubmit_unexported(self) -> None:
    """Export native checkpoints up to the resumed iteration that never reached
    the publication ledger, such as one whose background export was
    interrupted with the process. Later ones are retrained and re-saved."""
    known = {e["step"] for e in self.publisher.ledger}
    for native in sorted((self.run_dir / "checkpoints").glob("policy_*.pt")):
      state = torch.load(native, map_location="cpu", weights_only=False)
      if state["iteration"] <= self.trainer.iteration and state["iteration"] not in known:
        print(f"[export] resubmitting unpublished {native.name}", flush=True)
        self._submit_export(native, tracker_weights(state), state["iteration"])

  def _submit_export(self, native: Path, tracker_state: dict, it: int) -> None:
    args = (
      str(self.run_dir),
      str(self.motions),
      self.examples_names,
      self.standing_poses,
      self.cfg.action_clip,
      self.binding["bundle_id"],
      self.job_id,
      {k: v.detach().cpu().clone() for k, v in tracker_state.items()},
      it,
      str(native),
      E.sha256(native),
      self.cfg.reference_offsets,
    )
    if self.exporter is None:
      self._finish_export(*export_policy_checkpoint(*args))
    else:
      self.exports.append(self.exporter.submit(export_policy_checkpoint, *args))

  def _finish_export(self, record: dict, metrics: dict) -> None:
    self.publisher.submit(record)
    self.log(metrics, record["step"])

  def drain_exports(self, wait: bool = False) -> None:
    """Publish finished background exports in step order; a failed export
    stops the run visibly."""
    while self.exports and (wait or self.exports[0].done()):
      self._finish_export(*self.exports.pop(0).result())

  # --- loop ---------------------------------------------------------------------
  def log(self, metrics: dict, step: int) -> None:
    metrics = metrics | self.publisher.summary()
    nonfinite = {k: v for k, v in metrics.items() if not math.isfinite(v)}
    if nonfinite:
      raise FloatingPointError(f"nonfinite Stage 1 metrics at step {step}: {nonfinite}")
    self.telemetry.log(metrics, step)

  def evaluate(self) -> None:
    t = self.trainer
    metrics, before, after = t.evaluate()
    with (self.run_dir / "preservation.jsonl").open("a") as f:
      f.write(
        json.dumps(
          {"iteration": t.iteration, "before": before, "after": after, "preserved": before == after}
        )
        + "\n"
      )
    self.log(metrics, t.iteration)
    print(
      f"[eval] it={t.iteration} "
      + " ".join(
        f"{k}={v:.4g}"
        for k, v in sorted(metrics.items())
        if k.startswith(
          (
            "eval_val/succ",
            "eval_val/norm",
            "eval_train/succ",
            "eval/",
            "model_heldout/body_pos_err_m_8",
          )
        )
      ),
      flush=True,
    )

  def run(self, iterations: int) -> None:
    self._state_ready = False
    try:
      self._run(iterations)
    except BaseException:
      # Preserve the original failure, even if disk or telemetry cleanup fails.
      cleanups = [("telemetry close", self.telemetry.close)]
      if self._state_ready:
        cleanups.insert(0, ("native resume save", self.save_resumable))
      for label, cleanup in cleanups:
        try:
          cleanup()
        except BaseException as error:
          print(f"[shutdown] {label} failed: {error!r}", file=sys.stderr, flush=True)
      if self.exporter is not None:
        try:
          self.exporter.shutdown(wait=False, cancel_futures=True)
        except BaseException as error:
          print(f"[shutdown] exporter cleanup failed: {error!r}", file=sys.stderr, flush=True)
      raise
    else:
      self.telemetry.close()

  def _run(self, iterations: int) -> None:
    cfg, t = self.cfg, self.trainer
    resume = self.run_dir / "resume.pt"
    if resume.exists():
      t.load_resumable(torch.load(resume, map_location=self.device, weights_only=False))
      self._state_ready = True
      print(f"[resume] iteration {t.iteration}, replay {t.replay.count}", flush=True)
      self.resubmit_unexported()
    else:
      self._state_ready = True
      t0 = time.time()
      t.initial_fill()
      self.log(t.pop_stats() | {"time/initial_fill_s": time.time() - t0}, t.iteration)
      self.evaluate()
    stop = min(iterations, cfg.max_iterations)
    stop_request = self.run_dir / "STOP"
    while t.iteration < stop:
      t.train_iteration()
      it = t.iteration
      if stop_request.exists():  # graceful stop: final evaluation, checkpoint and resume state
        print(f"[stop] {stop_request} found at iteration {it}", flush=True)
        stop = it
      if it % cfg.log_interval == 0:
        stats = t.pop_stats()
        self.log(stats, it)
        print(
          f"[train] it={it} "
          + " ".join(
            f"{k}={stats[k]:.4g}"
            for k in (
              "real/reward",
              "episodes/length_s",
              "model/loss",
              "actor/imagined_return",
              "critic/loss",
              "time/iteration_s",
            )
            if k in stats
          ),
          flush=True,
        )
      if it % cfg.eval_interval == 0 or it == stop:
        self.evaluate()
      if it % cfg.checkpoint_interval == 0 or it == stop:
        self.policy_checkpoint()
      if it % cfg.resume_interval == 0 or it == stop:
        self.save_resumable()
      self.drain_exports()
    self.drain_exports(wait=True)
    if self.exporter is not None:
      self.exporter.shutdown()
    self.log(self.publisher.summary(), t.iteration)


def parse_overrides(pairs: list[str]) -> dict:
  types = {f.name: f.type for f in fields(Stage1Cfg)}
  out = {}
  for pair in pairs:
    key, value = pair.split("=", 1)
    if key not in types:
      raise KeyError(f"unknown Stage1Cfg field {key}")
    if key == "example_clips":
      out[key] = tuple(v for v in value.split(",") if v)
    else:
      out[key] = json.loads(value)
    if key == "reference_offsets":
      if not isinstance(out[key], list) or any(type(v) is not int for v in out[key]):
        raise ValueError("reference_offsets must be a JSON integer array")
      out[key] = tuple(out[key])
  return out


def hash_config(cfg: Stage1Cfg) -> str:
  return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def env_device() -> str:
  return os.environ.get("GPC_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
