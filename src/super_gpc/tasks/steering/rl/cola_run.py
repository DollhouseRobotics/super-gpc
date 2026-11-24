"""Joystick CoLA run lifecycle: Miniverse run, loop, checkpoints and resume.

Layout like Stage 2: ``config.json``, ``miniverse.json`` (run binding and the
frozen prior's identity), Miniverse run files, ``metrics.jsonl``,
``preservation.jsonl``, ``resume.pt``, ``checkpoints/latest.pt`` and numbered
``checkpoints/policy_*.pt``, ``exports/iter_*`` with one ledger entry in
``checkpoints.json`` per published step. Re-running with the same directory
resumes the same Miniverse run; a different prior is refused.

The run directory is also the Miniverse run directory (see ``telemetry``):
every published step is a checkpoint of the source bundle in
``checkpoints/step-<N>/``.
"""

from __future__ import annotations

import gc
import json
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import torch

from super_gpc.rl import prior as P
from super_gpc.rl.cola import ColaPolicy
from super_gpc.rl.networks import Tracker
from super_gpc.tasks.steering.rl import cola_export as X
from super_gpc.tasks.steering.rl.cola import ColaTrainer
from super_gpc.tasks.steering.rl.cola_cfg import ColaCfg
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.rl import export as E
from super_gpc.tasks.tracking.rl.stage1_run import provenance
from super_gpc.tasks.tracking.rl.stage2_cfg import Stage2Cfg
from super_gpc.tasks.tracking.rl.telemetry import CheckpointPublisher, Telemetry


def _policy_from_state(state: dict) -> ColaPolicy:
  cfg = ColaCfg.from_dict(state["cfg"])
  d = state["prior_info"]["d_model"]
  s2 = Stage2Cfg(d_model=d)
  prior = P.CodePrior(O.OBS_DIM, d, s2.num_layers, s2.num_heads, s2.dropout)
  prior.load_state_dict(state["prior"])
  tracker = Tracker(O.ENCODER_DIM, O.HISTORY_DIM, O.ACTION_DIM)
  tracker.load_state_dict(state["tracker"])
  policy = ColaPolicy(prior, tracker, cfg.action_clip, cfg.rank, cfg.alpha, cfg.command_dim)
  policy.adapters.load_state_dict(state["adapters"])
  return policy.eval()


def export_checkpoint(run_dir: str, bundle_id: str, state: dict, native: str, sha: str) -> dict:
  """Export, parity-check and closed-loop-check one saved policy state."""
  cfg = ColaCfg.from_dict(state["cfg"])
  policy = _policy_from_state(state)
  graph = X.ColaPolicyGraph(policy, cfg)
  it = state["iteration"]
  out = Path(run_dir) / "exports" / f"iter_{it:07d}"
  info = X.build_bundle(
    out,
    graph,
    bundle_id,
    {"iteration": it, "native_sha256": sha},
    {"iteration": it},
    cfg.speed_max,
    embodiment_from=Path(run_dir) / "bundle" / "source" / "embodiment",
    gaze=X.gaze_constants(cfg),
  )
  onnx = Path(info["onnx"])
  record = {
    "step": it,
    "label": f"iter-{it}",
    "models": {"policy": str(onnx)},
    "onnx": str(onnx),
    "onnx_sha256": info["onnx_sha256"],
    "native": native,
    "native_sha256": sha,
    "prior": state["prior_info"],
    "parity": X.parity(graph, onnx, X.parity_samples(it)),
    "latency": X.latency(onnx),
    "closed_loop": X.closed_loop_check(out, policy, cfg.fallen_height),
  }
  (out.parent / f"iter_{it:07d}.json").write_text(json.dumps(record, indent=2))
  # Free ONNX Runtime sessions held in reference cycles: the long-lived export
  # worker rarely runs a full collection, so their threads would accumulate.
  gc.collect()
  return record


def export_metrics(record: dict) -> dict:
  loop = record["closed_loop"]
  out = record["parity"] | record["latency"]
  out |= {
    "export/closed_loop_fell": float(loop["fell_at_step"] >= 0),
    "export/closed_loop_token_mismatch_steps": float(loop["token_mismatch_steps"]),
    "export/closed_loop_max_action_diff": loop["max_abs_action_diff"],
  }
  for i, seg in enumerate(loop["segments"]):
    out[f"export/closed_loop_seg{i}_forward_mps"] = seg["mean_body_velocity_second_half"][0]
    out[f"export/closed_loop_seg{i}_left_mps"] = seg["mean_body_velocity_second_half"][1]
    if "look_yaw_range_rad" in seg:
      out[f"export/closed_loop_seg{i}_look_yaw_range_rad"] = seg["look_yaw_range_rad"]
  return out


class ColaRun:
  """Joystick steering run. Other CoLA tasks subclass it and replace the
  trainer, source bundle, export function, naming and printed keys."""

  stage = "rlft-cola"
  title = "super-gpc joystick CoLA RLFT"
  run_prefix = "rlft"
  motions: Path | None = None  # the joystick task reads no motion archive
  export_fn = staticmethod(export_checkpoint)
  eval_keys = (
    "reward/task",
    "success_rate",
    "survival_rate",
    "task/velocity_error_mps",
    "task/facing_error_rad",
    "idle/gaze_error_rad",
    "idle/head_yaw_from_pelvis_rad",
    "idle/offcenter_look_error_rad",
    "idle/arm_offset_rad",
    "idle/pelvis_speed_mps",
    "idle/ankle_speed_mps",
    "floor/getup_rate",
    "floor/getup_s",
    "policy/kl_base_sum",
    "training_state_preserved",
  )
  train_keys = (
    "reward/task",
    "task/velocity_error_mps",
    "task/facing_error_rad",
    "idle/gaze_error_rad",
    "episodes/length_s",
    "policy/kl_base_sum",
    "actor/updates",
    "time/iteration_s",
  )

  def make_trainer(self) -> ColaTrainer:
    return ColaTrainer(self.cfg, self.prior_path, self.device)

  def source_bundle(self, out: Path, bundle_id: str, provenance: dict, metadata: dict) -> dict:
    graph = X.ColaPolicyGraph(self.trainer.policy, self.cfg)
    gaze = X.gaze_constants(self.cfg)
    return X.build_bundle(
      out, graph, bundle_id, provenance, metadata, self.cfg.speed_max, gaze=gaze
    )

  def export_metrics(self, record: dict) -> dict:
    return export_metrics(record)

  def extra_provenance(self) -> dict:
    return {}

  def __init__(
    self,
    cfg: ColaCfg,
    prior: Path,
    run_dir: Path,
    device: str,
    name: str,
  ):
    self.cfg, self.run_dir, self.device, self.job_id = cfg, Path(run_dir), device, name
    self.prior_path = Path(prior)
    self.binding_path = self.run_dir / "miniverse.json"
    self.trainer = self.make_trainer()
    self.run_dir.mkdir(parents=True, exist_ok=True)
    self.binding = self._bind_miniverse()
    self.telemetry = Telemetry(self.run_dir, f"super-gpc {self.run_prefix} {self.job_id}")
    self.publisher = CheckpointPublisher(self.run_dir, self.telemetry.run)
    self.exporter: ProcessPoolExecutor | None = None
    self.exports: list[tuple[int, Future]] = []
    self.export_failures = 0

  def _bind_miniverse(self) -> dict:
    info = self.trainer.prior_info
    if self.binding_path.exists():
      binding = json.loads(self.binding_path.read_text())
      if binding["prior"]["sha256"] != info["sha256"]:
        raise ValueError("this run directory was created with a different Stage 2 prior")
      return binding
    prov = provenance(self.cfg, self.motions, self.device, [])
    prov |= {"stage": self.stage, "prior": info} | self.extra_provenance()
    bundle_id = f"super-gpc-{self.run_prefix}-{self.job_id}"
    meta = {
      "stage": self.title,
      "job": self.job_id,
      "prior": info,
      "code": prov["code"],
      "robot": prov["robot"],
      "visibility": "private",
    }
    built = self.source_bundle(
      self.run_dir / "bundle" / "source", bundle_id, {"iteration": 0, "job": self.job_id}, meta
    )
    binding = {
      "bundle_id": bundle_id,
      "source_bundle": built,
      "prior": info,
      "provenance": prov,
      "created": time.time(),
    }
    self.binding_path.write_text(json.dumps(binding, indent=2))
    return binding

  def _save(self, state: dict, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)

  def checkpoint(self, numbered: bool) -> None:
    t, it = self.trainer, self.trainer.iteration
    self._save(t.resumable_state(), self.run_dir / "resume.pt")
    state = t.policy_state()
    native = self.run_dir / "checkpoints" / (f"policy_{it:07d}.pt" if numbered else "latest.pt")
    self._save(state, native)
    args = (str(self.run_dir), self.binding["bundle_id"], state, str(native), E.sha256(native))
    if self.exporter is None:
      record = self.export_fn(*args)
      self.publisher.submit(record)
      self.log(self.export_metrics(record), it)
    else:
      self.exports.append((it, self.exporter.submit(self.export_fn, *args)))

  def drain_exports(self, wait: bool = False) -> None:
    while self.exports and (wait or self.exports[0][1].done()):
      step, future = self.exports.pop(0)
      try:
        record = future.result()
      except Exception as error:  # native checkpoint kept on disk; training continues
        self.export_failures += 1
        print(f"[export] step {step} failed:\n{traceback.format_exc()}", flush=True)
        if isinstance(error, BrokenProcessPool) and not wait:
          self.exporter.shutdown(wait=False)
          self.exporter = self._start_exporter()
        continue
      self.publisher.submit(record)
      self.log(self.export_metrics(record), record["step"])

  def _start_exporter(self) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
      self.cfg.export_workers, mp_context=E.CTX, initializer=E.export_worker_init, initargs=(2,)
    )

  def log(self, metrics: dict, step: int) -> None:
    metrics = metrics | self.publisher.summary()
    metrics |= {
      "export/backlog": float(len(self.exports)),
      "export/failures": float(self.export_failures),
    }
    self.telemetry.log(metrics, step)

  def evaluate(self, base: bool = False) -> None:
    t = self.trainer
    metrics, before, after = t.evaluate(base=base)
    with (self.run_dir / "preservation.jsonl").open("a") as f:
      rec = {"iteration": t.iteration, "base": base, "before": before, "after": after}
      f.write(json.dumps(rec | {"preserved": before == after}) + "\n")
    self.log(metrics, t.iteration)
    p = "eval_base" if base else "eval"
    shown = " ".join(
      f"{k}={metrics[f'{p}/{k}']:.4g}" for k in self.eval_keys if f"{p}/{k}" in metrics
    )
    print(f"[{p}] it={t.iteration} {shown}", flush=True)

  def run(self, iterations: int) -> None:
    cfg, t = self.cfg, self.trainer
    resume = self.run_dir / "resume.pt"
    if resume.exists():
      t.load_resumable(torch.load(resume, map_location="cpu", weights_only=False))
      print(f"[resume] iteration {t.iteration}", flush=True)
    else:
      t.start()
      self.evaluate(base=True)  # frozen-prior baseline, same seeds
    if cfg.export_workers > 0:
      self.exporter = self._start_exporter()
    stop = min(iterations, cfg.target_transitions // (cfg.num_envs * cfg.rollout_steps))
    try:
      while t.iteration < stop:
        t.train_iteration()
        it = t.iteration
        stats = t.pop_stats()
        self.log(stats, it)
        shown = " ".join(f"{k}={stats[k]:.4g}" for k in self.train_keys if k in stats)
        print(f"[train] it={it} {shown}", flush=True)
        if it % cfg.eval_interval == 0 or it == stop:
          self.evaluate()
        if it % cfg.checkpoint_interval == 0 or it == stop:
          self.checkpoint(numbered=it % cfg.numbered_interval == 0 or it == stop)
        self.drain_exports()
    finally:
      self.drain_exports(wait=True)
      if self.exporter is not None:
        self.exporter.shutdown()
      self.log(self.publisher.summary(), t.iteration)
      self.telemetry.close()
