"""Stage 2 run lifecycle: provenance, Miniverse run, loop, checkpoints, resume.

The run directory is the durable identity of one prior-training job, laid out
like Stage 1's: ``miniverse.json`` (run binding, teacher identity),
Miniverse run files, ``metrics.jsonl``, ``preservation.jsonl``,
``resume.pt`` (resumable state, every ``latest_interval`` iterations),
``checkpoints/latest.pt`` (latest inference state, same cadence), and
``checkpoints/policy_*.pt`` (numbered retention). Every saved inference step
has its own ``exports/iter_*`` bundle and ``checkpoints.json`` publication
entry, including latest-only saves. Restarting with the same directory and
teacher resumes the same Miniverse run; a different teacher is refused.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from super_gpc.rl import prior as P
from super_gpc.rl.networks import Tracker
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.mdp.motion_library import MotionLibrary
from super_gpc.tasks.tracking.rl import export as E
from super_gpc.tasks.tracking.rl import stage2_export as X
from super_gpc.tasks.tracking.rl.stage1_run import provenance
from super_gpc.tasks.tracking.rl.stage2 import Stage2Trainer, dist_world
from super_gpc.tasks.tracking.rl.stage2_cfg import Stage2Cfg
from super_gpc.tasks.tracking.rl.telemetry import CheckpointPublisher, Telemetry

TEACHER_STATUSES = ("smoke", "stage1")
NAME = "GPC motion prior"


def check_teacher_status(status: str, info: dict) -> None:
  """The launcher declares what the teacher is; refuse inconsistent labels so a
  smoke tracker is never published as a Stage 1 teacher."""
  if status not in TEACHER_STATUSES:
    raise ValueError(f"teacher status must be one of {TEACHER_STATUSES}")
  stage = info["stage"]
  if ("smoke" in stage or "smoke" in info) != (status == "smoke"):
    raise ValueError(f"teacher file stage {stage!r} does not match status {status!r}")
  if status == "stage1" and stage != "1":
    raise ValueError(f"status 'stage1' needs a Stage 1 checkpoint, got stage {stage!r}")


def export_checkpoint(
  run_dir: str,
  motions: str,
  start_names: list[str],
  cfg_dict: dict,
  bundle_id: str,
  job_id: str,
  teacher_status: str,
  state: dict,
  native: str,
  native_sha256: str,
) -> dict:
  """Export, parity-check and closed-loop-check one saved policy state.

  The result is the publication ledger record.
  """
  cfg = Stage2Cfg.from_dict(cfg_dict)
  run_dir_p = Path(run_dir)
  prior = P.CodePrior(
    O.OBS_DIM,
    cfg.d_model,
    cfg.num_layers,
    cfg.num_heads,
    cfg.dropout,
    cfg.norm_ema_decay,
    cfg.norm_clamp,
  )
  prior.load_state_dict(state["prior"])
  prior.eval()
  tracker = Tracker(O.ENCODER_DIM, O.HISTORY_DIM, O.ACTION_DIM)
  tracker.load_state_dict(state["tracker"])
  tracker.eval().requires_grad_(False)
  starts = MotionLibrary(Path(motions), "cpu", clip_names=start_names)
  graph = X.PriorPolicyGraph(prior, tracker, cfg.top_p, cfg.temperature, cfg.action_clip)
  it = state["iteration"]
  out = run_dir_p / "exports" / f"iter_{it:07d}"
  info = X.build_bundle(
    out,
    graph,
    starts,
    cfg.demo_seeds,
    bundle_id,
    NAME,
    {"iteration": it, "job": job_id, "native_sha256": native_sha256},
    {"iteration": it},
    embodiment_from=run_dir_p / "bundle" / "source" / "embodiment",
  )
  onnx = Path(info["onnx"])
  parity = X.parity(graph, onnx, X.parity_samples(starts, it))
  loop = X.closed_loop_check(out, starts, prior, tracker, cfg)
  record = {
    "step": it,
    "label": f"iter-{it}",
    "models": {"prior": str(onnx)},
    "onnx": str(onnx),
    "onnx_sha256": info["onnx_sha256"],
    "native": native,
    "native_sha256": native_sha256,
    "teacher": state["teacher"] | {"status": teacher_status},
    "parity": parity,
    "latency": X.latency(onnx),
    "closed_loop": loop,
  }
  (out.parent / f"iter_{it:07d}.json").write_text(json.dumps(record, indent=2))
  return record


def export_metrics(record: dict) -> dict:
  loop = record["closed_loop"]
  return (
    record["parity"]
    | record["latency"]
    | {
      "export/closed_loop_survival_rate": loop["survival_rate"],
      "export/closed_loop_mean_survival_s": loop["mean_survival_s"],
      "export/closed_loop_token_mismatch_steps": float(loop["token_mismatch_steps"]),
      "export/closed_loop_max_action_diff": loop["max_abs_action_diff"],
    }
  )


def merge_rank_stats(per_rank: list[dict[str, list[float]]]) -> dict[str, list[float]]:
  out: dict[str, list[float]] = {}
  for stats in per_rank:
    for k, v in stats.items():
      out.setdefault(k, []).extend(v)
  return out


class Stage2Run:
  """One prior-training job. Under ``torchrun`` every rank constructs a run;
  rank 0 alone owns the Miniverse run, telemetry, evaluation, checkpoints and
  publication, and the other ranks only collect and optimize."""

  def __init__(
    self,
    cfg: Stage2Cfg,
    motions: Path,
    teacher: Path,
    teacher_status: str,
    run_dir: Path,
    device: str,
    name: str,
  ):
    self.cfg, self.motions, self.run_dir, self.device = cfg, Path(motions), Path(run_dir), device
    self.teacher_path = Path(teacher)
    self.job_id = name
    self.rank, self.world_size = dist_world()
    self.main = self.rank == 0
    self.trainer = Stage2Trainer(cfg, self.motions, self.teacher_path, device, evaluate=self.main)
    check_teacher_status(teacher_status, self.trainer.teacher_info)
    self.teacher_status = teacher_status
    self.start_names = list(cfg.example_clips) or E.select_examples(self.motions)
    if not self.main:
      return
    self.run_dir.mkdir(parents=True, exist_ok=True)
    self.starts = MotionLibrary(self.motions, "cpu", clip_names=self.start_names)
    self.binding_path = self.run_dir / "miniverse.json"
    self.binding = self._bind_miniverse()
    self.telemetry = Telemetry(self.run_dir, f"super-gpc stage2 {self.job_id}")
    self.publisher = CheckpointPublisher(self.run_dir, self.telemetry.run)

  # --- Miniverse binding --------------------------------------------------
  def _graph(self) -> X.PriorPolicyGraph:
    t, c = self.trainer, self.cfg
    return X.PriorPolicyGraph(t.prior, t.teacher, c.top_p, c.temperature, c.action_clip)

  def _metadata(self, prov: dict) -> dict:
    return {
      "stage": "super-gpc Stage 2 generative prior",
      "job": self.job_id,
      "width": {"d_model": self.cfg.d_model, "ff": self.cfg.ff_size},
      "sampling": {"top_p": self.cfg.top_p, "temperature": self.cfg.temperature},
      "teacher": self.trainer.teacher_info | {"status": self.teacher_status},
      "start_poses": self.start_names,
      "code": prov["code"],
      "dataset_revision": prov["dataset"].get("revision"),
      "robot": prov["robot"],
      "visibility": "private",
    }

  def _bind_miniverse(self) -> dict:
    teacher = self.trainer.teacher_info
    if self.binding_path.exists():
      binding = json.loads(self.binding_path.read_text())
      if binding["teacher"]["sha256"] != teacher["sha256"]:
        raise ValueError("this run directory was created with a different teacher checkpoint")
      return binding
    prov = provenance(self.cfg, self.motions, self.device, self.start_names)
    prov |= {"stage": "2", "teacher": teacher | {"status": self.teacher_status}}
    bundle_id = f"super-gpc-stage2-{self.job_id}"
    info = X.build_bundle(
      self.run_dir / "bundle" / "source",
      self._graph(),
      self.starts,
      self.cfg.demo_seeds,
      bundle_id,
      NAME,
      {"iteration": 0, "job": self.job_id},
      self._metadata(prov),
    )
    binding = {
      "bundle_id": bundle_id,
      "source_bundle": info,
      "teacher": teacher | {"status": self.teacher_status},
      "provenance": prov,
      "created": time.time(),
    }
    self.binding_path.write_text(json.dumps(binding, indent=2))
    return binding

  # --- checkpoints ------------------------------------------------------------
  def _save(self, state: dict, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)

  def save_latest(self) -> None:
    """Resumable state plus the latest inference state (GPC-code 'last' cadence).
    Collective: every rank contributes its trainer RNG and collector state."""
    t = self.trainer
    rng, collector = t.rng_state(), t.collector.state_dict()
    if self.world_size > 1:
      gathered: list = [None] * self.world_size
      dist.all_gather_object(gathered, (rng, collector))
      rngs, collectors = [g[0] for g in gathered], [g[1] for g in gathered]
    else:
      rngs, collectors = [rng], [collector]
    if self.main:
      self._save(t.resumable_state(rngs, collectors), self.run_dir / "resume.pt")
      self._save(t.policy_state(), self.run_dir / "checkpoints" / "latest.pt")

  def policy_checkpoint(self, *, numbered: bool = True) -> None:
    """Save the native inference state, then export, check and publish it."""
    t, it = self.trainer, self.trainer.iteration
    filename = f"policy_{it:07d}.pt" if numbered else "latest.pt"
    native = self.run_dir / "checkpoints" / filename
    state = t.policy_state()
    state = {
      k: ({n: v.detach().cpu().clone() for n, v in v.items()} if k in ("prior", "tracker") else v)
      for k, v in state.items()
    }
    self._save(state, native)
    record = export_checkpoint(
      str(self.run_dir),
      str(self.motions),
      self.start_names,
      self.cfg.to_dict(),
      self.binding["bundle_id"],
      self.job_id,
      self.teacher_status,
      state,
      str(native),
      E.sha256(native),
    )
    self.publisher.submit(record)
    self.log(export_metrics(record), it)

  # --- loop ---------------------------------------------------------------------
  def log(self, metrics: dict, step: int) -> None:
    self.telemetry.log(metrics | self.publisher.summary(), step)

  def evaluate(self) -> None:
    t, q = self.trainer, self.cfg.quality_interval
    metrics, before, after = t.evaluate(q > 0 and t.iteration % q == 0)
    with (self.run_dir / "preservation.jsonl").open("a") as f:
      record = {"iteration": t.iteration, "before": before, "after": after}
      f.write(json.dumps(record | {"preserved": before == after}) + "\n")
    self.log(metrics, t.iteration)
    shown = (
      "gen_train/surv",
      "gen_train/code_match",
      "quality/free/upright/full/fall_stuck",
      "quality/push_9.8mps/upright/full/recovered",
      "quality/free/upright/10s/apd_root",
      "eval/",
    )
    print(
      f"[eval] it={t.iteration} "
      + " ".join(f"{k}={v:.4g}" for k, v in sorted(metrics.items()) if k.startswith(shown)),
      flush=True,
    )

  def _gather_stats(self) -> dict[str, float]:
    t = self.trainer
    if self.world_size > 1:
      gathered: list = [None] * self.world_size
      dist.all_gather_object(gathered, t.stats)
      t.stats = merge_rank_stats(gathered)
    out = t.pop_stats()
    out["progress/transitions"] = float(t.transitions * self.world_size)
    out["progress/ranks"] = float(self.world_size)
    return out

  def run(self, iterations: int) -> None:
    cfg, t = self.cfg, self.trainer
    resume = self.run_dir / "resume.pt"
    if resume.exists():
      t.load_resumable(torch.load(resume, map_location="cpu", weights_only=False))
      print(f"[resume] iteration {t.iteration}, transitions {t.transitions}", flush=True)
    else:
      t.start()
      if self.main:
        self.evaluate()  # untrained-prior baseline at iteration 0
    per_iteration = cfg.num_envs * cfg.rollout_steps * self.world_size
    stop = min(iterations, cfg.max_transitions // per_iteration)
    while t.iteration < stop:
      t.train_iteration()
      it = t.iteration
      if it % cfg.log_interval == 0:
        stats = self._gather_stats()
        if self.main:
          self.log(stats, it)
          keys = ("prior/loss", "prior/accuracy", "episodes/length_s", "time/iteration_s")
          print(
            f"[train] it={it} " + " ".join(f"{k}={stats[k]:.4g}" for k in keys if k in stats),
            flush=True,
          )
      numbered = it % cfg.checkpoint_interval == 0 or it == stop
      latest = it % cfg.latest_interval == 0 or it == stop
      if self.main:
        q = cfg.quality_interval
        if it % cfg.eval_interval == 0 or it == stop or (q > 0 and it % q == 0):
          self.evaluate()
        if numbered or latest:
          self.policy_checkpoint(numbered=numbered)
      if latest:
        self.save_latest()
    if self.main:
      self.log(self.publisher.summary(), t.iteration)
      self.telemetry.close()
