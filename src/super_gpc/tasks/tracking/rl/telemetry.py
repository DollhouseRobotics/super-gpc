"""Miniverse training-run files: metrics and published checkpoints.

Every run directory is also a Miniverse run directory (``miniverse agent-help
run-directory``, miniverse-sdk 0.9). ``miniverse.init`` keeps ``run.json``,
``base.mini`` (a copy of the run's source bundle, ``bundle/source.mini``) and
one Rerun metric file per process start under ``metrics/``. A published
checkpoint is ``checkpoints/step-<N>/``, beside the native ``policy_*.pt``
files, holding the exported ONNX model that replaces the base's model.
``miniverse serve`` on the parent of the run directories shows their metrics
and runs any checkpoint in the browser; ``miniverse push RUN_DIR [--watch]``
mirrors a run to miniverse.bot. Metrics are also appended to ``metrics.jsonl``
and every publication is recorded in the ledger ``checkpoints.json``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import miniverse


class Telemetry:
  """Open or resume the Miniverse run in ``run_dir``, based on its source bundle."""

  def __init__(self, run_dir: Path, name: str):
    # Only the process that owns the run constructs this (rank 0 under torchrun).
    base = Path(run_dir) / "bundle" / "source.mini"
    self.run = miniverse.init(run_dir, base, name=name, rank=0)
    self.lock = threading.Lock()
    self.local = (Path(run_dir) / "metrics.jsonl").open("a")

  def log(self, metrics: dict[str, float], step: int) -> None:
    clean = {k: float(v) for k, v in metrics.items() if v == v and abs(v) != float("inf")}
    with self.lock:
      self.local.write(json.dumps({"step": step, "time": time.time(), "metrics": clean}) + "\n")
      self.local.flush()
      self.run.log(clean, step=step)

  def close(self) -> None:
    with self.lock:
      self.run.finish()
      self.local.close()


class CheckpointPublisher:
  """Publish exported checkpoints into the Miniverse run, with a durable ledger.

  Publication is a local copy, so it runs inline. A resumed run publishes any
  ledger entry not yet published.
  """

  def __init__(self, run_dir: Path, run: miniverse.Run):
    self.path = Path(run_dir) / "checkpoints.json"
    self.ledger: list[dict] = json.loads(self.path.read_text()) if self.path.exists() else []
    self.run = run
    for entry in self.ledger:
      if entry["status"] != "published":
        self._publish(entry)
    self._save()

  def _save(self) -> None:
    tmp = self.path.with_suffix(".tmp")
    tmp.write_text(json.dumps(self.ledger, indent=2))
    tmp.replace(self.path)

  def _publish(self, entry: dict) -> None:
    models = entry.get("models") or {"tracker": entry["onnx"]}
    step = entry["step"]
    try:
      directory = self.run.checkpoint(step, models, label=entry.get("label", f"iter-{step}"))
    except FileExistsError:  # published before the ledger was saved
      directory = self.run.dir / "checkpoints" / f"step-{step}"
    except (OSError, ValueError) as error:  # kept in the ledger; retried on resume
      entry["status"], entry["error"] = "failed", repr(error)
      print(f"[publish] step {step} failed: {error!r}", flush=True)
      return
    entry.pop("error", None)
    entry |= {"status": "published", "path": str(directory), "published_time": time.time()}
    print(f"[publish] step {step}: {directory}", flush=True)

  def submit(self, entry: dict) -> None:
    entry = dict(entry, status="pending")
    self.ledger.append(entry)
    self._publish(entry)
    self._save()

  def summary(self) -> dict[str, float]:
    published = sum(e["status"] == "published" for e in self.ledger)
    return {
      "publish/checkpoints_published": float(published),
      "publish/checkpoints_failed": float(len(self.ledger) - published),
      "publish/checkpoints_total": float(len(self.ledger)),
    }
