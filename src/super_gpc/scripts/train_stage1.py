"""Train or resume a Stage 1 tracker run.

Example:
  gpc-train-stage1 --motions data/motions --run-dir runs/stage1 --iterations 20000 \\
    --set seed=1
Re-running the same command resumes from ``<run-dir>/resume.pt`` and the same
Miniverse run. View it with ``miniverse serve`` on the run directory's parent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_gpc.tasks.tracking.rl.stage1_cfg import Stage1Cfg
from super_gpc.tasks.tracking.rl.stage1_run import Stage1Run, env_device, parse_overrides


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("--motions", type=Path, required=True, help="prepared motion directory")
  p.add_argument("--run-dir", type=Path, required=True)
  p.add_argument("--iterations", type=int, required=True, help="stop after this iteration")
  p.add_argument("--device", default=env_device())
  p.add_argument("--set", nargs="*", default=[], help="Stage1Cfg overrides, key=json")
  args = p.parse_args()
  overrides = parse_overrides(args.set)
  config_path = args.run_dir / "config.json"
  if config_path.exists():  # resume: the stored configuration is authoritative
    stored = json.loads(config_path.read_text())
    for key in ("actor_betas", "example_clips", "reference_offsets"):
      if key in stored:
        stored[key] = tuple(stored[key])
      if key in overrides:
        overrides[key] = tuple(overrides[key])
    cfg = Stage1Cfg(**stored)
    if any(getattr(cfg, k) != v for k, v in overrides.items()):
      raise SystemExit(f"overrides differ from stored config {config_path}")
  else:
    cfg = Stage1Cfg(**overrides)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg.to_dict(), indent=2))
  run = Stage1Run(cfg, args.motions, args.run_dir, args.device, args.run_dir.name)
  print(
    json.dumps(
      {
        "miniverse_run": str(run.telemetry.run.dir),
        "bundle": run.binding["bundle_id"],
      }
    ),
    flush=True,
  )
  run.run(args.iterations)


if __name__ == "__main__":
  main()
