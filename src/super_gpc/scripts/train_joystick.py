"""Train or resume joystick CoLA RLFT on a frozen Stage 2 prior.

Example:
  gpc-train-joystick --prior runs/stage2/checkpoints/policy_0023000.pt \\
    --run-dir runs/joystick --iterations 2500
Re-running the same command resumes ``<run-dir>/resume.pt`` and the same
Miniverse run; the stored configuration is authoritative.
View it with ``miniverse serve`` on the run directory's parent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from super_gpc.tasks.steering.rl.cola_cfg import ColaCfg
from super_gpc.tasks.steering.rl.cola_run import ColaRun
from super_gpc.tasks.tracking.rl.stage1_run import env_device


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("--prior", type=Path, required=True, help="Stage 2 policy checkpoint")
  p.add_argument("--run-dir", type=Path, required=True)
  p.add_argument("--iterations", type=int, required=True, help="stop after this iteration")
  p.add_argument("--device", default=env_device())
  p.add_argument("--set", nargs="*", default=[], help="ColaCfg overrides, key=json")
  args = p.parse_args()
  names = {f.name for f in fields(ColaCfg)}
  overrides = {}
  for pair in args.set:
    key, value = pair.split("=", 1)
    if key not in names:
      raise SystemExit(f"unknown ColaCfg field {key}")
    overrides[key] = json.loads(value)
  config_path = args.run_dir / "config.json"
  if config_path.exists():
    cfg = ColaCfg.from_dict(json.loads(config_path.read_text()))
    if any(getattr(cfg, k) != v for k, v in overrides.items()):
      raise SystemExit(f"overrides differ from stored config {config_path}")
  else:
    cfg = ColaCfg(**overrides)
    cfg.validate()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg.to_dict(), indent=2))
  run = ColaRun(cfg, args.prior, args.run_dir, args.device, args.run_dir.name)
  print(
    json.dumps({"miniverse_run": str(run.telemetry.run.dir), "bundle": run.binding["bundle_id"]}),
    flush=True,
  )
  run.run(args.iterations)


if __name__ == "__main__":
  main()
