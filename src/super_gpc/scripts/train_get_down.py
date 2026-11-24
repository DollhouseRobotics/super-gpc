"""Train or resume get-down CoLA RLFT (AMP) on a frozen Stage 2 prior.

Example:
  gpc-train-get-down --prior runs/stage2-w1024/checkpoints/policy_0023000.pt \\
    --motions data/motions --starts data/joystick_starts.pt \\
    --run-dir runs/get-down --iterations 2000
``--motions`` supplies the start states and the AMP reference clips (it must
contain bones-floor); ``--starts`` is the joystick-rollout start bank from
gpc-collect-joystick-starts. Re-running the same command resumes
``<run-dir>/resume.pt`` and the same Miniverse run; the stored configuration
is authoritative.
View it with ``miniverse serve`` on the run directory's parent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from super_gpc.tasks.get_down.rl.get_down_cfg import GetDownCfg
from super_gpc.tasks.get_down.rl.get_down_run import GetDownRun
from super_gpc.tasks.tracking.rl.stage1_run import env_device


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("--prior", type=Path, required=True, help="Stage 2 policy checkpoint")
  p.add_argument("--motions", type=Path, required=True, help="prepared motion archive")
  p.add_argument("--starts", type=Path, required=True, help="joystick-rollout start bank")
  p.add_argument("--run-dir", type=Path, required=True)
  p.add_argument("--iterations", type=int, required=True, help="stop after this iteration")
  p.add_argument("--device", default=env_device())
  p.add_argument("--set", nargs="*", default=[], help="GetDownCfg overrides, key=json")
  args = p.parse_args()
  names = {f.name for f in fields(GetDownCfg)}
  overrides = {}
  for pair in args.set:
    key, value = pair.split("=", 1)
    if key not in names:
      raise SystemExit(f"unknown GetDownCfg field {key}")
    overrides[key] = json.loads(value)
  config_path = args.run_dir / "config.json"
  if config_path.exists():
    cfg = GetDownCfg.from_dict(json.loads(config_path.read_text()))
    if any(getattr(cfg, k) != v for k, v in overrides.items()):
      raise SystemExit(f"overrides differ from stored config {config_path}")
  else:
    cfg = GetDownCfg(**overrides)
    cfg.validate()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg.to_dict(), indent=2))
  run = GetDownRun(
    cfg, args.prior, args.motions, args.starts, args.run_dir, args.device, args.run_dir.name
  )
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
