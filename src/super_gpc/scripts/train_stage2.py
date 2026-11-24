"""Train or resume a Stage 2 generative-prior run.

Collects fresh real-simulator expert rollouts with the frozen teacher tracker
and trains the observation-conditioned autoregressive prior over its grouped
FSQ codes at the accepted width 1024. ``--iterations`` is a finite run budget
(the 10-trillion transition GPC launcher ceiling is never used as a budget).

Example:
  gpc-train-stage2 --motions data/motions \\
    --teacher runs/stage1/checkpoints/policy_0020000.pt --teacher-status stage1 \\
    --run-dir runs/stage2 --iterations 2000
Re-running the same command resumes ``<run-dir>/resume.pt`` and the same
Miniverse run; the stored configuration is authoritative.
View it with ``miniverse serve`` on the run directory's parent.

Data-parallel training on several GPUs (one rank per GPU, 1024 environments
and 1024-example minibatches per rank, averaged gradients):
  torchrun --standalone --nproc-per-node 4 -m super_gpc.scripts.train_stage2 ...
Do not run it under CUDA MPS, which hit a fatal illegal memory access.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

import torch
import torch.distributed as dist

from super_gpc.tasks.tracking.rl.stage1_run import env_device
from super_gpc.tasks.tracking.rl.stage2_cfg import Stage2Cfg
from super_gpc.tasks.tracking.rl.stage2_run import TEACHER_STATUSES, Stage2Run


def parse_overrides(pairs: list[str]) -> dict:
  names = {f.name for f in fields(Stage2Cfg)}
  out = {}
  for pair in pairs:
    key, value = pair.split("=", 1)
    if key not in names:
      raise KeyError(f"unknown Stage2Cfg field {key}")
    out[key] = tuple(value.split(",")) if key == "example_clips" else json.loads(value)
  return out


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("--motions", type=Path, required=True, help="prepared motion directory")
  p.add_argument("--teacher", type=Path, required=True, help="Stage 1 policy checkpoint")
  p.add_argument(
    "--teacher-status",
    required=True,
    choices=TEACHER_STATUSES,
    help="what the launcher asserts the teacher is; checked against the file",
  )
  p.add_argument("--run-dir", type=Path, required=True)
  p.add_argument("--iterations", type=int, required=True, help="stop after this iteration")
  p.add_argument("--device", default=env_device())
  p.add_argument("--set", nargs="*", default=[], help="Stage2Cfg overrides, key=json")
  args = p.parse_args()
  world = int(os.environ.get("WORLD_SIZE", "1"))
  rank = int(os.environ.get("RANK", "0"))
  if world > 1:
    local = int(os.environ["LOCAL_RANK"])
    args.device = f"cuda:{local}"
    torch.cuda.set_device(args.device)
    dist.init_process_group("nccl", device_id=torch.device(args.device))
  overrides = parse_overrides(args.set)
  config_path = args.run_dir / "config.json"
  if config_path.exists():  # resume: the stored configuration is authoritative
    cfg = Stage2Cfg.from_dict(json.loads(config_path.read_text()))
    stored = cfg.to_dict()
    if any(stored.get(k) != v for k, v in overrides.items() if k != "example_clips"):
      raise SystemExit(f"overrides differ from stored config {config_path}")
  else:
    cfg = Stage2Cfg(**overrides)
    cfg.validate()
    if world > 1:
      dist.barrier()  # every rank has read that no stored config exists
    if rank == 0:
      args.run_dir.mkdir(parents=True, exist_ok=True)
      config_path.write_text(json.dumps(cfg.to_dict(), indent=2))
  run = Stage2Run(
    cfg,
    args.motions,
    args.teacher,
    args.teacher_status,
    args.run_dir,
    args.device,
    args.run_dir.name,
  )
  if rank == 0:
    print(
      json.dumps(
        {
          "miniverse_run": str(run.telemetry.run.dir),
          "bundle": run.binding["bundle_id"],
          "teacher": run.binding["teacher"],
          "ranks": world,
        }
      ),
      flush=True,
    )
  run.run(args.iterations)
  if world > 1:
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
