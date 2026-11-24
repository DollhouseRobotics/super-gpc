"""Score Stage 2 prior checkpoints by unconditional sampling (see
``stage2_quality``).

Each checkpoint (a ``policy_*.pt`` of the run) gets the same starts, seeds and
push directions. The JSON report holds per-condition metrics per checkpoint
and seed, plus the provenance needed to reproduce it.

Example:
  gpc-evaluate-stage2-quality --motions data/motions --run-dir runs/stage2-w1024 \\
    --teacher runs/stage1/checkpoints/policy_0003000.pt \\
    --checkpoints runs/stage2-w1024/checkpoints/policy_0025000.pt --report quality.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from super_gpc.tasks.tracking.rl import export as E
from super_gpc.tasks.tracking.rl import stage2_quality as Q
from super_gpc.tasks.tracking.rl.stage1_run import env_device
from super_gpc.tasks.tracking.rl.stage2 import Stage2Trainer
from super_gpc.tasks.tracking.rl.stage2_cfg import Stage2Cfg


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("--motions", type=Path, required=True)
  p.add_argument("--run-dir", type=Path, required=True, help="for the run's config.json")
  p.add_argument("--teacher", type=Path, required=True)
  p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
  p.add_argument("--report", type=Path, required=True)
  p.add_argument("--starts", type=int, default=32)
  p.add_argument("--samples", type=int, default=8)
  p.add_argument("--pushes", type=float, nargs="*", default=[2.4, 9.8], help="impulses, m/s")
  p.add_argument("--seconds", type=float, default=20.0)
  p.add_argument("--push-s", type=float, default=4.0)
  p.add_argument("--seeds", type=int, nargs="+", default=[0])
  p.add_argument("--device", default=env_device())
  args = p.parse_args()
  cfg = Stage2Cfg.from_dict(json.loads((args.run_dir / "config.json").read_text()))
  t = Stage2Trainer(cfg, args.motions, args.teacher, args.device, collect=False, evaluate=False)
  rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
  report = {
    "code_revision": rev,
    "teacher": {"path": str(args.teacher), "sha256": t.teacher_info["sha256"]},
    "protocol": {
      k: getattr(args, k) for k in ("starts", "samples", "pushes", "seconds", "push_s", "seeds")
    }
    | {"top_p": cfg.top_p, "temperature": cfg.temperature, "nominal_physics": True},
    "results": [],
  }
  for path in args.checkpoints:
    policy = torch.load(path, map_location=args.device, weights_only=False)
    if policy["teacher"]["sha256"] != t.teacher_info["sha256"]:
      raise SystemExit(f"{path} was trained against a different teacher")
    t.prior.load_state_dict(policy["prior"])
    for seed in args.seeds:
      t0 = time.time()
      traj = Q.rollouts(
        t.prior,
        t.teacher,
        t.kin,
        t.kin_torso,
        args.motions,
        args.device,
        starts=args.starts,
        samples=args.samples,
        pushes=tuple(args.pushes),
        seconds=args.seconds,
        push_s=args.push_s,
        top_p=cfg.top_p,
        temperature=cfg.temperature,
        action_clip=cfg.action_clip,
        gamma=cfg.gamma,
        seed=seed,
      )
      entry = {
        "checkpoint": str(path),
        "sha256": E.sha256(path),
        "iteration": policy["iteration"],
        "seed": seed,
        "starts": traj["starts"],
        "floor_starts": [
          n for n, f in zip(traj["starts"], traj["floor_start"].tolist(), strict=True) if f
        ],
        "seconds_wall": round(time.time() - t0, 1),
        "metrics": Q.metrics(traj),
      }
      report["results"].append(entry)
      args.report.write_text(json.dumps(report, indent=2))
      free = entry["metrics"]["free"]
      shown = ("all/10s/fall_stuck_rate", "upright/10s/fall_stuck_rate", "all/10s/apd_pose_m")
      print(
        f"[quality] it={entry['iteration']} seed={seed} "
        + " ".join(f"{k}={free[k]:.4g}" for k in shown),
        flush=True,
      )


if __name__ == "__main__":
  main()
