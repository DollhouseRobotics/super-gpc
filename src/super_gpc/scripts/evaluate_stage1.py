"""Evaluate a Stage 1 run's native state and demonstrate training-state preservation.

Loads ``<run-dir>/resume.pt``, collects ``--collect-steps`` real training steps
so every training episode has nonzero progress, motion clock, return and
history, then runs the held-out/training-clip evaluation side rollout in the
separate evaluation environment. The JSON report records per-env episode
lengths, motion clip/frame, returns and state digests before and after, and the
evaluation metrics. Nothing is written to the run's checkpoints or Miniverse.

With ``--standing`` it instead runs only the native-MuJoCo standing evaluation
(``rl/standing.py``, clean and with sensor noise) of ``--checkpoint`` (a
``policy_*.pt`` or ``resume.pt``; default the run's ``resume.pt``) on the run's
embodiment and standing poses. It needs no GPU.

Examples:
  gpc-evaluate-stage1 --motions data/motions --run-dir runs/stage1 \\
    --report runs/stage1/preservation_report.json
  gpc-evaluate-stage1 --motions data/motions --run-dir runs/stage1 --standing \\
    --checkpoint runs/stage1/checkpoints/policy_0002000.pt --report standing.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from super_gpc.rl.networks import Tracker, tracker_weights
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.rl import standing as S
from super_gpc.tasks.tracking.rl.stage1 import Stage1Trainer, build_kinematics
from super_gpc.tasks.tracking.rl.stage1_cfg import Stage1Cfg
from super_gpc.tasks.tracking.rl.stage1_run import env_device


def snapshot(t: Stage1Trainer, envs: int) -> dict:
  e = t.train_env
  return {
    "ep_len_steps": e.ep_len[:envs].tolist(),
    "mjlab_episode_length": e.env.episode_length_buf[:envs].tolist(),
    "clip": e.cmd.clip[:envs].tolist(),
    "frame": e.cmd.frame[:envs].tolist(),
    "ep_return": [round(x, 6) for x in e.ep_return[:envs].tolist()],
    "pelvis_z": [round(x, 6) for x in e.state()[:envs, 2].tolist()],
    "history_abs_sum": [round(x, 4) for x in e.history[:envs].abs().sum((1, 2)).tolist()],
    "mean_ep_len_s": float(e.ep_len.float().mean() / 50),
    "nonzero_progress_fraction": float((e.ep_len > 0).float().mean()),
  }


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("--motions", type=Path, required=True)
  p.add_argument("--run-dir", type=Path, required=True)
  p.add_argument("--report", type=Path, required=True)
  p.add_argument("--collect-steps", type=int, default=37)
  p.add_argument("--device", default=env_device())
  p.add_argument("--standing", action="store_true", help="only the standing evaluation")
  p.add_argument("--checkpoint", type=Path, help="tracker file for --standing")
  args = p.parse_args()
  stored = json.loads((args.run_dir / "config.json").read_text())
  for key in ("actor_betas", "example_clips", "reference_offsets"):
    stored[key] = tuple(stored[key])
  cfg = Stage1Cfg(**stored)
  if args.standing:
    standing(args, cfg)
    return
  t = Stage1Trainer(cfg, args.motions, args.device)
  t.load_resumable(
    torch.load(args.run_dir / "resume.pt", map_location=args.device, weights_only=False)
  )
  t.collect(args.collect_steps, cfg.action_noise_std)
  t.pop_stats()
  shown = min(8, cfg.num_envs)
  before_state = snapshot(t, shown)
  metrics, before, after = t.evaluate()
  after_state = snapshot(t, shown)
  t.collect(1, cfg.action_noise_std)  # training continues from the preserved episodes
  continued = snapshot(t, shown)
  report = {
    "iteration": t.iteration,
    "collect_steps_before_eval": args.collect_steps,
    "fingerprint_before": before,
    "fingerprint_after": after,
    "fingerprint_equal": before == after,
    "episodes_before": before_state,
    "episodes_after": after_state,
    "episodes_equal": before_state == after_state,
    "after_one_more_training_step": continued,
    "metrics": metrics,
  }
  args.report.parent.mkdir(parents=True, exist_ok=True)
  args.report.write_text(json.dumps(report, indent=2))
  print(
    json.dumps(
      {k: report[k] for k in ("iteration", "fingerprint_equal", "episodes_equal")}
      | {
        "mean_ep_len_s": before_state["mean_ep_len_s"],
        "nonzero_progress_fraction": before_state["nonzero_progress_fraction"],
      }
    )
  )


def standing(args, cfg: Stage1Cfg) -> None:
  path = args.checkpoint or args.run_dir / "resume.pt"
  state = torch.load(path, map_location="cpu", weights_only=False)
  tracker = Tracker(O.ENCODER_DIM, O.HISTORY_DIM, O.ACTION_DIM)
  tracker.load_state_dict(tracker_weights(state))
  _, kin_torso = build_kinematics("cpu")
  poses_path = args.run_dir / "standing_poses.json"
  poses = json.loads(poses_path.read_text()) if poses_path.exists() else S.quiet_poses(args.motions)
  xml = args.run_dir / "bundle" / "source" / "embodiment" / "robot.xml"
  results = [
    S.standing_check(xml, poses, tracker, kin_torso, cfg.action_clip, noisy)
    for noisy in (False, True)
  ]
  report = {"iteration": state.get("iteration"), "checkpoint": str(path), "standing": results}
  args.report.parent.mkdir(parents=True, exist_ok=True)
  args.report.write_text(json.dumps(report, indent=2))
  print(
    json.dumps(
      {"iteration": report["iteration"]} | {k: v for r in results for k, v in r["metrics"].items()}
    )
  )


if __name__ == "__main__":
  main()
