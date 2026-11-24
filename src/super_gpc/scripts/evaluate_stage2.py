"""Evaluate a Stage 2 prior, demonstrate training-state preservation and
optionally export a checkpoint for local parity and closed-loop checks.

Loads ``<run-dir>/resume.pt`` (or the prior of ``--checkpoint``, a
``policy_*.pt``/``latest.pt`` file of the same run), collects
``--collect-steps`` real expert steps so every training episode has nonzero
progress, motion clock, return and history, then runs the held-out token and
closed-loop generation evaluation in the separate evaluation environment. The
JSON report records per-env episode state and state digests before and after.
With ``--export DIR`` it also builds the checkpoint's Miniverse source bundle
locally, checks native/ONNX parity and runs the bundle's controller in native
MuJoCo. Nothing is written to the run's checkpoints or to Miniverse.

Example:
  gpc-evaluate-stage2 --motions data/motions --run-dir runs/stage2 \\
    --teacher runs/stage1/checkpoints/policy_0020000.pt \\
    --report runs/stage2/eval_report.json --export runs/stage2/local_export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from super_gpc.tasks.tracking.mdp.motion_library import MotionLibrary
from super_gpc.tasks.tracking.rl import export as E
from super_gpc.tasks.tracking.rl import stage2_export as X
from super_gpc.tasks.tracking.rl.stage1_run import env_device
from super_gpc.tasks.tracking.rl.stage2 import Stage2Trainer
from super_gpc.tasks.tracking.rl.stage2_cfg import Stage2Cfg


def snapshot(t: Stage2Trainer, envs: int) -> dict:
  e = t.train_env
  return {
    "ep_len_steps": e.ep_len[:envs].tolist(),
    "mjlab_episode_length": e.env.episode_length_buf[:envs].tolist(),
    "clip": e.cmd.clip[:envs].tolist(),
    "frame": e.cmd.frame[:envs].tolist(),
    "ep_return": [round(x, 6) for x in e.ep_return[:envs].tolist()],
    "pelvis_z": [round(x, 6) for x in e.state()[:envs, 2].tolist()],
    "history_abs_sum": [round(x, 4) for x in e.history[:envs].abs().sum((1, 2)).tolist()],
    "obs_norm_updates": int(t.prior.obs_norm.updates),
    "mean_ep_len_s": float(e.ep_len.float().mean() / 50),
    "nonzero_progress_fraction": float((e.ep_len > 0).float().mean()),
  }


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("--motions", type=Path, required=True)
  p.add_argument("--run-dir", type=Path, required=True)
  p.add_argument("--teacher", type=Path, required=True)
  p.add_argument("--checkpoint", type=Path, help="evaluate this policy file's prior instead")
  p.add_argument("--report", type=Path, required=True)
  p.add_argument("--collect-steps", type=int, default=37)
  p.add_argument("--export", type=Path, help="build and check a local bundle in this directory")
  p.add_argument("--device", default=env_device())
  args = p.parse_args()
  cfg = Stage2Cfg.from_dict(json.loads((args.run_dir / "config.json").read_text()))
  t = Stage2Trainer(cfg, args.motions, args.teacher, args.device)
  t.load_resumable(
    torch.load(args.run_dir / "resume.pt", map_location=args.device, weights_only=False)
  )
  if args.checkpoint:
    policy = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if policy["teacher"]["sha256"] != t.teacher_info["sha256"]:
      raise SystemExit("checkpoint was trained against a different teacher")
    t.prior.load_state_dict(policy["prior"])
    t.iteration = policy["iteration"]
  for _ in range(-(-args.collect_steps // cfg.rollout_steps)):  # whole collection chunks
    t.collect()
  t.pop_stats()
  shown = min(8, cfg.num_envs)
  before_state = snapshot(t, shown)
  metrics, before, after = t.evaluate()
  after_state = snapshot(t, shown)
  report = {
    "iteration": t.iteration,
    "teacher": t.teacher_info,
    "fingerprint_before": before,
    "fingerprint_after": after,
    "fingerprint_equal": before == after,
    "episodes_before": before_state,
    "episodes_after": after_state,
    "episodes_equal": before_state == after_state,
    "metrics": metrics,
  }
  if args.export:
    names = list(cfg.example_clips) or E.select_examples(args.motions)
    starts = MotionLibrary(args.motions, "cpu", clip_names=names)
    graph = X.PriorPolicyGraph(t.prior, t.teacher, cfg.top_p, cfg.temperature, cfg.action_clip)
    info = X.build_bundle(
      args.export,
      graph,
      starts,
      cfg.demo_seeds,
      f"super-gpc-stage2-local-{args.run_dir.name}",
      "GPC motion prior",
      {"iteration": t.iteration, "local": True},
      {"iteration": t.iteration, "local": True},
    )
    onnx = Path(info["onnx"])
    report["export"] = info | {
      "parity": X.parity(graph, onnx, X.parity_samples(starts, t.iteration)),
      "latency": X.latency(onnx),
      "closed_loop": X.closed_loop_check(args.export, starts, t.prior, t.teacher, cfg),
    }
  args.report.parent.mkdir(parents=True, exist_ok=True)
  args.report.write_text(json.dumps(report, indent=2))
  summary = {k: report[k] for k in ("iteration", "fingerprint_equal", "episodes_equal")}
  summary |= {
    "nonzero_progress_fraction": before_state["nonzero_progress_fraction"],
    "heldout_val_ce": metrics.get("heldout_val/token_cross_entropy"),
    "gen_val_survival_s": metrics.get("gen_val/survival_s"),
  }
  if args.export:
    summary["parity_token_mismatch"] = report["export"]["parity"][
      "export/parity_token_mismatch_samples"
    ]
    summary["closed_loop_token_mismatch_steps"] = report["export"]["closed_loop"][
      "token_mismatch_steps"
    ]
  print(json.dumps(summary))


if __name__ == "__main__":
  main()
