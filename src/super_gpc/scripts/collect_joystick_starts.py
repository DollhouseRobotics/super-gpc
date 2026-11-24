"""Record get-down start states from a trained joystick CoLA checkpoint.

Example:
  gpc-collect-joystick-starts --cola runs/joystick/checkpoints/policy_0002500.pt \\
    --out data/joystick_starts.pt
Writes the bank and a JSON summary next to it (``<out>.json``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from super_gpc.tasks.get_down.rl.joystick_starts import collect
from super_gpc.tasks.tracking.rl.stage1_run import env_device
from super_gpc.tasks.tracking.rl.stage2 import file_sha256


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("--cola", type=Path, required=True, help="joystick CoLA policy checkpoint")
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--steps", type=int, default=1500, help="policy steps (50 Hz)")
  p.add_argument("--seed", type=int, default=31337)
  p.add_argument("--device", default=env_device())
  args = p.parse_args()
  bank = collect(args.cola, args.num_envs, args.steps, args.device, args.seed)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  torch.save(bank, args.out)
  summary = bank["meta"] | {
    "states": len(bank["states"]),
    "examples": bank["examples"],
    "sha256": file_sha256(args.out),
  }
  args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary), flush=True)


if __name__ == "__main__":
  main()
