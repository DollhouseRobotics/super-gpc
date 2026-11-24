"""Record get-down start states from joystick CoLA steering rollouts.

Get-down starts half of its episodes from states recorded while a joystick
controller runs, so it learns to get down from standing, moving and disturbed
configurations, not only from the motion corpus. This runs a trained joystick CoLA
checkpoint in the MJLab steering environment (training commands with a
raised stop probability, Stage 1 physical randomization and sensor noise),
adds random planar velocity pushes, and keeps env-local generalized states of
robots that have not fallen. Labels: ``disturbed`` within one second after a
push, else ``standing`` when the command and the pelvis are near rest, else
``moving``. The bank also names a few representative states for the
Miniverse start switcher.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import torch

from super_gpc.rl import prior as P
from super_gpc.tasks.get_down import mdp as M
from super_gpc.tasks.steering import mdp as SM
from super_gpc.tasks.steering.rl.cola import tf32
from super_gpc.tasks.steering.rl.cola_cfg import ColaCfg
from super_gpc.tasks.steering.rl.cola_env import SteeringEnv
from super_gpc.tasks.steering.rl.cola_run import _policy_from_state
from super_gpc.tasks.tracking.rl.stage2 import file_sha256

DISTURBED_STEPS = 50  # 1 s after a push


def _examples(states: torch.Tensor, labels: torch.Tensor, commands: torch.Tensor) -> dict:
  """First bank state of each named kind (deterministic)."""
  fwd, left = commands[:, 0], commands[:, 1]
  moving = labels == M.LABELS.index("moving")
  kinds = {
    "standing": labels == M.LABELS.index("standing"),
    "walking forward": moving & (fwd > 0.8) & (left.abs() < 0.2),
    "walking backward": moving & (fwd < -0.5) & (left.abs() < 0.2),
    "walking sideways": moving & (left.abs() > 0.6) & (fwd.abs() < 0.2),
    "just pushed": labels == M.LABELS.index("disturbed"),
  }
  out = {}
  for name, mask in kinds.items():
    hit = torch.nonzero(mask).squeeze(-1)
    if len(hit):
      out[name] = int(hit[0])
  return out


@torch.no_grad()
def collect(
  cola: Path,
  num_envs: int,
  steps: int,
  device: str,
  seed: int,
  record_every: int = 25,
  warmup_steps: int = 25,
  stop_probability: float = 0.25,
  push_probability: float = 0.01,
  push_mps: tuple[float, float] = (0.2, 1.0),
) -> dict:
  state = torch.load(cola, map_location="cpu", weights_only=False)
  if state.get("stage") != "rlft-cola":
    raise ValueError(f"{cola} is not a joystick CoLA policy checkpoint")
  policy = _policy_from_state(state).to(device)
  cfg = replace(ColaCfg.from_dict(state["cfg"]), stop_probability=stop_probability)
  env = SteeringEnv(cfg, num_envs, device, seed)
  gen = torch.Generator(device=device).manual_seed(seed + 1)
  torch.manual_seed(seed + 2)
  idx = env.robot.indexing
  env.reset()
  since_push = torch.full((num_envs,), 10**6, device=device)
  rows: dict[str, list[torch.Tensor]] = {"states": [], "labels": [], "commands": []}
  falls = pushes = 0
  for t in range(steps):
    inp = env.inputs()
    uniform = torch.rand(num_envs, P.NUM_TOKENS, device=device, generator=gen)
    with tf32(cfg.tf32):
      tokens, _ = policy.sample(inp["obs"], inp["command"], uniform)
    out = env.step(policy.decode(tokens, inp["history"]))
    falls += int(out["terminated"].sum())
    done = out["done"]
    since_push = torch.where(done, 10**6, since_push + 1)
    env.finish(done)
    r = torch.rand(num_envs, 3, device=device, generator=gen)
    hit = torch.nonzero(r[:, 0] < push_probability).squeeze(-1)
    if len(hit):
      speed = push_mps[0] + (push_mps[1] - push_mps[0]) * r[hit, 1]
      angle = (2 * r[hit, 2] - 1) * math.pi
      dv = speed[:, None] * torch.stack((angle.cos(), angle.sin()), -1)
      qvel = env.robot.data.data.qvel
      lin = idx.free_joint_v_adr[:2]
      qvel[hit[:, None], lin[None]] += dv
      since_push[hit] = 0
      pushes += len(hit)
    if t % record_every:
      continue
    s = env.state()
    yaw = SM.yaw_wxyz(s[:, 3:7])
    upright = SM.uprightness_wxyz(s[:, 3:7])
    keep = (s[:, 2] > cfg.fallen_height) & (upright > cfg.fallen_uprightness)
    keep &= (env.ep_len >= warmup_steps) & torch.isfinite(s).all(-1)
    command = env.commands.local(yaw)
    rest = (command[:, :2].norm(dim=-1) < 0.1) & (s[:, 7:9].norm(dim=-1) < 0.2)
    label = torch.where(
      since_push < DISTURBED_STEPS,
      M.LABELS.index("disturbed"),
      torch.where(rest, M.LABELS.index("standing"), M.LABELS.index("moving")),
    )
    rows["states"].append(s[keep].cpu())
    rows["labels"].append(label[keep].cpu())
    rows["commands"].append(command[keep].cpu())
  bank = {k: torch.cat(v) for k, v in rows.items()}
  counts = {name: int((bank["labels"] == i).sum()) for i, name in enumerate(M.LABELS)}
  return bank | {
    "kind": "joystick-starts",
    "label_names": list(M.LABELS),
    "examples": _examples(bank["states"], bank["labels"], bank["commands"]),
    "meta": {
      "cola_checkpoint": str(cola),
      "cola_sha256": file_sha256(cola),
      "cola_iteration": int(state["iteration"]),
      "prior": state["prior_info"],
      "num_envs": num_envs,
      "steps": steps,
      "seed": seed,
      "record_every": record_every,
      "warmup_steps": warmup_steps,
      "stop_probability": stop_probability,
      "push_probability": push_probability,
      "push_mps": list(push_mps),
      "disturbed_steps": DISTURBED_STEPS,
      "terminations": falls,
      "pushes": pushes,
      "counts": counts,
    },
  }
