"""Unconditional-sampling quality of a Stage 2 prior.

``starts`` initial states x ``samples`` nucleus-sampled rollouts each, under
free and pushed conditions (impulses of 2.4 and 9.8 m/s, as in GPC Table 9,
Appendix C.3), run for 20 s. Diversity is the pose/root average pairwise
distance (APD) over the first 10 s; robustness is survival and the rate of
falls the robot does not recover from ("fallen and stuck"); smoothness is
normalized jerk (whole body and feet, 0.4 s windows) and mean acceleration.

The prior drives the frozen Stage 1 decoder exactly as at deployment. Starts
are frame 0 of fixed corpus clips with nominal physics and no reset
perturbation, so samples from one start differ only through sampling and
sensor noise. Push directions are seeded per (start, sample) and identical
across checkpoints. This uses its own environment; no training state exists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1
from super_gpc.rl import prior as P
from super_gpc.rl.networks import FSQ_HALF_LEVELS
from super_gpc.tasks.tracking.mdp import kinematics as K
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.mdp.motion_library import POLICY_HZ, MotionLibrary
from super_gpc.tasks.tracking.rl.real_env import RealEnv

FALL_HEIGHT_M = 0.45  # fallen when the pelvis is below this ...
FALL_UPRIGHT = 0.6  # ... or uprightness (body z . world z) is below this
STUCK_FRAMES = 50  # stuck: this many consecutive fallen frames ...
STUCK_MOVE_M = 0.20  # ... while the pelvis moves less than this horizontally
APD_FRAMES = 500  # diversity window: the first 10 s at 50 Hz
JERK_WINDOW_S = 0.4  # GPC C.3, Hogan and Sternad (2009)
JERK_MIN_PATH_M = 0.01  # windows with less body travel are skipped (jerk undefined)
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def start_clips(motions: Path, count: int) -> list[str]:
  """``count`` clips spread evenly over the sorted corpus (all data, no split)."""
  clips = json.loads((Path(motions) / "clips.json").read_text())
  names = sorted(c["name"] for c in clips if c["frames"] > POLICY_HZ)
  stride = max(1, len(names) // count)
  return names[::stride][:count]


def heading_pose(body_pos: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
  """Body positions relative to the pelvis, in its yaw-only heading frame."""
  w, x, y, z = root_quat.unbind(-1)
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  half = -yaw / 2
  zero = torch.zeros_like(half)
  q = torch.stack((torch.cos(half), zero, zero, torch.sin(half)), -1)[..., None, :]
  rel = body_pos - body_pos[..., :1, :]
  return K.quat_rotate(q.expand(*rel.shape[:-1], 4), rel)


@torch.no_grad()
def rollouts(
  prior: P.CodePrior,
  teacher,
  kin,
  kin_torso,
  motions: Path,
  device: str,
  *,
  starts: int,
  samples: int,
  pushes: tuple[float, ...],
  seconds: float,
  push_s: float,
  top_p: float,
  temperature: float,
  action_clip: float,
  gamma: float,
  seed: int,
) -> dict:
  """Run every (condition, start, sample) rollout in one batch; return the
  recorded trajectories on the CPU, shaped (condition, start, sample, frame, ...)."""
  names = start_clips(motions, starts)
  lib = MotionLibrary(motions, device, clip_names=names)
  conditions = (0.0, *pushes)
  C, S, N = len(conditions), len(names), samples
  steps = round(seconds * POLICY_HZ)
  env = RealEnv(
    C * S * N,
    device,
    lib,
    None,
    kin,
    kin_torso,
    gamma,
    steps,
    reset_perturbation=False,
    nominal_physics=True,
    cuda_graphs=True,
  )
  torch.manual_seed(seed)
  gen = torch.Generator(device=device).manual_seed(seed)
  clip = torch.arange(S, device=device)[None, :, None].expand(C, S, N).reshape(-1)
  env.reset(assignment=(clip, torch.zeros_like(clip)))
  s0 = env.state().reshape(C, S, N, -1)[0, :, 0]  # one env per start (samples start equal)
  down = torch.tensor([0.0, 0.0, -1.0], device=device)
  up0 = -K.quat_rotate_inv(s0[:, 3:7], down.expand_as(s0[:, :3]))[:, 2]
  floor = (s0[:, 2] < FALL_HEIGHT_M) | (up0 < FALL_UPRIGHT)
  angle = torch.rand(S, N, generator=gen, device=device) * 2 * math.pi  # shared by conditions
  speed = torch.tensor(conditions, device=device)[:, None, None]
  push = torch.stack((angle.cos(), angle.sin()), -1)[None] * speed[..., None]
  push = push.reshape(-1, 2)
  pushed = speed.expand(C, S, N).reshape(-1) > 0
  prior.eval()
  root, up, bodies = [], [], []
  for t in range(steps):
    if t == round(push_s * POLICY_HZ) and pushed.any():
      robot = env.robot
      ids = pushed.nonzero().squeeze(-1)
      vel = robot.data.root_link_vel_w[ids].clone()
      vel[:, :2] += push[ids]
      robot.write_root_link_velocity_to_sim(vel, env_ids=ids)
    history = env.history.reshape(env.num_envs, -1)
    uniform = torch.rand(env.num_envs, P.NUM_TOKENS, generator=gen, device=device)
    tokens, _ = prior.generate(history[:, -O.OBS_DIM :], uniform, top_p, temperature)
    codes = P.tokens_to_codes(tokens)
    action = teacher.decode(codes / FSQ_HALF_LEVELS, history).clamp(-action_clip, action_clip)
    s = env.step(action, auto_reset=False)["transition"]["next_state"]
    b = kin(s, velocities=False)
    root.append(s[:, :3])
    up.append(-K.quat_rotate_inv(s[:, 3:7], down.expand_as(s[:, :3]))[:, 2])
    bodies.append(torch.cat((b.pos, heading_pose(b.pos, s[:, 3:7])), -1))
  shape = (C, S, N, steps)
  out = {
    "conditions": conditions,
    "starts": names,
    "floor_start": floor.cpu(),  # starts already fallen (crawl or floor poses)
    "root": torch.stack(root, 1).reshape(*shape, 3).cpu(),
    "up": torch.stack(up, 1).reshape(shape).cpu(),
    "bodies": torch.stack(bodies, 1).reshape(*shape, len(kin.body_names), 6).cpu(),
  }
  del env
  return out


def stuck_onset(root: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Per rollout: ever fallen, and the first frame of a stuck window (or T)."""
  fallen = (root[..., 2] < FALL_HEIGHT_M) | (up < FALL_UPRIGHT)
  T, W = fallen.shape[-1], STUCK_FRAMES
  win = fallen.float().unfold(-1, W, 1).min(-1).values > 0  # all W frames fallen
  xy = root[..., :2].unfold(-2, W, 1)  # (..., T-W+1, 2, W)
  moved = (xy - xy[..., :1]).norm(dim=-2).max(-1).values
  stuck = win & (moved < STUCK_MOVE_M)
  first = torch.where(stuck.any(-1), stuck.float().argmax(-1), torch.full(stuck.shape[:-1], T))
  return fallen.any(-1), first


def pairwise(x: torch.Tensor) -> torch.Tensor:
  """(..., N, F, D) -> (..., N, N, F) Euclidean distances between samples."""
  return (x[..., :, None, :, :] - x[..., None, :, :, :]).norm(dim=-1)


def normalized_jerk(pos: torch.Tensor) -> tuple[torch.Tensor, float]:
  """Hogan-Sternad normalized jerk over non-overlapping 0.4 s windows of body
  positions (..., T, B, 3): sqrt(0.5 * T^5 / L^2 * integral |jerk|^2 dt)."""
  dt = 1.0 / POLICY_HZ
  w = round(JERK_WINDOW_S * POLICY_HZ)
  T = pos.shape[-3] // w * w
  win = pos[..., :T, :, :].unflatten(-3, (T // w, w))  # (..., nw, w, B, 3)
  jerk = win.diff(n=3, dim=-3) / dt**3
  length = win.diff(dim=-3).norm(dim=-1).sum(-2)  # (..., nw, B)
  dur = (w - 1) * dt
  integral = (jerk.norm(dim=-1) ** 2).sum(-2) * dt
  nj = torch.sqrt(0.5 * dur**5 / length.clamp_min(1e-9) ** 2 * integral)
  valid = length >= JERK_MIN_PATH_M
  return nj[valid], float(valid.float().mean())


def metrics(traj: dict) -> dict:
  """Per condition, for all starts and separately for upright and floor
  starts (a floor start is already "fallen", so its rates measure getting up)."""
  floor = traj["floor_start"]
  out: dict[str, dict] = {}
  for group, sel in (("all", torch.ones_like(floor)), ("upright", ~floor), ("floor", floor)):
    if sel.any():
      sub = {k: traj[k][:, sel] for k in ("root", "up", "bodies")}
      for cond, m in _metrics(sub | {"conditions": traj["conditions"]}, group == "all").items():
        out.setdefault(cond, {}).update({f"{group}/{k}": v for k, v in m.items()})
        out[cond][f"{group}/starts"] = float(sel.sum())
  return out


def _metrics(traj: dict, per_start: bool = False) -> dict:
  root, up = traj["root"].double(), traj["up"].double()
  world = traj["bodies"][..., :3].double()
  pose = traj["bodies"][..., 3:].double()
  feet = [G1.TRACKED_BODIES.index(n) for n in FEET]
  out: dict[str, dict] = {}
  N = root.shape[2]
  off_diag = ~torch.eye(N, dtype=torch.bool)
  for c, speed in enumerate(traj["conditions"]):
    name = "free" if speed == 0 else f"push_{speed:g}mps"
    m: dict[str, float] = {}
    for label, frames in (("10s", APD_FRAMES), ("full", root.shape[-2])):
      r, u = root[c, :, :, :frames], up[c, :, :, :frames]
      ever, first = stuck_onset(r, u)
      stuck = first < frames
      m[f"{label}/fall_stuck_rate"] = float(stuck.float().mean())
      m[f"{label}/ever_fallen_rate"] = float(ever.float().mean())
      m[f"{label}/recovered_rate"] = float((ever & ~stuck).float().mean())
      m[f"{label}/survival_rate"] = float((~ever).float().mean())
      if per_start:
        m[f"{label}/per_start_fall_stuck_rate"] = stuck.float().mean(-1).tolist()
    # APD over the 10 s window. Pose: includes post-fall frames.
    F = APD_FRAMES
    d_pose = pairwise(pose[c, :, :, :F].flatten(-2)) / math.sqrt(pose.shape[-2])
    m["10s/apd_pose_m"] = float(d_pose.mean(-1)[:, off_diag].mean())
    # Root: common non-stuck prefix; pairs sharing fewer than 50 frames excluded.
    _, first = stuck_onset(root[c, :, :, :F], up[c, :, :, :F])
    common = torch.minimum(first[:, :, None], first[:, None, :])  # (S, N, N)
    d_root = pairwise(root[c, :, :, :F, :2])
    mask = torch.arange(F)[None, None, None] < common[..., None]
    per_pair = (d_root * mask).sum(-1) / common.clamp_min(1)
    keep = (common >= STUCK_FRAMES) & off_diag
    m["10s/apd_root_m"] = float(per_pair[keep].mean()) if keep.any() else float("nan")
    m["10s/apd_root_pairs"] = float(keep.sum())
    # GPC smoothness over the full rollout.
    nj, frac = normalized_jerk(world[c])
    njf, _ = normalized_jerk(world[c][..., feet, :])
    m["full/norm_jerk_mean"], m["full/norm_jerk_median"] = float(nj.mean()), float(nj.median())
    m["full/foot_norm_jerk_median"] = float(njf.median())
    m["full/jerk_valid_window_fraction"] = frac
    acc = world[c].diff(n=2, dim=-3).norm(dim=-1) * POLICY_HZ**2
    m["full/mean_accel_mps2"] = float(acc.mean())
    out[name] = m
  return out
