"""Standing-in-place evaluation of a tracker in native MuJoCo.

Holds single quiet poses for ``seconds``: the reference is one frame with zero
velocity, repeated for every current and future frame. The robot starts at
that pose on the exported embodiment with the training ground plane, nominal
physics and the training-code inference path (the export closed-loop check
shows it equals the ONNX controller). Reports falls, pelvis and foot drift,
joint error and the physical target rates of ``mdp/jitter.py`` after the
settling period, clean and with training-scale sensor noise drawn from a
local generator (the global RNG is not used).

Poses are the quietest upright one-second windows of distinct held-out
validation clips, chosen deterministically from the prepared corpus.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1
from super_gpc.rl.networks import Tracker
from super_gpc.tasks.tracking.mdp import jitter as J
from super_gpc.tasks.tracking.mdp import kinematics as K
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.mdp import rewards as R
from super_gpc.tasks.tracking.mdp.motion_library import POLICY_HZ
from super_gpc.tasks.tracking.tracking_env_cfg import DECIMATION

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def quiet_poses(motions: Path, count: int = 5) -> list[dict]:
  """Center frames of the quietest upright 1 s windows of ``count`` clips:
  validation clips when the corpus has a split, otherwise any clip."""
  motions = Path(motions)
  clips = json.loads((motions / "clips.json").read_text())
  split = "val" if any(c["split"] == "val" for c in clips) else "train"
  qpos = np.load(motions / "qpos.npy", mmap_mode="r")
  qvel = np.load(motions / "qvel.npy", mmap_mode="r")
  cands = []
  for c in clips:
    if c["split"] != split or c["frames"] < POLICY_HZ + 1:
      continue
    v = np.asarray(qvel[c["offset"] : c["offset"] + c["frames"]])
    speed = np.sqrt((v[:, 6:] ** 2).mean(1) + (v[:, 0:3] ** 2).sum(1))
    window = np.convolve(speed, np.ones(POLICY_HZ) / POLICY_HZ, "valid")
    i = int(window.argmin())
    row = c["offset"] + i + POLICY_HZ // 2
    q = np.asarray(qpos[row])
    if q[2] > 0.70 and abs(q[4]) < 0.1 and abs(q[5]) < 0.1:  # upright pelvis (WXYZ)
      cands.append((float(window[i]), c["name"], int(row)))
  cands.sort()
  return [
    dict(clip=name, row=row, window_speed=score, qpos=np.asarray(qpos[row], np.float32).tolist())
    for score, name, row in cands[:count]
  ]


def standing_check(
  robot_xml: Path,
  poses: list[dict],
  tracker: Tracker,
  kin_torso: K.Kinematics,
  action_clip: float,
  noisy: bool,
  seconds: float = 10.0,
  seed: int = 0,
  start_raise_m: float = 0.0,
) -> dict:
  """``start_raise_m`` lifts the initial robot pose (not the reference), for
  embodiments whose soles are thicker than the bare foot."""
  import mujoco

  spec = mujoco.MjSpec.from_file(str(robot_xml))
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05])
  model = spec.compile()
  data = mujoco.MjData(model)
  names = G1.ACTION_JOINT_NAMES
  jq = [int(model.jnt_qposadr[model.joint(n).id]) for n in names]
  jv = [int(model.jnt_dofadr[model.joint(n).id]) for n in names]
  feet = [model.body(n).id for n in FEET]
  default_q = torch.tensor([G1.DEFAULT_JOINT_POS[n] for n in names])
  scale = torch.tensor([G1.ACTION_SCALE[n] for n in names])
  noise_scale = O.noise_scale("cpu")
  gen = torch.Generator().manual_seed(seed)
  tracker = copy.deepcopy(tracker).cpu().eval()
  kin_torso = copy.deepcopy(kin_torso).cpu()
  steps = round(seconds * POLICY_HZ)
  targets = torch.zeros(steps, len(poses), len(names))
  alive = torch.zeros(steps, len(poses), dtype=torch.bool)
  rows = []

  def state() -> torch.Tensor:
    s = np.concatenate((data.qpos[0:7], data.qvel[0:6], data.qpos[jq], data.qvel[jv]))
    return torch.from_numpy(s.astype(np.float32))[None]

  with torch.no_grad():
    for p, pose in enumerate(poses):
      q = np.asarray(pose["qpos"], np.float64)
      ref = torch.zeros(1, 67)
      ref[0, 0:7] = torch.from_numpy(q[0:7]).float()
      ref[0, 13:40] = torch.from_numpy(q[7:34]).float()
      future = ref[:, None].expand(1, O.FUTURE, 67)
      mujoco.mj_resetData(model, data)
      data.qpos[0:7], data.qpos[jq] = q[0:7], q[7:34]
      data.qpos[2] += start_raise_m
      mujoco.mj_forward(model, data)
      feet0 = data.xpos[feet, :2].copy()
      prev, hist, errs, fell = torch.zeros(1, O.ACTION_DIM), None, [], -1
      for i in range(steps):
        s = state()
        obs = O.actor_observation(s, prev, torch.zeros(1, len(names)), default_q)
        if noisy:
          obs = obs + (2.0 * torch.rand(obs.shape, generator=gen) - 1.0) * noise_scale
        hist = (
          obs.repeat(1, O.HISTORY) if hist is None else torch.cat((hist[:, O.OBS_DIM :], obs), -1)
        )
        a = tracker(O.encoder_features(kin_torso, s, ref, future), hist)
        a = a.clamp(-action_clip, action_clip)
        targets[i, p] = default_q + scale * a[0]
        alive[i, p] = True
        data.ctrl[:] = targets[i, p].numpy()
        for _ in range(DECIMATION):
          mujoco.mj_step(model, data)
        s1 = state()
        if i >= J.SETTLE_STEPS:
          errs.append(float((s1[0, 13:40] - ref[0, 13:40]).square().mean().sqrt()))
        prev = a
        if bool(R.physical_failure(s1, ref)[0]):
          fell = i + 1
          break
      rows.append(
        dict(
          clip=pose["clip"],
          fell_at_step=fell,
          pelvis_xy_drift_m=float(np.linalg.norm(data.qpos[0:2] - q[0:2])),
          foot_xy_displacement_m=float(np.linalg.norm(data.xpos[feet, :2] - feet0, axis=1).max()),
          joint_pos_rmse_rad=float(np.mean(errs)) if errs else None,
        )
      )
  live = alive[1:] & alive[:-1]
  live[: J.SETTLE_STEPS] = False
  sums = J.target_rate_sums(targets, live, POLICY_HZ)
  per_pose = [
    J.target_rate_metrics({k: v[p : p + 1] for k, v in sums.items()}, names, "")
    for p in range(len(poses))
  ]
  for row, rates in zip(rows, per_pose, strict=True):
    row.update(rates)
  prefix = "standing_noisy/" if noisy else "standing_clean/"
  ok = [r for r in rows if r["joint_pos_rmse_rad"] is not None]
  metrics = {
    f"{prefix}fall_fraction": float(np.mean([r["fell_at_step"] >= 0 for r in rows])),
    f"{prefix}pelvis_xy_drift_m": float(np.mean([r["pelvis_xy_drift_m"] for r in rows])),
    f"{prefix}pelvis_xy_drift_max_m": float(np.max([r["pelvis_xy_drift_m"] for r in rows])),
    f"{prefix}foot_xy_displacement_max_m": float(
      np.max([r["foot_xy_displacement_m"] for r in rows])
    ),
  } | J.target_rate_metrics(sums, names, prefix)
  if ok:
    metrics[f"{prefix}joint_pos_rmse_rad"] = float(np.mean([r["joint_pos_rmse_rad"] for r in ok]))
  return dict(metrics=metrics, poses=rows, seconds=seconds, noisy=noisy, seed=seed)
