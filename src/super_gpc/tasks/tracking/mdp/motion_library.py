"""Reference-motion corpus preparation and GPU motion library.

Prepared format (one directory):
  qpos.npy  (T, 34) float32: pelvis pos (3), WXYZ quat (4), 27 joints (action order)
  qvel.npy  (T, 33) float32: pelvis lin vel world (3), ang vel body (3), 27 joint vel
  clips.json: per clip name, group, split, offset, frames, category and source ids
  manifest.json: corpus revision, checksums, rates, layout, split rule and totals

Source shards store 30 Hz ProtoMotions tensors. The probe in
docs/spec.md shows ``dps`` uses MJCF tree joint order, ``grs``
XYZW quaternions and ``gts[:, 0]`` the pelvis origin (FK error 2.4e-7 m), while
stored velocities are forward differences that do not equal link-origin
velocities. We therefore resample generalized coordinates to the 50 Hz policy
rate and derive all velocities and body poses consistently from them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from super_gpc.tasks.tracking.mdp import kinematics as K

POLICY_HZ = 50
SOURCE_HZ = 30


def _slerp(q0: np.ndarray, q1: np.ndarray, a: np.ndarray) -> np.ndarray:
  dot = (q0 * q1).sum(-1, keepdims=True)
  q1 = np.where(dot < 0, -q1, q1)
  dot = np.abs(dot).clip(max=1.0)
  theta = np.arccos(dot)
  s = np.sin(theta)
  small = s < 1e-6
  w0 = np.where(small, 1 - a, np.sin((1 - a) * theta) / np.where(small, 1, s))
  w1 = np.where(small, a, np.sin(a * theta) / np.where(small, 1, s))
  q = w0 * q0 + w1 * q1
  return q / np.linalg.norm(q, axis=-1, keepdims=True)


def split_of(group: str, val_percent: int) -> str:
  h = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
  return "val" if h < val_percent else "train"


def resample_shard(data: dict, tree_to_action: np.ndarray, records: list[dict]):
  """Resample one shard's clips to 50 Hz generalized state."""
  lengths = data["motion_num_frames"].numpy()
  starts = data["length_starts"].numpy()
  root_p = data["gts"][:, 0].numpy()
  root_q = data["grs"][:, 0].numpy()[:, [3, 0, 1, 2]]  # XYZW -> WXYZ
  dof = data["dps"].numpy()[:, tree_to_action]
  qpos_out, qvel_out, meta = [], [], []
  for rec in records:
    i = rec["outputIndex"]
    n, s = int(lengths[i]), int(starts[i])
    if rec["frames"] != n:
      raise ValueError(f"frame count mismatch for {rec['filename']}")
    duration = (n - 1) / SOURCE_HZ
    m = int(np.floor(duration * POLICY_HZ + 1e-9)) + 1
    src = np.arange(m) * (SOURCE_HZ / POLICY_HZ)
    i0 = np.minimum(np.floor(src).astype(np.int64), n - 2)
    a = (src - i0)[:, None]
    p = root_p[s + i0] * (1 - a) + root_p[s + i0 + 1] * a
    q = _slerp(root_q[s + i0], root_q[s + i0 + 1], a)
    j = dof[s + i0] * (1 - a) + dof[s + i0 + 1] * a
    dt = 1.0 / POLICY_HZ
    v = np.gradient(p, dt, axis=0)
    jd = np.gradient(j, dt, axis=0)
    qt = torch.from_numpy(q).double()
    rel = K.quat_to_rotvec(K.quat_mul(K.quat_conj(qt[:-1]), qt[1:])).numpy() / dt
    w = np.empty_like(p)
    w[0], w[-1] = rel[0], rel[-1]
    w[1:-1] = 0.5 * (rel[1:] + rel[:-1])
    qpos_out.append(np.concatenate([p, q, j], 1).astype(np.float32))
    qvel_out.append(np.concatenate([v, w, jd], 1).astype(np.float32))
    meta.append(m)
  return qpos_out, qvel_out, meta


class MotionLibrary:
  """Prepared clips on the training device."""

  def __init__(
    self,
    directory: Path,
    device: str,
    split: str | None = None,
    clip_names: list[str] | None = None,
  ):
    directory = Path(directory)
    self.manifest = json.loads((directory / "manifest.json").read_text())
    clips = json.loads((directory / "clips.json").read_text())
    if clip_names is not None:
      by_name = {c["name"]: c for c in clips}
      clips = [by_name[n] for n in clip_names]
    elif split is not None:
      clips = [c for c in clips if c["split"] == split]
    qpos = np.load(directory / "qpos.npy", mmap_mode="r")
    qvel = np.load(directory / "qvel.npy", mmap_mode="r")
    rows = np.concatenate([np.arange(c["offset"], c["offset"] + c["frames"]) for c in clips])
    self.clips = clips
    self.device = device
    self.qpos = torch.from_numpy(np.ascontiguousarray(qpos[rows])).to(device)
    self.qvel = torch.from_numpy(np.ascontiguousarray(qvel[rows])).to(device)
    lengths = torch.tensor([c["frames"] for c in clips], dtype=torch.long)
    self.lengths = lengths.to(device)
    self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]]).to(device)
    self.num_clips = len(clips)
    # One-second sampling bins (50 frames). A bin is a valid start if it has a
    # frame that is not the final clip frame.
    nb = ((lengths - 1 + POLICY_HZ - 1) // POLICY_HZ).clamp_min(1)
    self.bin_clip = torch.repeat_interleave(torch.arange(len(clips)), nb).to(device)
    first = torch.cat([torch.zeros(1, dtype=torch.long), nb.cumsum(0)[:-1]])
    self.bin_first = first.to(device)
    self.bin_index = (torch.arange(int(nb.sum())) - torch.repeat_interleave(first, nb)).to(device)
    self.num_bins = int(nb.sum())

  def state(self, clip: torch.Tensor, frame: torch.Tensor, pad_zero_velocity: bool = True):
    """Generalized reference state; frames past clip end hold the final pose
    with zero velocities (encoder lookups only)."""
    last = self.lengths[clip] - 1
    past = frame > last
    row = self.offsets[clip] + torch.minimum(frame, last)
    qp, qv = self.qpos[row], self.qvel[row]
    if pad_zero_velocity:
      qv = torch.where(past[..., None], torch.zeros_like(qv), qv)
    root_v, root_w, jd = qv[..., 0:3], qv[..., 3:6], qv[..., 6:33]
    return torch.cat([qp[..., 0:7], root_v, root_w, qp[..., 7:34], jd], -1)

  def sample_start(self, bins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    clip = self.bin_clip[bins]
    lo = self.bin_index[bins] * POLICY_HZ
    hi = torch.minimum(lo + POLICY_HZ, self.lengths[clip] - 1)  # exclusive
    frame = lo + (torch.rand(bins.shape, device=self.device) * (hi - lo)).long()
    return clip, torch.minimum(frame, hi - 1)
