"""FIFO replay of real simulator transitions, ``capacity`` steps per env.

Each slot stores one policy step of every env: the pre-action physical state,
the noisy actor observation actually seen, the executed action, the next
physical state, realized context, motion clip/frame and boundary labels.
``episode`` identifies a contiguous segment: it changes at every reset and at
every collection gap (resume), so neither model windows nor reconstructed
observation histories cross a boundary. ``ep_step`` counts steps within the
segment and reconstructs the reset padding of the decoder history exactly.

The fill level is mirrored on the device (``count_t``) and sampling bounds are
computed from it, so sampling never reads the host and can be replayed from a
CUDA graph while the buffer keeps filling.
"""

from __future__ import annotations

import torch

from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.mdp.events import CONTEXT_DIM

FIELDS = {
  "state": 67,
  "next_state": 67,
  "obs": O.OBS_DIM,
  "action": O.ACTION_DIM,
  "context": CONTEXT_DIM,
}
INDEX_FIELDS = ("clip", "frame", "episode", "ep_step")
FLAG_FIELDS = ("failure", "clip_end", "timeout")


class Replay:
  def __init__(self, capacity: int, num_envs: int, device: str):
    self.capacity, self.num_envs, self.device = capacity, num_envs, device
    self.data = {k: torch.zeros(capacity, num_envs, d, device=device) for k, d in FIELDS.items()}
    for k in INDEX_FIELDS:
      self.data[k] = torch.zeros(capacity, num_envs, dtype=torch.int32, device=device)
    for k in FLAG_FIELDS:
      self.data[k] = torch.zeros(capacity, num_envs, dtype=torch.bool, device=device)
    self.count = 0  # vector steps written in total
    self.count_t = torch.zeros((), dtype=torch.long, device=device)

  def add(self, **values: torch.Tensor) -> None:
    slot = self.count % self.capacity
    for k, v in values.items():
      self.data[k][slot] = v
    self.count += 1
    self.count_t.fill_(self.count)

  def _uniform(self, lo: torch.Tensor, hi: torch.Tensor, n: int) -> torch.Tensor:
    """``n`` integers uniform on [lo, hi) with device-side bounds."""
    u = torch.rand(n, device=self.device, dtype=torch.float64)
    return torch.minimum(lo + (u * (hi - lo).double()).long(), hi - 1)

  @property
  def size(self) -> int:
    return min(self.count, self.capacity)

  def _oldest(self) -> int:
    return max(0, self.count - self.capacity)

  def _gather(self, field: str, t: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    return self.data[field][t % self.capacity, env]

  def sample_windows(self, batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute start steps and envs of ``length``-step windows inside one segment."""
    lo, hi = self._oldest(), self.count - length  # inclusive start range
    if hi < lo:
      raise RuntimeError("replay holds no full window")
    out_t, out_e, have = [], [], 0
    while have < batch:
      n = 2 * (batch - have) + 64
      t = torch.randint(lo, hi + 1, (n,), device=self.device)
      e = torch.randint(0, self.num_envs, (n,), device=self.device)
      ok = self._gather("episode", t, e) == self._gather("episode", t + length - 1, e)
      out_t.append(t[ok])
      out_e.append(e[ok])
      have += int(ok.sum())
    return torch.cat(out_t)[:batch], torch.cat(out_e)[:batch]

  def sample_windows_fixed(
    self, batch: int, length: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``sample_windows`` without host reads: draws 2 x batch + 64 uniform
    candidates and keeps the first ``batch`` that stay inside one segment
    (a stable sort, so the kept windows are uniform over valid windows).
    Returns starts, envs and a validity mask; a window is invalid only in the
    vanishing case of fewer valid candidates than ``batch``, and callers must
    give it zero weight."""
    n = 2 * batch + 64
    lo = (self.count_t - self.capacity).clamp_min(0)
    t = self._uniform(lo, self.count_t - length + 1, n)
    e = torch.randint(0, self.num_envs, (n,), device=self.device)
    ok = self._gather("episode", t, e) == self._gather("episode", t + length - 1, e)
    idx = torch.sort((~ok).to(torch.uint8), stable=True).indices[:batch]
    return t[idx], e[idx], ok[idx]

  def window(
    self, t: torch.Tensor, env: torch.Tensor, length: int, fields=None
  ) -> dict[str, torch.Tensor]:
    steps = t[:, None] + torch.arange(length, device=self.device)
    e = env[:, None].expand_as(steps)
    return {k: self._gather(k, steps, e) for k in (fields or self.data)}

  def sample_starts(self, batch: int, history: int) -> dict[str, torch.Tensor]:
    """Imagination start states with their reconstructed actor histories."""
    lo = (self.count_t - self.capacity).clamp_min(0) + history - 1
    t = self._uniform(lo, self.count_t, batch)
    e = torch.randint(0, self.num_envs, (batch,), device=self.device)
    out = {k: self._gather(k, t, e) for k in ("state", "context", "clip", "frame")}
    back = torch.arange(history - 1, -1, -1, device=self.device)  # oldest first
    ep_step = self._gather("ep_step", t, e).long()
    offset = torch.minimum(back[None], ep_step[:, None])
    out["history"] = self._gather("obs", t[:, None] - offset, e[:, None].expand_as(offset))
    return out

  def state_dict(self) -> dict:
    n = self.size
    if self.count <= self.capacity:
      data = {k: v[:n] for k, v in self.data.items()}
    else:  # rotate so that index 0 is the oldest slot
      s = self.count % self.capacity
      data = {k: torch.cat((v[s:], v[:s])) for k, v in self.data.items()}
    return {"count": self.count, "capacity": self.capacity, "data": data}

  def load_state_dict(self, state: dict) -> None:
    assert state["capacity"] == self.capacity
    data = state["data"]
    n = next(iter(data.values())).shape[0]
    self.count = state["count"]
    self.count_t.fill_(self.count)
    s = self.count % self.capacity if self.count > self.capacity else 0
    # Slot of the oldest stored step is ``s``; restore positions modulo capacity.
    idx = (torch.arange(n, device=self.device) + s) % self.capacity
    for k, v in data.items():
      self.data[k][idx] = v.to(self.device)
