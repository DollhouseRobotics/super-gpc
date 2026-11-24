"""Joint-target jitter diagnostics (evaluation metrics, not rewards).

Rates are measured on the physical joint position targets in radians,
``default + action_scale * action``; the default cancels in differences. The
total rate is the RMS 50 Hz finite difference. The fast rate first applies a
second-order Butterworth high-pass at ``HIGHPASS_HZ`` to the targets,
separating rapid command oscillation from intended motion, which is mostly
slower. The filter starts at rest on the first target (zero output), as does
the Miniverse controller's copy of it.
"""

from __future__ import annotations

import math

import torch

HIGHPASS_HZ = 5.0
SETTLE_STEPS = 25  # 0.5 s at 50 Hz excluded after every reset


def highpass_coefficients(rate_hz: float, cutoff_hz: float = HIGHPASS_HZ) -> list[float]:
  """Bilinear-transform Butterworth high-pass: ``[b0, b1, b2, a1, a2]``."""
  k = math.tan(math.pi * cutoff_hz / rate_hz)
  norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
  return [
    norm,
    -2.0 * norm,
    norm,
    2.0 * (k * k - 1.0) * norm,
    (1.0 - math.sqrt(2.0) * k + k * k) * norm,
  ]


def highpass(x: torch.Tensor, rate_hz: float) -> torch.Tensor:
  """Filter ``x`` (T, ...) along time."""
  b0, b1, b2, a1, a2 = highpass_coefficients(rate_hz)
  x1 = x2 = x[0]
  y1 = y2 = torch.zeros_like(x[0])
  out = []
  for t in range(x.shape[0]):
    y = b0 * x[t] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
    out.append(y)
    x2, x1, y2, y1 = x1, x[t], y1, y
  return torch.stack(out)


def target_rate_sums(
  targets: torch.Tensor, valid: torch.Tensor, rate_hz: float
) -> dict[str, torch.Tensor]:
  """Per-env, per-joint squared-rate sums.

  ``targets`` is (T, n, J); ``valid`` (T - 1, n) marks the differences between
  steps t and t + 1 to include (both steps in the same live episode, after
  settling).
  """
  m = valid.float()[..., None]
  rate = targets.diff(dim=0) * rate_hz
  fast = highpass(targets, rate_hz).diff(dim=0) * rate_hz
  return dict(
    rate_sq=(rate.square() * m).sum(0),
    fast_sq=(fast.square() * m).sum(0),
    count=valid.float().sum(0),
  )


def target_rate_metrics(
  sums: dict[str, torch.Tensor], joint_names: list[str] | tuple[str, ...], prefix: str
) -> dict[str, float]:
  """Reduce ``target_rate_sums`` (already restricted to the wanted envs)."""
  count = float(sums["count"].sum())
  if count == 0:
    return {}
  rate_sq, fast_sq = sums["rate_sq"].sum(0), sums["fast_sq"].sum(0)
  joints = len(joint_names)
  per_joint_fast = (fast_sq / count).sqrt()
  worst = int(per_joint_fast.argmax())
  return {
    f"{prefix}target_rate_rms_rad_s": float((rate_sq.sum() / (count * joints)).sqrt()),
    f"{prefix}target_rate_fast_rms_rad_s": float((fast_sq.sum() / (count * joints)).sqrt()),
    f"{prefix}target_rate_fast_power_fraction": float(
      fast_sq.sum() / rate_sq.sum().clamp_min(1e-12)
    ),
    f"{prefix}target_rate_fast_max_joint_rad_s": float(per_joint_fast[worst]),
    f"{prefix}target_rate_fast_max_joint_index": float(worst),
  }
