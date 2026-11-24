"""Stage 1 networks: FSQ tracker, learned dynamics and critic.

Widths follow the specification (SuperMimic Table A.4, with a 16-scalar FSQ
latent).
Activations, normalization and initialization are not published; the choices
here are recorded in docs/spec.md.
"""

from __future__ import annotations

import torch
from torch import nn

FSQ_DIM = 16
FSQ_HALF_LEVELS = 3  # seven levels: integers -3..3


def tracker_weights(checkpoint: dict) -> dict:
  """Inference uses the EMA policy; older checkpoints have only live weights."""
  return checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["tracker"]


def mlp(
  inp: int, hidden: tuple[int, ...], out: int, act: type[nn.Module], layer_norm: bool = False
) -> nn.Sequential:
  layers: list[nn.Module] = []
  for h in hidden:
    layers.append(nn.Linear(inp, h))
    if layer_norm:
      layers.append(nn.LayerNorm(h))
    layers.append(act())
    inp = h
  layers.append(nn.Linear(inp, out))
  return nn.Sequential(*layers)


class Normalizer(nn.Module):
  """Fixed affine input normalization, fitted once from the initial replay fill."""

  def __init__(self, dim: int):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("std", torch.ones(dim))

  @torch.no_grad()
  def fit(self, x: torch.Tensor, min_std: float = 1e-2) -> None:
    x = x.reshape(-1, x.shape[-1]).double()
    self.mean.copy_(x.mean(0).float())
    self.std.copy_(x.std(0).clamp_min(min_std).float())

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / self.std


def round_half_even(x: torch.Tensor) -> torch.Tensor:
  """``torch.round`` (half to even) from Floor/compare ops, which the Miniverse
  ONNX runtime allowlist accepts (it rejects ``Round``)."""
  lower = torch.floor(x)
  frac = x - lower
  even = lower - 2 * torch.floor(lower / 2) == 0
  up = (frac > 0.5) | ((frac == 0.5) & ~even)
  return torch.where(up, lower + 1, lower)


def fsq(pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Finite scalar quantization (Mentzer et al. 2023) with seven levels:
  round(3 tanh(e)) with a straight-through gradient.

  Returns (decoder latent codes / 3, integer codes, bounded pre-round values).
  Rounds half to even, exactly like ``torch.round``.
  """
  bounded = FSQ_HALF_LEVELS * torch.tanh(pre)
  codes = round_half_even(bounded)
  latent = (bounded + (codes - bounded).detach()) / FSQ_HALF_LEVELS
  return latent, codes, bounded


class Tracker(nn.Module):
  """Reference encoder -> 16x7 FSQ -> sensor-history-conditioned decoder."""

  def __init__(self, encoder_dim: int, history_dim: int, action_dim: int):
    super().__init__()
    widths = (2048, 1024, 512, 256)
    self.encoder_norm = Normalizer(encoder_dim)
    self.history_norm = Normalizer(history_dim)
    self.encoder = mlp(encoder_dim, widths, FSQ_DIM, nn.ELU)
    self.decoder = mlp(history_dim + FSQ_DIM, widths, action_dim, nn.ELU)
    with torch.no_grad():  # start near the default pose
      self.decoder[-1].weight.mul_(0.01)
      self.decoder[-1].bias.zero_()

  def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fsq(self.encoder(self.encoder_norm(features)))

  def decode(self, latent: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    return self.decoder(torch.cat((self.history_norm(history), latent), -1))

  def forward(self, features: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    return self.decode(self.encode(features)[0], history)


class WorldModel(nn.Module):
  """Deterministic velocity-delta dynamics with fixed normalization.

  ``dynamics`` maps normalized (local state, executed action, normalized
  context) to normalized velocity deltas. Full-state error normalization
  retains pose supervision after analytic integration; termination is analytic.
  """

  def __init__(
    self,
    local_dim: int,
    action_dim: int,
    context_dim: int,
    increment_dim: int,
    state_error_dim: int,
  ):
    super().__init__()
    self.local_norm = Normalizer(local_dim)
    self.increment_norm = Normalizer(increment_dim)
    self.error_norm = Normalizer(state_error_dim)
    self.dynamics = mlp(
      local_dim + action_dim + context_dim,
      (2048, 2048, 2048),
      increment_dim,
      nn.SiLU,
      layer_norm=True,
    )

  def increment(self, local: torch.Tensor, action: torch.Tensor, context: torch.Tensor):
    out = self.dynamics(torch.cat((self.local_norm(local), action, context), -1))
    return out * self.increment_norm.std + self.increment_norm.mean


class Critic(nn.Module):
  VALUE_SCALE = 100.0  # returns reach 5 / (1 - 0.99) = 500

  def __init__(self, dim: int):
    super().__init__()
    self.norm = Normalizer(dim)
    self.net = mlp(dim, (256, 128), 1, nn.ELU)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.VALUE_SCALE * self.net(self.norm(x))[..., 0]
