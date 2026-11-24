"""Adversarial motion prior (AMP) discriminator and style reward.

Peng et al. 2021 (AMP, Section 5 and Appendix): a discriminator over
state transitions ``(s, s')`` is trained with the least-squares GAN
objective to output +1 on reference-motion transitions and -1 on policy
transitions, with a gradient penalty on the reference samples. The style
reward is ``max(0, 1 - 0.25 (D(s, s') - 1)^2)``. Inputs are standardized by
running statistics of both reference and policy transitions.
"""

from __future__ import annotations

import torch
from torch import nn


class RunningNorm(nn.Module):
  """Mean and variance over every sample seen (batched parallel update)."""

  def __init__(self, dim: int, clip: float = 5.0):
    super().__init__()
    self.clip = clip
    self.register_buffer("count", torch.zeros((), dtype=torch.float64))
    self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
    self.register_buffer("m2", torch.zeros(dim, dtype=torch.float64))

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    x = x.double()
    n, mean = x.shape[0], x.mean(0)
    m2 = (x - mean).square().sum(0)
    total = self.count + n
    delta = mean - self.mean
    self.m2 += m2 + delta.square() * self.count * n / total
    self.mean += delta * n / total
    self.count.copy_(total)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    std = (self.m2 / self.count.clamp_min(1)).sqrt().clamp_min(1e-2)
    return ((x - self.mean.float()) / std.float()).clamp(-self.clip, self.clip)


class AmpDiscriminator(nn.Module):
  def __init__(self, input_dim: int, hidden: tuple[int, ...]):
    super().__init__()
    self.norm = RunningNorm(input_dim)
    layers, width = [], input_dim
    for h in hidden:
      layers += [nn.Linear(width, h), nn.ReLU()]
      width = h
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(width, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(self.norm(x))).squeeze(-1)

  @torch.no_grad()
  def style_reward(self, x: torch.Tensor) -> torch.Tensor:
    return (1 - 0.25 * (self(x) - 1).square()).clamp_min(0)

  def loss(
    self, policy: torch.Tensor, expert: torch.Tensor, grad_penalty: float, logit_reg: float
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Least-squares GAN loss with the reference-sample gradient penalty and
    a weight penalty on the output layer."""
    expert = expert.detach().requires_grad_(True)
    d_expert, d_policy = self(expert), self(policy)
    lsgan = 0.5 * ((d_expert - 1).square().mean() + (d_policy + 1).square().mean())
    (grad,) = torch.autograd.grad(d_expert.sum(), expert, create_graph=True)
    gp = grad.square().sum(-1).mean()
    reg = self.head.weight.square().sum()
    loss = lsgan + 0.5 * grad_penalty * gp + logit_reg * reg
    stats = {
      "disc/lsgan_loss": lsgan.detach(),
      "disc/grad_penalty": gp.detach(),
      "disc/expert_logit": d_expert.mean().detach(),
      "disc/policy_logit": d_policy.mean().detach(),
      "disc/accuracy": 0.5
      * ((d_expert > 0).float().mean() + (d_policy < 0).float().mean()).detach(),
    }
    return loss, stats
