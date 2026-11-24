"""Muon on MLP hidden matrices, Adam on the remaining trainable parameters.

Adapted from BlinkMimic's ``src/blinkmimic/optim.py`` at
https://github.com/fuyichen0725/blinkmimic/tree/d264658d6c9ce61fb52f9f0f8985a369f3ccc409
without its optional step gating or checks. Muon uses torch's
``match_rms_adamw`` scaling so it shares Adam's learning rate.
"""

# MIT License
# Copyright (c) 2026 Fuyi Chen
# Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import torch
from torch import nn


class MuonAdam:
  """Split ``model`` parameters by the Linear layers in each supplied MLP.

  All but each MLP's first and last Linear weights use Muon (zero weight
  decay); remaining parameters requiring gradients use Adam. As upstream,
  both partitions must be nonempty. ``betas`` and ``fused`` configure Adam;
  Muon retains torch's defaults apart from learning rate, decay and scaling.

  With ``capturable=True``, each learning rate is a scalar device tensor:
  change it with ``group["lr"].fill_(value)``. Adam is always fused and
  capturable in this mode. The Muon update below avoids torch Muon's host
  read of its learning rate, allowing changes during CUDA graph replay.
  """

  def __init__(
    self,
    model: nn.Module,
    mlps: list[nn.Module],
    lr: float,
    betas: tuple[float, float] = (0.9, 0.95),
    fused: bool = True,
    capturable: bool = False,
  ):
    hidden = {
      id(layer.weight)
      for mlp in mlps
      for layer in [m for m in mlp.modules() if isinstance(m, nn.Linear)][1:-1]
    }
    muon = [p for p in model.parameters() if id(p) in hidden]
    adam = [p for p in model.parameters() if id(p) not in hidden and p.requires_grad]
    self.capturable = capturable
    if capturable:
      device = muon[0].device
      self.muon = torch.optim.Muon(
        muon,
        lr=torch.tensor(float(lr), device=device),
        weight_decay=0.0,
        adjust_lr_fn="match_rms_adamw",
      )
      self.adam = torch.optim.Adam(
        adam,
        lr=torch.tensor(float(lr), device=device),
        betas=betas,
        fused=True,
        capturable=True,
      )
    else:
      self.muon = torch.optim.Muon(muon, lr=lr, weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
      self.adam = torch.optim.Adam(adam, lr=lr, betas=betas, fused=fused)

  @property
  def param_groups(self):
    # Loading optimizer state replaces group dicts; do not cache this list.
    return self.muon.param_groups + self.adam.param_groups

  def zero_grad(self, set_to_none: bool = True) -> None:
    self.muon.zero_grad(set_to_none=set_to_none)
    self.adam.zero_grad(set_to_none=set_to_none)

  @torch.no_grad()
  def _muon_step_tensor_lr(self) -> None:
    from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

    for group in self.muon.param_groups:
      assert group["weight_decay"] == 0.0
      for param in group["params"]:
        if param.grad is None:
          continue
        state = self.muon.state[param]
        if "momentum_buffer" not in state:
          state["momentum_buffer"] = torch.zeros_like(
            param.grad, memory_format=torch.preserve_format
          )
        buf = state["momentum_buffer"]
        buf.lerp_(param.grad, 1 - group["momentum"])
        update = param.grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
        update = _zeropower_via_newtonschulz(
          update, group["ns_coefficients"], group["ns_steps"], group["eps"]
        )
        ratio = _adjust_lr(1.0, group["adjust_lr_fn"], param.shape)
        # Convert before multiplication: bf16 times a 0-dim fp32 tensor stays bf16.
        delta = update.to(param.dtype) * (group["lr"] * -ratio)
        param.add_(delta)

  @torch.no_grad()
  def step(self) -> None:
    if self.capturable:
      self._muon_step_tensor_lr()
    else:
      self.muon.step()
    self.adam.step()

  def state_dict(self) -> dict:
    return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

  def load_state_dict(self, sd: dict) -> None:
    lrs = [group["lr"] for group in self.param_groups]
    self.muon.load_state_dict(sd["muon"])
    self.adam.load_state_dict(sd["adam"])
    if self.capturable:
      # Keep tensor identities used by the training loop and captured graphs.
      for group, lr in zip(self.param_groups, lrs, strict=True):
        lr.fill_(float(group["lr"]))
        group["lr"] = lr
