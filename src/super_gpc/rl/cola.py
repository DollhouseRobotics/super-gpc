"""CoLA: task-conditioned DoRA adapters on the frozen Stage 2 code prior.

The FiLM task input is the 4-D joystick command for steering, or another
task's ``task_dim`` observation (get-down). Joystick RLFT follows the G1 gpc
steering implementation (spec "Joystick RLFT"): every layer of the frozen
prior's causal transformer gets a DoRA residual whose rank-space activations
are FiLM-modulated by the task input,
``layer(x) + m * normalize(scale * (FiLM(x A, c)) B)`` (upstream ProtoMotions
``TransformerLayerWithDoRA``, reimplemented here). ``m`` starts
at zero, so the adapted policy is exactly the base prior at initialization
and the first update learns ``m``.

The prior's observation encoder, token embedding, positions, head and
observation statistics, and the frozen Stage 1 decoder, never change.
Tokens are drawn from the full-support temperature-1 categorical (not the
Stage 2 top-p nucleus), so PPO likelihood ratios are exact. Each draw inverts
the vocabulary-order CDF with an explicit uniform, so the Miniverse
controller owns the randomness and the export reproduces training draws.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from super_gpc.rl import prior as P
from super_gpc.rl.networks import FSQ_HALF_LEVELS, Tracker

COMMAND_DIM = 4  # local commanded velocity (x, y) and commanded facing (cos, sin)


class DoRA(nn.Module):
  def __init__(self, d_model: int, c_dim: int, rank: int, alpha: float):
    super().__init__()
    self.A = nn.Parameter(torch.randn(d_model, rank) / math.sqrt(d_model))
    self.B = nn.Parameter(torch.randn(rank, d_model) * 1e-4)
    self.m = nn.Parameter(torch.zeros(1, d_model))  # exact base equivalence at start
    self.scaling = alpha / rank
    self.gamma = nn.Linear(c_dim, rank)
    self.beta = nn.Linear(c_dim, rank)
    nn.init.zeros_(self.gamma.weight)
    nn.init.ones_(self.gamma.bias)
    nn.init.zeros_(self.beta.weight)
    nn.init.zeros_(self.beta.bias)

  def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    xa = (x @ self.A) * self.gamma(c)[:, None] + self.beta(c)[:, None]
    out = self.scaling * (xa @ self.B)
    return self.m * (out / (out.norm(dim=-1, keepdim=True) + 1e-6))


def categorical_draw(logits: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
  """Full-support temperature-1 draw by inverting the vocabulary-order CDF."""
  probs = torch.softmax(logits.float(), -1)
  cdf = torch.cumsum(probs, -1)
  pos = (cdf <= uniform.float()[..., None] * cdf[..., -1:]).sum(-1)
  return pos.clamp(max=probs.shape[-1] - 1)


def categorical_statistics(logits, tokens, base_logp, entropy_grad: bool = True):
  """Per-token selected log-probabilities and prefix-summed entropy and
  KL(adapted || base), from the adapted logits and the frozen base's
  log-probabilities. Conditionals are exact; the sum of conditional KLs at
  prefixes drawn from the adapted policy is an unbiased estimate of the
  sequence KL (G1 gpc ``categorical_statistics``). Without ``entropy_grad``
  the entropy is a statistic only, which skips its full-vocabulary backward."""
  lp = logits.float().log_softmax(-1)
  p = lp.exp()
  selected = lp.gather(-1, tokens[..., None]).squeeze(-1)
  with torch.set_grad_enabled(entropy_grad and torch.is_grad_enabled()):
    entropy = -(p * lp).sum(-1).sum(-1)
  kl = (p * (lp - base_logp.detach())).sum(-1).sum(-1)
  return selected, entropy, kl


@torch.library.custom_op("super_gpc::frozen_bf16_linear", mutates_args=())
def frozen_bf16_linear(
  x: torch.Tensor, w_t: torch.Tensor, b: torch.Tensor, bf16_out: bool
) -> torch.Tensor:
  """``x @ w_t + b`` from BF16 operands with FP32 accumulation: FP32 output,
  or BF16 with a BF16 ``b``. The weights are frozen; backward returns only the
  input gradient. A custom op so ``torch.compile`` keeps it as one extern GEMM
  (Inductor cannot lower ``addmm`` with ``out_dtype``)."""
  x16 = x.bfloat16()
  if bf16_out:
    return torch.addmm(b, x16, w_t)
  return torch.addmm(b, x16, w_t, out_dtype=torch.float32)


@frozen_bf16_linear.register_fake
def _(x, w_t, b, bf16_out):
  return x.new_empty(x.shape[0], w_t.shape[1], dtype=torch.bfloat16 if bf16_out else torch.float32)


@torch.library.custom_op("super_gpc::frozen_bf16_linear_input_grad", mutates_args=())
def _input_grad(grad: torch.Tensor, w_t: torch.Tensor) -> torch.Tensor:
  return torch.mm(grad.bfloat16(), w_t.t(), out_dtype=torch.float32)


@_input_grad.register_fake
def _(grad, w_t):
  return grad.new_empty(grad.shape[0], w_t.shape[0], dtype=torch.float32)


def _setup_context(ctx, inputs, output):
  x, w_t, _, _ = inputs
  ctx.save_for_backward(w_t)
  ctx.dtype = x.dtype


def _backward(ctx, grad):
  (w_t,) = ctx.saved_tensors
  return _input_grad(grad, w_t).to(ctx.dtype), None, None, None


frozen_bf16_linear.register_autograd(_backward, setup_context=_setup_context)


class ColaPolicy(nn.Module):
  """Frozen prior + frozen decoder + trainable command adapters."""

  def __init__(
    self,
    prior: P.CodePrior,
    tracker: Tracker,
    action_clip: float,
    rank: int = 32,
    alpha: float = 64.0,
    task_dim: int = COMMAND_DIM,
  ):
    super().__init__()
    self.prior = prior.eval().requires_grad_(False)
    self.tracker = tracker.eval().requires_grad_(False)
    self.action_clip = action_clip
    d = prior.head.in_features
    self.adapters = nn.ModuleList(
      DoRA(d, task_dim, rank, alpha) for _ in self.prior.transformer.layers
    )
    self._w16: dict = {}  # frozen Linear -> BF16 (weight, FP32 bias); empty: FP32

  def set_bf16(self, enabled: bool) -> None:
    """Training option: evaluate the frozen transformer and head Linears from
    BF16 weight copies with FP32 accumulation and FP32 outputs (the
    feed-forward hidden layer stays BF16). LayerNorm, attention scores,
    residuals, embeddings, adapters and the decoder stay FP32. Call after
    moving to the device; exports and checks build their own FP32 policy."""
    self._w16 = {}
    if not enabled:
      return

    def put(key, w, b):
      b = b.detach().float()
      self._w16[key] = (w.detach().bfloat16().t(), b, b.bfloat16())

    for layer in self.prior.transformer.layers:
      sa = layer.self_attn
      put(sa, sa.in_proj_weight, sa.in_proj_bias)
      for m in (sa.out_proj, layer.linear1, layer.linear2):
        put(m, m.weight, m.bias)
    put(self.prior.head, self.prior.head.weight, self.prior.head.bias)

  def __getstate__(self):
    # Copies (exports, checks) and pickles evaluate the FP32 prior.
    return super().__getstate__() | {"_w16": {}}

  def _linear(self, key, x, w, b, bf16_out: bool = False):
    p = self._w16.get(key)
    if p is None:
      return F.linear(x, w, b)
    w_t, b32, b16 = p
    y = frozen_bf16_linear(x.reshape(-1, x.shape[-1]), w_t, b16 if bf16_out else b32, bf16_out)
    return y.reshape(*x.shape[:-1], -1)

  def train(self, mode: bool = True):
    super().train(mode)
    self.prior.eval()  # no dropout; statistics fixed
    self.tracker.eval()
    return self

  def adapter_parameters(self) -> list[nn.Parameter]:
    return list(self.adapters.parameters())

  def _block(self, layer: nn.TransformerEncoderLayer, x: torch.Tensor, past=None):
    """Post-LN ``TransformerEncoderLayer`` (eval) on new positions ``x`` (B, L, d)
    after ``past`` cached (keys, values), with causal attention.

    Same math as the PyTorch layer, but attention over at most four positions
    is a plain matmul/softmax: PyTorch's fused FP32 attention kernels spent
    over half of the rollout GPU time on these tiny sequences.
    """
    sa = layer.self_attn
    B, L, d = x.shape
    h = sa.num_heads
    q, k, v = self._linear(sa, x, sa.in_proj_weight, sa.in_proj_bias).chunk(3, -1)
    if past is not None:
      k, v = torch.cat((past[0], k), 1), torch.cat((past[1], v), 1)
    S = k.shape[1]

    def heads(t):
      return t.reshape(B, t.shape[1], h, d // h).transpose(1, 2)

    scores = heads(q) @ heads(k).transpose(-1, -2) * (d // h) ** -0.5
    if L > 1:
      causal = torch.ones(L, S, dtype=torch.bool, device=x.device).tril(S - L)
      scores = scores.masked_fill(~causal, float("-inf"))
    a = (scores.softmax(-1) @ heads(v)).transpose(1, 2).reshape(B, L, d)
    out, l1, l2 = sa.out_proj, layer.linear1, layer.linear2
    y = layer.norm1(x + self._linear(out, a, out.weight, out.bias))
    f = layer.activation(self._linear(l1, y, l1.weight, l1.bias, bf16_out=True))
    return layer.norm2(y + self._linear(l2, f, l2.weight, l2.bias)), (k, v)

  def _head(self, x: torch.Tensor) -> torch.Tensor:
    head = self.prior.head
    return self._linear(head, x, head.weight, head.bias)

  def _hidden(self, seq, command, base: bool, layer0=None) -> torch.Tensor:
    x = seq
    for i, (layer, adapter) in enumerate(
      zip(self.prior.transformer.layers, self.adapters, strict=True)
    ):
      y = layer0 if i == 0 and layer0 is not None else self._block(layer, x)[0]
      x = y if base else y + adapter(x, command)
    return x

  def _sequence(self, obs: torch.Tensor, tokens: torch.Tensor, n: int) -> torch.Tensor:
    pr = self.prior
    ctx = pr.obs_encoder(pr.obs_norm(obs))[:, None]
    if n > 1:
      emb = pr.token_embed.weight.t()[tokens[:, : n - 1]] + pr.token_embed.bias
      ctx = torch.cat((ctx, emb), 1)
    return ctx + pr.pos[:, :n]

  def logits(
    self,
    obs: torch.Tensor,
    command: torch.Tensor,
    tokens: torch.Tensor,
    base: bool = False,
    layer0: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Teacher-forced causal logits (B, 4, V); position k sees tokens[:, :k].
    ``layer0`` is the frozen first block's output from ``rollout_step``: it
    depends only on frozen weights and the inputs, so PPO need not recompute it."""
    seq = self._sequence(obs, tokens, P.NUM_TOKENS)
    return self._head(self._hidden(seq, command, base, layer0))

  @torch.no_grad()
  def rollout_step(self, obs: torch.Tensor, command: torch.Tensor, uniform: torch.Tensor):
    """``sample`` for training, with the frozen base evaluated in the same
    key/value-cached pass. Adapters add after each block, so both policies
    share the first block and its cache; later blocks run the adapted and
    base rows as one batch. Returns (tokens, adapted logits, base logits at
    the sampled prefix, first-block outputs (B, 4, d))."""
    pr = self.prior
    B = len(obs)
    layers = list(pr.transformer.layers)
    cache: list = [None] * len(layers)
    x = pr.obs_encoder(pr.obs_norm(obs))[:, None]
    toks, out, first = [], [], []
    for k in range(P.NUM_TOKENS):
      if k:
        x = (pr.token_embed.weight.t()[toks[-1]] + pr.token_embed.bias)[:, None]
      h = x + pr.pos[:, k : k + 1]
      for i, (layer, adapter) in enumerate(zip(layers, self.adapters, strict=True)):
        y, cache[i] = self._block(layer, h, cache[i])
        if i == 0:
          first.append(y)
          y = y.repeat(2, 1, 1)
        y[:B] += adapter(h[:B], command)
        h = y
      lg = self._head(h[:, 0])
      toks.append(categorical_draw(lg[:B], uniform[:, k]))
      out.append(lg)
    lg = torch.stack(out, 1)
    return torch.stack(toks, 1), lg[:B], lg[B:], torch.cat(first, 1)

  def sample(self, obs: torch.Tensor, command: torch.Tensor, uniform: torch.Tensor, base=False):
    """Draw the 4 tokens of one step with a key/value cache. Returns (tokens,
    logits (B, 4, V) at the sampled prefix), equal to the teacher-forced
    ``logits`` up to floating-point summation order."""
    pr = self.prior
    layers = list(pr.transformer.layers)
    cache: list = [None] * len(layers)
    x = pr.obs_encoder(pr.obs_norm(obs))[:, None]
    toks, out = [], []
    for k in range(P.NUM_TOKENS):  # unrolled; exports to ONNX
      if k:
        x = (pr.token_embed.weight.t()[toks[-1]] + pr.token_embed.bias)[:, None]
      x = x + pr.pos[:, k : k + 1]
      for i, (layer, adapter) in enumerate(zip(layers, self.adapters, strict=True)):
        y, cache[i] = self._block(layer, x, cache[i])
        x = y if base else y + adapter(x, command)
      lg = self._head(x[:, 0])
      toks.append(categorical_draw(lg, uniform[:, k]))
      out.append(lg)
    return torch.stack(toks, 1), torch.stack(out, 1)

  def decode(self, tokens: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    """Normalized executed action from the frozen decoder."""
    codes = P.tokens_to_codes(tokens)
    action = self.tracker.decode(codes / FSQ_HALF_LEVELS, history)
    return action.clamp(-self.action_clip, self.action_clip)
