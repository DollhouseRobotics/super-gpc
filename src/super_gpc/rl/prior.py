"""Stage 2 observation-conditioned autoregressive prior over grouped FSQ codes.

Our decisions, with original-GPC code defaults for details they leave open
(spec "Stage 2 prior: accepted decisions"):

- The 16 seven-level FSQ codes of the frozen tracker are packed into 4 tokens
  of 4 consecutive scalars each, little-endian mixed radix over the level
  index ``code + 3`` (GPC-code ``FSQTokenization``), vocabulary 7**4 = 2401.
- The context is ONE current 87-scalar sensor frame, normalized with running
  mean/variance (statistics EMA 0.999, clamp +-5), then linear-GELU-linear to
  the model width. There is no past-step token context: autoregression runs
  only over the 4 tokens of the current control step.
- Sequence [context, token0, token1, token2] plus learned positions through a
  post-LayerNorm causal transformer encoder (GELU, dropout 0.1 in training);
  position k predicts token k.
- Generation: temperature-scaled softmax, nucleus top-p, renormalize, draw a
  discrete token, token by token.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from super_gpc.rl.networks import FSQ_DIM, FSQ_HALF_LEVELS

LEVELS = 2 * FSQ_HALF_LEVELS + 1  # 7
GROUP = 4  # FSQ scalars per token
NUM_TOKENS = FSQ_DIM // GROUP  # 4
VOCAB = LEVELS**GROUP  # 2401
D_MODEL, FF_SIZE = 1024, 4096  # model and feed-forward width (spec, original GPC Table 10)


def codes_to_tokens(codes: torch.Tensor) -> torch.Tensor:
  """(..., 16) integer FSQ codes in -3..3 -> (..., 4) int64 tokens in 0..2400.

  Scalar ``4 g + i`` contributes ``(code + 3) * 7**i`` to token ``g``.
  """
  idx = (codes.round().long() + FSQ_HALF_LEVELS).reshape(*codes.shape[:-1], NUM_TOKENS, GROUP)
  basis = LEVELS ** torch.arange(GROUP, device=codes.device)
  return (idx * basis).sum(-1)


def tokens_to_codes(tokens: torch.Tensor) -> torch.Tensor:
  """Inverse of ``codes_to_tokens``: (..., 4) tokens -> (..., 16) float codes.

  Uses float Floor arithmetic (exact below 2**24) so the same function exports
  to ONNX without integer division.
  """
  t = tokens.float()[..., None]
  basis = (LEVELS ** torch.arange(GROUP + 1, device=tokens.device)).float()
  digits = torch.floor(t / basis[:GROUP]) - LEVELS * torch.floor(t / basis[1:])
  return (digits - FSQ_HALF_LEVELS).reshape(*tokens.shape[:-1], FSQ_DIM)


class RunningObsNorm(nn.Module):
  """GPC-code ``RunningMeanStd`` in EMA mode: statistics EMA, not weight EMA.

  ``update`` is called on each training minibatch before it is normalized;
  evaluation and generation never update. Statistics are float64 buffers and
  part of the native checkpoint, so they resume exactly.
  """

  def __init__(self, dim: int, decay: float = 0.999, clamp: float = 5.0, eps: float = 1e-5):
    super().__init__()
    self.decay, self.clamp, self.eps = decay, clamp, eps
    self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
    self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
    self.register_buffer("updates", torch.zeros((), dtype=torch.long))

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    """One EMA step from a minibatch. Under data parallelism the batch moments
    are those of the union of all ranks' minibatches (all-reduced), so every
    rank keeps identical statistics."""
    x = x.reshape(-1, x.shape[-1]).double()
    d = self.decay
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
      moments = torch.stack((x.sum(0), x.square().sum(0), torch.full_like(x[0], len(x))))
      dist.all_reduce(moments)
      mean = moments[0] / moments[2]
      var = (moments[1] / moments[2] - mean.square()).clamp_min(0.0)
    else:
      mean, var = x.mean(0), x.var(0, unbiased=False)
    self.mean.mul_(d).add_((1 - d) * mean)
    self.var.mul_(d).add_((1 - d) * var)
    self.updates += 1

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = (x - self.mean.float()) / torch.sqrt(self.var.float() + self.eps)
    return y.clamp(-self.clamp, self.clamp)


class CodePrior(nn.Module):
  def __init__(
    self,
    obs_dim: int,
    d_model: int,
    num_layers: int = 3,
    num_heads: int = 8,
    dropout: float = 0.1,
    norm_decay: float = 0.999,
    norm_clamp: float = 5.0,
  ):
    super().__init__()
    if d_model != D_MODEL:
      raise ValueError(f"the Stage 2 prior has width {D_MODEL}, not {d_model}")
    self.obs_norm = RunningObsNorm(obs_dim, norm_decay, norm_clamp)
    self.obs_encoder = nn.Sequential(
      nn.Linear(obs_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
    )
    # GPC projects one-hot tokens with a biased linear layer; index its columns.
    self.token_embed = nn.Linear(VOCAB, d_model)
    self.pos = nn.Parameter(torch.randn(1, NUM_TOKENS, d_model))
    layer = nn.TransformerEncoderLayer(
      d_model,
      num_heads,
      FF_SIZE,
      dropout,
      activation="gelu",
      batch_first=True,
      norm_first=False,  # post-LN, as the source TransformerEncoderLayer
    )
    self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
    self.head = nn.Linear(d_model, VOCAB)
    mask = torch.triu(torch.full((NUM_TOKENS, NUM_TOKENS), float("-inf")), diagonal=1)
    self.register_buffer("causal_mask", mask, persistent=False)

  def forward(self, obs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return self.logits(obs, tokens)

  def logits(self, obs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Causal logits (B, 4, 2401). Position k sees the observation and
    ``tokens[:, :k]``; ``tokens[:, 3]`` (if given) is never read."""
    ctx = self.obs_encoder(self.obs_norm(obs))
    emb = self.token_embed.weight.t()[tokens[:, : NUM_TOKENS - 1]] + self.token_embed.bias
    seq = torch.cat((ctx[:, None], emb), 1) + self.pos
    # is_causal=False with the explicit mask avoids the causal-mask probe (ONNX Trilu).
    return self.head(self.transformer(seq, mask=self.causal_mask, is_causal=False))

  def generate(
    self,
    obs: torch.Tensor,
    uniform: torch.Tensor,
    top_p: float,
    temperature: float,
    export: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the 4 tokens of one control step with ``uniform`` (B, 4) draws.

    Unrolled, with functional token updates so the same code exports to ONNX.
    Returns (tokens, per-position logits). Call in eval mode.
    """
    if self.training:
      raise RuntimeError("generate in eval mode (dropout disabled, statistics fixed)")
    tokens = torch.zeros(obs.shape[0], NUM_TOKENS, dtype=torch.long, device=obs.device)
    slots = torch.arange(NUM_TOKENS, device=obs.device)
    out = []
    for k in range(NUM_TOKENS):
      lg = self.logits(obs, tokens)[:, k]
      tok = nucleus_sample(lg, uniform[:, k], top_p, temperature, export)
      tokens = torch.where(slots == k, tok[:, None], tokens)
      out.append(lg)
    return tokens, torch.stack(out, 1)


def nucleus_keep(probs: torch.Tensor, top_p: float, export: bool = False) -> torch.Tensor:
  """GPC-code top-p keep mask in vocabulary order: keep a token when the
  probability mass ranked strictly before it is at most ``top_p`` (the most
  likely token is always kept). Ties rank the lower index first.

  ``export`` ranks with ``TopK`` over all but one token instead of a full sort,
  which TensorRT cannot compile. That is exact for top_p < 1 - 1/vocab: the
  lowest-ranked token has probability at most 1/vocab, so the mass before it
  always exceeds top_p. ONNX ``TopK`` also breaks exact ties by lower index
  (checked against ONNX Runtime); eager ``torch.topk`` may not, so the export
  path is only for the ONNX graph.
  """
  n = probs.shape[-1]
  if export:
    if top_p >= 1 - 1 / n:
      raise ValueError("TopK export is exact only for top_p < 1 - 1/vocab")
    sp, idx = torch.topk(probs, n - 1, dim=-1)
  else:
    sp, idx = torch.sort(probs, dim=-1, descending=True, stable=True)
  cum = torch.cumsum(sp, -1)
  before = torch.cat((torch.zeros_like(cum[..., :1]), cum[..., :-1]), -1)
  return torch.zeros_like(probs, dtype=torch.bool).scatter(-1, idx, before <= top_p)


def nucleus_sample(
  logits: torch.Tensor,
  uniform: torch.Tensor,
  top_p: float,
  temperature: float,
  export: bool = False,
) -> torch.Tensor:
  """Discrete draw from the temperature-scaled, renormalized top-p nucleus.

  Same keep rule and distribution as GPC-code ``nucleus_sampling``. The draw
  inverts the kept CDF in vocabulary order with the explicit ``uniform`` in
  [0, 1), so the Miniverse controller owns and reproduces the randomness.
  """
  if temperature <= 0:
    raise ValueError("temperature must be positive")
  probs = F.softmax(logits.float() / temperature, -1)
  keep = nucleus_keep(probs, top_p, export)
  kept = probs * keep
  cdf = torch.cumsum(kept, -1) / kept.sum(-1, keepdim=True)
  pos = (cdf <= uniform.float()[..., None]).sum(-1)
  last = (keep * torch.arange(probs.shape[-1], device=probs.device)).amax(-1)
  return torch.minimum(pos, last)
