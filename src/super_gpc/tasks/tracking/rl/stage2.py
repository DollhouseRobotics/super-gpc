"""Stage 2: supervised code prior over the frozen teacher's FSQ codes.

Each iteration (GPC-code supervised expert-rollout loop):

1. Collect ``rollout_steps`` (64) fresh real-simulator vector steps with the
   frozen teacher: its encoder labels the aligned reference window at the
   same state and time as the prior observation, its decoder drives the robot
   from those codes and the ten-frame sensor history, plus Gaussian action
   noise (sigma 1e-4). Episodes continue across iterations.
2. One shuffled pass of minibatches over the (current 87-D sensor frame,
   4 packed tokens) pairs: update the observation statistics, teacher-forced
   cross-entropy with label smoothing 0.01, AdamW, global-norm clip 100.

Only the prior's observation encoder and transformer are optimized; the
teacher's encoder, decoder and normalizers are frozen. There is no persistent
replay, world model, imagination, critic or PPO loss.
"""

from __future__ import annotations

import hashlib
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from super_gpc.rl import prior as P
from super_gpc.rl.networks import FSQ_HALF_LEVELS, Tracker, tracker_weights
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.mdp import rewards as R
from super_gpc.tasks.tracking.mdp.commands import ReturnSampler
from super_gpc.tasks.tracking.mdp.motion_library import POLICY_HZ, MotionLibrary
from super_gpc.tasks.tracking.rl import stage2_quality as Q
from super_gpc.tasks.tracking.rl.real_env import RealEnv
from super_gpc.tasks.tracking.rl.stage1 import build_kinematics, evaluation_clips
from super_gpc.tasks.tracking.rl.stage2_cfg import Stage2Cfg

SUM_KEYS = ("episodes/finished", "episodes/failures", "episodes/timeouts", "episodes/clip_ends")


def file_sha256(path: Path) -> str:
  with Path(path).open("rb") as f:
    return hashlib.file_digest(f, "sha256").hexdigest()


def tensor_digest(tensors) -> str:
  h = hashlib.sha256()
  for t in tensors:
    h.update(t.detach().contiguous().cpu().numpy().tobytes())
  return h.hexdigest()[:16]


def load_teacher(path: Path) -> tuple[Tracker, dict]:
  """Frozen teacher from a Stage 1 ``policy_state()`` file.

  ``tracker`` is a ``Tracker(690, 870, 27)`` state dict with its fixed
  normalizers and ``cfg`` holds ``action_clip`` and ``reference_offsets``.
  """
  state = torch.load(path, map_location="cpu", weights_only=False)
  tracker = Tracker(O.ENCODER_DIM, O.HISTORY_DIM, O.ACTION_DIM)
  tracker.load_state_dict(tracker_weights(state))
  tracker.eval().requires_grad_(False)
  info = {
    "path": str(path),
    "sha256": file_sha256(path),
    "stage": str(state.get("stage", "1")),
    "iteration": int(state.get("iteration", -1)),
    "action_clip": float(state["cfg"]["action_clip"]),
    "reference_offsets": tuple(state["cfg"].get("reference_offsets", range(1, 11))),
    "tracker_digest": tensor_digest(tracker.state_dict().values()),
  }
  if "smoke" in state:
    info["smoke"] = state["smoke"]
  return tracker, info


def training_fingerprint(
  env: RealEnv | None, sampler: ReturnSampler | None, device: str, **extra
) -> dict:
  """Digests of everything a side rollout must leave untouched. Without a
  training environment (evaluation-only trainers) only the prior,
  optimizer, teacher and RNG are covered."""
  tensors = {"rng_cpu": torch.get_rng_state()}
  if device.startswith("cuda"):
    tensors["rng_cuda"] = torch.cuda.get_rng_state(device)
  out = {k: tensor_digest(v) for k, v in extra.items()}
  if env is None:
    return out | {k: tensor_digest([v]) for k, v in tensors.items()}
  sim = env.env.sim
  tensors |= {
    "qpos": sim.data.qpos,
    "qvel": sim.data.qvel,
    "ctrl": sim.data.ctrl,
    "dr_fields": torch.cat(
      [getattr(sim.model, f).reshape(env.num_envs, -1) for f in env.phys.model_fields[:8]], -1
    ),
    "encoder_bias": env.robot.data.encoder_bias,
    "context": env.context,
    "motion_clip_frame": torch.stack(
      (env.cmd.clip, env.cmd.frame, env.cmd.start_bin, env.cmd.start_frame)
    ),
    "mjlab_episode_length": env.env.episode_length_buf,
    "history": env.history,
    "prev_action": env.prev_action,
    "episode_counters": torch.stack(
      (env.ep_len.float(), env.ep_return, env.ep_disc_return, env.episode.float())
    ),
    "sampler": torch.cat([v.float() for v in sampler.state_dict().values()]),
  }
  out |= {k: tensor_digest([v]) for k, v in tensors.items()}
  return out | {"next_episode_id": str(env.next_episode_id)}


@torch.no_grad()
def code_stats(codes: torch.Tensor, tokens: torch.Tensor, prefix: str) -> dict[str, float]:
  """FSQ level usage and token statistics over a batch of (N, 16) codes."""
  levels = torch.arange(-FSQ_HALF_LEVELS, FSQ_HALF_LEVELS + 1, device=codes.device)
  hist = (codes[..., None] == levels).float().mean(0)
  ent = -(hist * (hist + 1e-12).log()).sum(-1) / math.log(P.LEVELS)
  out = {
    f"{prefix}/level_entropy_ratio": float(ent.mean()),
    f"{prefix}/levels_used_mean": float((hist > 0).float().sum(-1).mean()),
    f"{prefix}/extreme_code_fraction": float((codes.abs() == FSQ_HALF_LEVELS).float().mean()),
    f"{prefix}/unique_step_code_fraction": torch.unique(tokens, dim=0).shape[0] / len(tokens),
  }
  for k in range(P.NUM_TOKENS):
    counts = torch.bincount(tokens[:, k], minlength=P.VOCAB).float()
    p = counts / counts.sum()
    out[f"{prefix}/token{k}_entropy_ratio"] = float(
      -(p * (p + 1e-12).log()).sum() / math.log(P.VOCAB)
    )
    out[f"{prefix}/token{k}_used"] = float((counts > 0).sum())
  return out


@contextmanager
def tf32(enabled: bool):
  """TF32 matmuls/convolutions for the enclosed prior optimization only.

  The frozen teacher, simulator-side torch code, evaluation and export stay
  FP32: the flag is restored on exit.
  """
  m, c = torch.backends.cuda.matmul, torch.backends.cudnn
  prev = (m.allow_tf32, c.allow_tf32)
  m.allow_tf32 = c.allow_tf32 = enabled
  try:
    yield
  finally:
    m.allow_tf32, c.allow_tf32 = prev


class BF16Linear(torch.nn.Linear):
  """Linear whose training forward (and so backward) runs in BF16 and returns
  FP32. Weights, optimizer state and checkpoints stay FP32; eval mode is the
  plain FP32 layer, so evaluation, generation and export are unchanged."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if not self.training:
      return super().forward(x)
    b = None if self.bias is None else self.bias.to(torch.bfloat16)
    return F.linear(x.to(torch.bfloat16), self.weight.to(torch.bfloat16), b).float()


def bf16_linears(prior: P.CodePrior) -> None:
  """Train the transformer feed-forward, output head and observation-encoder
  Linears in BF16. LayerNorm, the residual stream, the token embedding and
  the loss keep FP32/TF32 (attention: ``TrainAttention``). BF16 autocast
  over the whole prior diverged.
  Swapping the class keeps parameters, their
  initialization and state-dict keys."""
  layers = [m for layer in prior.transformer.layers for m in (layer.linear1, layer.linear2)]
  layers += [prior.head] + [m for m in prior.obs_encoder if isinstance(m, torch.nn.Linear)]
  for m in layers:
    m.__class__ = BF16Linear


class TrainAttention(torch.nn.MultiheadAttention):
  """The prior's self-attention in training: the 4-position softmax attention
  is computed directly instead of ``scaled_dot_product_attention``, whose
  fused FP32 kernel is built for long sequences and dominated the optimizer
  pass. Projections, scores, softmax, attention dropout and the weighted sum
  stay exact FP32, also under TF32. Eval mode is the plain FP32 layer, so
  evaluation, generation and export are unchanged.
  Covers only the prior's use: batch-first self-attention without a key
  padding mask."""

  def forward(
    self,
    query,
    key,
    value,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False,
  ):
    if not self.training:
      return super().forward(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        attn_mask=attn_mask,
        average_attn_weights=average_attn_weights,
        is_causal=is_causal,
      )
    if key_padding_mask is not None or not (query is key is value) or not self.batch_first:
      raise ValueError("TrainAttention covers batch-first self-attention without padding")
    B, L, E = query.shape
    h = self.num_heads
    qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
    q, k, v = qkv.view(B, L, 3, h, E // h).permute(2, 0, 3, 1, 4)
    if is_causal and attn_mask is None:
      attn_mask = torch.triu(torch.full((L, L), float("-inf"), device=query.device), 1)
    # Elementwise products, not matmuls: under the trainer's TF32 context a
    # matmul here would run in TF32.
    scores = (q.unsqueeze(-2) * k.unsqueeze(-3)).sum(-1) * (E // h) ** -0.5
    if attn_mask is not None:
      scores = scores + attn_mask
    p = F.dropout(scores.softmax(-1), self.dropout)
    o = (p.unsqueeze(-1) * v.unsqueeze(-3)).sum(-2)
    o = o.transpose(1, 2).reshape(B, L, E)
    return F.linear(o, self.out_proj.weight, self.out_proj.bias), None


def train_attention(prior: P.CodePrior) -> None:
  """Apply ``TrainAttention`` to every transformer layer; parameters,
  initialization and state-dict keys are unchanged."""
  for layer in prior.transformer.layers:
    layer.self_attn.__class__ = TrainAttention


def dist_world() -> tuple[int, int]:
  if dist.is_available() and dist.is_initialized():
    return dist.get_rank(), dist.get_world_size()
  return 0, 1


class Collector:
  """Frozen-teacher expert collection in one MJLab environment.

  The teacher is frozen, so collection does not depend on the prior.
  """

  def __init__(self, cfg: Stage2Cfg, motions: Path, teacher: Path, device: str, num_envs: int):
    self.cfg, self.device, self.num_envs = cfg, device, num_envs
    self.kin, self.kin_torso = build_kinematics(device)
    for k in (self.kin, self.kin_torso):
      k.requires_grad_(False)
    tracker, self.teacher_info = load_teacher(teacher)
    if self.teacher_info["action_clip"] != cfg.action_clip:
      raise ValueError("teacher and Stage 2 action clips differ")
    self.teacher = tracker.to(device)
    self.lib = MotionLibrary(motions, device, split="train")
    # Expert collection resets uniformly over 1 s bins (the SuperMimic sampler's
    # uniform component); the frozen teacher is not adapted, so no return-based
    # curriculum is used. Coverage remains an open spec decision.
    self.sampler = ReturnSampler(self.lib.num_bins, device, mix=(1.0, 0.0, 0.0))
    steps = round(cfg.episode_seconds * POLICY_HZ)
    self.env = RealEnv(
      num_envs,
      device,
      self.lib,
      self.sampler,
      self.kin,
      self.kin_torso,
      cfg.gamma,
      steps,
      cuda_graphs=cfg.cuda_graphs,
      compile_rewards=cfg.compile_rewards,
      reference_offsets=self.teacher_info["reference_offsets"],
    )
    self.transitions = 0

  def collect(self, obs: torch.Tensor, tokens: torch.Tensor, codes_out: torch.Tensor) -> dict:
    """Fill (T, n, 87) frames (the teacher's frame width), (T, n, 4) tokens and
    (T, n, 16) codes in place.

    Diagnostics accumulate on the device and are read back once per rollout,
    so the collection loop never waits for the GPU.
    """
    steps = self.collect_steps(obs, tokens, codes_out)
    while True:
      try:
        next(steps)
      except StopIteration as done:
        return done.value

  @torch.no_grad()
  def collect_steps(self, obs: torch.Tensor, tokens: torch.Tensor, codes_out: torch.Tensor):
    """``collect`` one environment step per ``next``; returns its diagnostics."""
    cfg, env = self.cfg, self.env
    T = obs.shape[0]
    sums: dict[str, torch.Tensor] = {}
    ep_len, ep_ret, ends = [], [], torch.zeros(4, device=self.device)

    def add(**values):
      for k, v in values.items():
        sums[k] = sums[k] + v if k in sums else torch.as_tensor(v, dtype=torch.float32).clone()

    for t in range(T):
      _, features, history = env.policy_inputs()
      latent, codes, bounded = self.teacher.encode(features)
      action = self.teacher.decode(latent, history)
      action = action + cfg.action_noise_std * torch.randn_like(action)
      clipped = action.abs() > cfg.action_clip
      action = action.clamp(-cfg.action_clip, cfg.action_clip)
      obs[t] = history[:, -O.OBS_DIM :]  # current noisy frame incl. executed prev action
      tokens[t] = P.codes_to_tokens(codes)
      codes_out[t] = codes
      out = env.step(action)
      info, tr = out["info"], out["transition"]
      terms = info["terms"]
      add(
        **{
          "real/reward": info["reward"].mean(),
          "real/failure_rate": tr["failure"].float().mean(),
          "real/err_body_pos_m": terms["err_body_pos_m"].mean(),
          "real/err_body_ori_rad": terms["err_body_ori_rad"].mean(),
          "real/err_anchor_pos_m": terms["err_anchor_pos_m"].mean(),
          "real/actuator_saturation": info["saturation"].mean(),
          "real/action_clip_fraction": clipped.float().mean(),
          "real/action_abs_mean": action.abs().mean(),
          "teacher/tanh_saturation_fraction": (bounded.abs() > 2.97).float().mean(),
          "sim/max_nefc": info["max_nefc"].float(),
          "sim/nefc_overflow_fraction": info["nefc_overflow"],
        }
      )
      ep = info["episodes"]
      done = info["done"]
      ends += torch.stack(
        (done.sum(), tr["failure"].sum(), tr["timeout"].sum(), tr["clip_end"].sum())
      ).float()
      if ep:
        ep_len.append(ep["length"] / POLICY_HZ)
        ep_ret.append(ep["normalized"])
      yield
    self.transitions += T * self.num_envs
    keys = list(sums)
    values = torch.stack([sums[k] for k in keys] + list(ends)).tolist()
    stats = {k: v / T for k, v in zip(keys, values[: len(keys)], strict=True)}
    stats |= dict(zip(SUM_KEYS, values[len(keys) :], strict=True))
    lists = {
      "episodes/length_s": torch.cat(ep_len).tolist() if ep_len else [],
      "episodes/normalized_return": torch.cat(ep_ret).tolist() if ep_ret else [],
    }
    return {"means": stats, "lists": lists}

  def state_dict(self) -> dict:
    rng = {"cpu": torch.get_rng_state()}
    if self.device.startswith("cuda"):
      rng["cuda"] = torch.cuda.get_rng_state(self.device)
    return {
      "sampler": {k: v.cpu() for k, v in self.sampler.state_dict().items()},
      "rng": rng,
      "next_episode_id": self.env.next_episode_id,
      "transitions": self.transitions,
    }

  def load_state_dict(self, state: dict) -> None:
    self.sampler.load_state_dict(state["sampler"])
    torch.set_rng_state(state["rng"]["cpu"].cpu())
    if "cuda" in state["rng"] and self.device.startswith("cuda"):
      torch.cuda.set_rng_state(state["rng"]["cuda"].cpu(), self.device)
    self.transitions = state.get("transitions", 0)
    # Live simulator episodes are not native state: restart them with fresh ids.
    self.env.next_episode_id = state["next_episode_id"]
    self.env.reset()


class Stage2Trainer:
  def __init__(
    self,
    cfg: Stage2Cfg,
    motions: Path,
    teacher: Path,
    device: str,
    collect: bool = True,
    evaluate: bool = True,
  ):
    """``collect=False`` builds no training environment (evaluation only);
    ``evaluate=False`` builds no evaluation environment (non-zero ranks)."""
    cfg.validate()
    self.cfg, self.device, self.motions = cfg, device, Path(motions)
    self.rank, self.world_size = dist_world()
    torch.manual_seed(cfg.seed)
    self.collector = Collector(cfg, motions, teacher, device, cfg.num_envs) if collect else None
    if self.collector is not None:
      self.kin, self.kin_torso = self.collector.kin, self.collector.kin_torso
      self.teacher, self.teacher_info = self.collector.teacher, self.collector.teacher_info
      self.train_env, self.sampler = self.collector.env, self.collector.sampler
    else:
      self.kin, self.kin_torso = build_kinematics(device)
      for k in (self.kin, self.kin_torso):
        k.requires_grad_(False)
      tracker, self.teacher_info = load_teacher(teacher)
      if self.teacher_info["action_clip"] != cfg.action_clip:
        raise ValueError("teacher and Stage 2 action clips differ")
      self.teacher = tracker.to(device)
    if evaluate:
      self._build_eval(motions)
    self.prior = P.CodePrior(
      O.OBS_DIM,
      cfg.d_model,
      cfg.num_layers,
      cfg.num_heads,
      cfg.dropout,
      cfg.norm_ema_decay,
      cfg.norm_clamp,
    ).to(device)
    if cfg.bf16_linears and device.startswith("cuda"):
      bf16_linears(self.prior)
    if cfg.attention_math and device.startswith("cuda"):
      train_attention(self.prior)
    # Data-parallel ranks average gradients (spec); the per-rank minibatch and
    # learning rate are unchanged. Observation statistics are synchronized
    # explicitly in ``RunningObsNorm.update``, so buffers are not broadcast.
    self.model: torch.nn.Module = self.prior
    if self.world_size > 1:
      self.model = DDP(self.prior, broadcast_buffers=False)
    self.opt = torch.optim.AdamW(
      self.prior.parameters(),
      lr=cfg.lr,
      betas=cfg.betas,
      eps=cfg.eps,
      weight_decay=cfg.weight_decay,
      fused=device.startswith("cuda"),
    )
    self._loss = (
      torch.compile(self.loss_terms, dynamic=False) if cfg.compile_prior else (self.loss_terms)
    )
    self.iteration = 0
    self.transitions = 0
    self.stats: dict[str, list[float]] = {}

  def _build_eval(self, motions: Path) -> None:
    cfg, device = self.cfg, self.device
    steps = round(cfg.episode_seconds * POLICY_HZ)
    self.eval_names = evaluation_clips(motions, cfg.eval_envs)
    self.eval_lib = MotionLibrary(motions, device, clip_names=self.eval_names)
    self.eval_split = [c["split"] for c in self.eval_lib.clips]
    self.eval_env = RealEnv(
      len(self.eval_names),
      device,
      self.eval_lib,
      None,
      self.kin,
      self.kin_torso,
      cfg.gamma,
      steps,
      reset_perturbation=False,
      cuda_graphs=cfg.cuda_graphs,
      compile_rewards=cfg.compile_rewards,
      reference_offsets=self.teacher_info["reference_offsets"],
    )

  def _record(self, **values) -> None:
    for k, v in values.items():
      self.stats.setdefault(k, []).append(float(v))

  def record_collection(self, result: dict, transitions: int) -> None:
    """Merge one rollout's collector diagnostics."""
    for k, v in result["means"].items():
      self.stats.setdefault(k, []).append(v)
    for k, v in result["lists"].items():
      self.stats.setdefault(k, []).extend(v)
    self.transitions += transitions

  def start(self) -> None:
    """Reset every training episode once; later iterations continue them."""
    self.train_env.reset()

  def teacher_digest(self) -> str:
    return tensor_digest(self.teacher.state_dict().values())

  # --- expert collection ---------------------------------------------------
  def collect(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One fresh rollout: (T, N, 87) frames, (T, N, 4) tokens, (T, N, 16) codes."""
    T, n = self.cfg.rollout_steps, self.cfg.num_envs
    obs = torch.empty(T, n, O.OBS_DIM, device=self.device)
    tokens = torch.empty(T, n, P.NUM_TOKENS, dtype=torch.long, device=self.device)
    codes = torch.empty(T, n, 16, device=self.device)
    result = self.collector.collect(obs, tokens, codes)
    self.record_collection(result, obs.shape[0] * obs.shape[1])
    return obs, tokens, codes

  # --- supervised optimization ---------------------------------------------
  def loss_terms(self, x: torch.Tensor, y: torch.Tensor):
    logits = self.model(x, y)
    flat, target = logits.reshape(-1, P.VOCAB), y.reshape(-1)
    loss = F.cross_entropy(flat, target, label_smoothing=self.cfg.label_smoothing)
    with torch.no_grad():
      ce = F.cross_entropy(flat, target)
      acc = (logits.argmax(-1) == y).float().mean(0)
    return loss, ce, acc

  def optimize(self, obs: torch.Tensor, tokens: torch.Tensor) -> None:
    """One shuffled pass. Per-minibatch diagnostics stay on the device and are
    read back once per pass."""
    cfg = self.cfg
    x, y = obs.reshape(-1, O.OBS_DIM), tokens.reshape(-1, P.NUM_TOKENS)
    before = torch.cat([p.detach().flatten() for p in self.prior.parameters()])
    self.prior.train()
    rows = []
    with tf32(cfg.tf32_prior and self.device.startswith("cuda")):
      for _ in range(cfg.mini_epochs):
        for idx in torch.randperm(len(x), device=self.device).split(cfg.batch_size):
          xb, yb = x[idx], y[idx]
          self.prior.obs_norm.update(xb)  # statistics EMA, then normalize (GPC-code order)
          loss, ce, acc = self._loss(xb, yb)
          self.opt.zero_grad(set_to_none=True)
          loss.backward()
          gn = torch.nn.utils.clip_grad_norm_(self.prior.parameters(), cfg.grad_clip_norm)
          self.opt.step()
          rows.append(torch.cat((torch.stack((loss.detach(), ce, ce.exp(), acc.mean(), gn)), acc)))
    self.prior.eval()
    after = torch.cat([p.detach().flatten() for p in self.prior.parameters()])
    names = ["prior/loss", "prior/cross_entropy", "prior/perplexity", "prior/accuracy"]
    names += ["prior/grad_norm"] + [f"prior/accuracy_token{k}" for k in range(P.NUM_TOKENS)]
    table = torch.stack(rows)
    summary = torch.cat(
      (
        table.mean(0),
        (table[:, 4] > cfg.grad_clip_norm).float().mean()[None],
        (after - before).norm()[None],
      )
    ).tolist()
    for k, v in zip(names, summary[: len(names)], strict=True):
      self.stats.setdefault(k, []).append(v)
    norm = self.prior.obs_norm
    self._record(
      **{
        "prior/grad_clipped": summary[len(names)],
        "optim/updates": len(rows),
        "optim/param_delta_norm": summary[len(names) + 1],
        "optim/lr": self.opt.param_groups[0]["lr"],
        "obs_norm/updates": float(norm.updates),
        "obs_norm/mean_abs": norm.mean.abs().mean().item(),
        "obs_norm/std_mean": norm.var.sqrt().mean().item(),
      }
    )

  def train_iteration(self) -> None:
    """Collect one fresh rollout and optimize one shuffled pass over it."""
    t0 = time.perf_counter()
    obs, tokens, codes = self.collect()
    t1 = time.perf_counter()
    self.optimize(obs, tokens)
    t2 = time.perf_counter()
    self.iteration += 1
    n = obs.shape[0] * obs.shape[1]
    stats = code_stats(codes.reshape(-1, 16), tokens.reshape(-1, P.NUM_TOKENS), "teacher_codes")
    t3 = time.perf_counter()
    self._record(
      **stats,
      **{
        "time/collect_s": t1 - t0,
        "time/optimize_s": t2 - t1,
        "time/code_stats_s": t3 - t2,
        "time/iteration_s": t3 - t0,
        "throughput/examples_per_s": n * self.cfg.mini_epochs / max(t2 - t1, 1e-9),
        "throughput/env_steps_per_s_per_rank": n / max(t3 - t0, 1e-9),
      },
    )

  def pop_stats(self) -> dict[str, float]:
    out = {}
    for k, v in self.stats.items():
      if v:
        out[k] = float(sum(v)) if k in SUM_KEYS + ("optim/updates",) else float(sum(v) / len(v))
    ends = sum(out.get(k, 0.0) for k in SUM_KEYS[1:])
    if ends > 0:
      out["episodes/failure_fraction"] = out.get("episodes/failures", 0.0) / ends
    out["progress/iteration"] = float(self.iteration)
    out["progress/transitions"] = float(self.transitions)
    self.stats = {}
    return out

  # --- evaluation -------------------------------------------------------------
  def fingerprint(self) -> dict:
    return training_fingerprint(
      getattr(self, "train_env", None),
      getattr(self, "sampler", None),
      self.device,
      prior=list(self.prior.state_dict().values()),
      optimizer=[
        v for s in self.opt.state_dict()["state"].values() for v in s.values() if torch.is_tensor(v)
      ],
      teacher=list(self.teacher.state_dict().values()),
    )

  def _start_eval(self) -> torch.Tensor:
    env = self.eval_env
    clips = torch.arange(env.num_envs, device=self.device)
    env.reset(assignment=(clips, torch.zeros_like(clips)))
    return torch.tensor([s == "val" for s in self.eval_split], device=self.device)

  @torch.no_grad()
  def evaluate(self, quality: bool = False) -> tuple[dict[str, float], dict, dict]:
    """Held-out token loss and closed-loop generation in the separate eval env.

    1. Teacher-driven rollouts (no action noise) from frame 0 of each eval clip
       give held-out (frame, token) pairs; the prior's eval-mode loss is
       reported per split, with the teacher's own success as a baseline.
    2. The prior then drives the frozen decoder from the same starts with
       nucleus sampling. A generation fails when the pelvis falls below
       ``fall_height_m``; SuperMimic reference failures and tracking errors
       against the clip are diagnostics (the prior sees no reference).
    3. With ``quality``, the unconditional-sampling quality evaluation
       (``stage2_quality``) in its own environment, as ``quality/*`` means
       over ``quality_seeds`` seeds.
    The global RNG is forked and the training-state fingerprint compared.
    """
    cfg = self.cfg
    self.prior.eval()
    before = self.fingerprint()
    devices = [torch.device(self.device)] if self.device.startswith("cuda") else []
    metrics: dict[str, float] = {}
    with torch.random.fork_rng(devices=devices):
      torch.manual_seed(cfg.eval_seed)
      metrics |= self._eval_teacher_labels()
      metrics |= self._eval_generation()
      if quality:
        metrics |= self._eval_quality()
    after = self.fingerprint()
    metrics["eval/training_state_preserved"] = float(before == after)
    metrics["eval/teacher_unchanged"] = float(
      self.teacher_digest() == self.teacher_info["tracker_digest"]
    )
    return metrics, before, after

  def _eval_quality(self) -> dict[str, float]:
    """Unconditional sampling at the default protocol of
    ``gpc-evaluate-stage2-quality`` (32 starts x 8 samples, 20 s, free and
    2.4/9.8 m/s pushes), scalar metrics averaged over seeds."""
    cfg, sums, counts = self.cfg, {}, {}
    t0 = time.perf_counter()
    for seed in range(cfg.quality_seeds):
      traj = Q.rollouts(
        self.prior,
        self.teacher,
        self.kin,
        self.kin_torso,
        self.motions,
        self.device,
        starts=32,
        samples=8,
        pushes=(2.4, 9.8),
        seconds=20.0,
        push_s=4.0,
        top_p=cfg.top_p,
        temperature=cfg.temperature,
        action_clip=cfg.action_clip,
        gamma=cfg.gamma,
        seed=seed,
      )
      for cond, m in Q.metrics(traj).items():
        for k, v in m.items():
          if isinstance(v, float) and math.isfinite(v):
            key = f"quality/{cond}/{k}"
            sums[key] = sums.get(key, 0.0) + v
            counts[key] = counts.get(key, 0) + 1
    out = {k: v / counts[k] for k, v in sums.items()}
    return out | {"quality/seconds": time.perf_counter() - t0}

  def _eval_teacher_labels(self) -> dict[str, float]:
    cfg, env = self.cfg, self.eval_env
    val = self._start_eval()
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)
    failed = torch.zeros_like(alive)
    steps = torch.zeros(env.num_envs, device=self.device)
    xs, ys, ms = [], [], []
    for _ in range(cfg.eval_steps):
      _, features, history = env.policy_inputs()
      latent, codes, _ = self.teacher.encode(features)
      action = self.teacher.decode(latent, history).clamp(-cfg.action_clip, cfg.action_clip)
      xs.append(history[:, -O.OBS_DIM :])
      ys.append(P.codes_to_tokens(codes))
      ms.append(alive.clone())
      out = env.step(action, auto_reset=False)
      steps += alive.float()
      failed |= alive & out["transition"]["failure"]
      alive &= ~out["info"]["done"]
      if not alive.any():
        break
    x, y, m = torch.stack(xs), torch.stack(ys), torch.stack(ms)
    out = {}
    for split, sel in (("val", val), ("train", ~val)):
      if not sel.any():
        continue
      mask = m & sel[None]
      xb, yb = x[mask], y[mask]
      ce, acc, n = 0.0, torch.zeros(P.NUM_TOKENS, device=self.device), 0
      for idx in torch.arange(len(xb), device=self.device).split(4096):
        logits = self.prior.logits(xb[idx], yb[idx])
        ce += float(
          F.cross_entropy(logits.reshape(-1, P.VOCAB), yb[idx].reshape(-1), reduction="sum")
        )
        acc += (logits.argmax(-1) == yb[idx]).float().sum(0)
        n += len(idx)
      ce /= max(n * P.NUM_TOKENS, 1)
      out |= {
        f"heldout_{split}/token_cross_entropy": ce,
        f"heldout_{split}/token_perplexity": math.exp(ce),
        f"heldout_{split}/token_accuracy": float(acc.sum() / max(n * P.NUM_TOKENS, 1)),
        f"heldout_{split}/examples": float(n),
        f"teacher_{split}/success_rate": float((~failed[sel]).float().mean()),
        f"teacher_{split}/episode_length_s": float(steps[sel].mean() / POLICY_HZ),
      }
      out |= {
        f"heldout_{split}/token{k}_accuracy": float(acc[k] / max(n, 1)) for k in range(P.NUM_TOKENS)
      }
    return out

  def _eval_generation(self) -> dict[str, float]:
    cfg, env = self.cfg, self.eval_env
    val = self._start_eval()
    n = env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=self.device)
    ref_failed = torch.zeros_like(alive)
    steps = torch.zeros(n, device=self.device)
    acc: dict[str, torch.Tensor] = {}
    gen_codes, gen_tokens = [], []
    gen_time = 0.0
    for _ in range(cfg.eval_steps):
      _, features, history = env.policy_inputs()
      t0 = time.perf_counter()
      uniform = torch.rand(n, P.NUM_TOKENS, device=self.device)
      tokens, _ = self.prior.generate(history[:, -O.OBS_DIM :], uniform, cfg.top_p, cfg.temperature)
      codes = P.tokens_to_codes(tokens)
      action = self.teacher.decode(codes / FSQ_HALF_LEVELS, history)
      action = action.clamp(-cfg.action_clip, cfg.action_clip)
      gen_time += time.perf_counter() - t0
      teacher_codes = self.teacher.encode(features)[1]
      out = env.step(action, auto_reset=False)
      tr, info = out["transition"], out["info"]
      m = alive.float()
      steps += m
      in_clip = m * (tr["frame"] < self.eval_lib.lengths[tr["clip"].long()] - 1).float()
      vals = {
        "reward": (info["reward"], in_clip),
        "err_body_pos_m": (info["terms"]["err_body_pos_m"], in_clip),
        "err_anchor_pos_m": (info["terms"]["err_anchor_pos_m"], in_clip),
        "saturation": (info["saturation"], m),
        "code_match_teacher": ((codes == teacher_codes).float().mean(-1), m),
        "step_match_teacher": ((codes == teacher_codes).all(-1).float(), m),
        "in_clip_steps": (torch.ones_like(m), in_clip),
      }
      for k, (v, w) in vals.items():
        acc[k] = acc.get(k, torch.zeros(n, device=self.device)) + torch.where(w > 0, v * w, 0.0)
        acc[k + "_w"] = acc.get(k + "_w", torch.zeros(n, device=self.device)) + w
      gen_codes.append(codes[alive])
      gen_tokens.append(tokens[alive])
      ref_failed |= alive & tr["failure"] & (in_clip > 0)
      alive &= tr["next_state"][:, 2] >= cfg.fall_height_m
      if not alive.any():
        break
    out: dict[str, float] = {
      "gen/step_latency_ms_batch": 1000 * gen_time / max(len(gen_codes), 1),
      "gen/batch": float(n),
    }
    out |= code_stats(torch.cat(gen_codes), torch.cat(gen_tokens), "gen_codes")
    for split, sel in (("val", val), ("train", ~val)):
      if not sel.any():
        continue
      out |= {
        f"gen_{split}/survival_rate": float(alive[sel].float().mean()),
        f"gen_{split}/fall_rate": float((~alive[sel]).float().mean()),
        f"gen_{split}/survival_s": float(steps[sel].mean() / POLICY_HZ),
        f"gen_{split}/reference_failure_rate": float(ref_failed[sel].float().mean()),
      }
      for k in (
        "reward",
        "err_body_pos_m",
        "err_anchor_pos_m",
        "saturation",
        "code_match_teacher",
        "step_match_teacher",
      ):
        w = acc[k + "_w"][sel].sum()
        if w > 0:
          out[f"gen_{split}/{k}"] = float(acc[k][sel].sum() / w)
      out[f"gen_{split}/normalized_reward"] = out.get(f"gen_{split}/reward", 0.0) / R.MAX_REWARD
    return out

  # --- native state ---------------------------------------------------------
  def policy_state(self) -> dict:
    """Self-contained inference state: prior (with its observation statistics)
    and the frozen teacher it decodes through."""
    return {
      "stage": "2",
      "prior": self.prior.state_dict(),
      "tracker": self.teacher.state_dict(),
      "teacher": self.teacher_info,
      "iteration": self.iteration,
      "transitions": self.transitions,
      "cfg": self.cfg.to_dict(),
    }

  def rng_state(self) -> dict:
    rng = {"cpu": torch.get_rng_state()}
    if self.device.startswith("cuda"):
      rng["cuda"] = torch.cuda.get_rng_state(self.device)
    return rng

  def resumable_state(
    self, rngs: list[dict] | None = None, collectors: list[dict] | None = None
  ) -> dict:
    """``rngs`` and ``collectors`` hold every rank's trainer RNG and collector
    state in rank order; a single-process run fills both itself."""
    return {
      **self.policy_state(),
      "optimizer": self.opt.state_dict(),
      "rngs": rngs or [self.rng_state()],
      "collectors": collectors or [self.collector.state_dict()],
    }

  def load_resumable(self, state: dict) -> None:
    """Restore the prior, optimizer, counters, this rank's trainer RNG and its
    collector state (sampler, RNG, episode ids; live episodes restart)."""
    if state["teacher"]["sha256"] != self.teacher_info["sha256"]:
      raise ValueError("resume state was trained against a different teacher checkpoint")
    self.prior.load_state_dict(state["prior"])
    self.opt.load_state_dict(state["optimizer"])
    self.iteration, self.transitions = state["iteration"], state["transitions"]
    rngs = state["rngs"]
    if len(rngs) != self.world_size or len(state["collectors"]) != self.world_size:
      raise ValueError(f"resume state has {len(rngs)} ranks; this run has {self.world_size}")
    # torch.load(map_location=device) moves the RNG ByteTensors too; the
    # generator setters require CPU tensors (as in Stage 1).
    rng = rngs[self.rank]
    torch.set_rng_state(rng["cpu"].cpu())
    if "cuda" in rng and self.device.startswith("cuda"):
      torch.cuda.set_rng_state(rng["cuda"].cpu(), self.device)
    if self.collector is not None:
      self.collector.load_state_dict(state["collectors"][self.rank])
