"""Joystick CoLA RLFT: categorical PPO on command adapters of the Stage 2 prior.

Port of the G1 gpc ``SteeringTrainer`` to MJLab and this repository's prior:

1. Collect ``rollout_steps`` (16) real steps in ``num_envs`` (4096) steering
   environments. The adapted prior samples four tokens from the current noisy
   frame and the local command; the frozen decoder turns the codes and the
   ten-frame history into the executed action. The rollout reward is the task
   reward minus ``kl_beta`` times the prefix-summed KL(adapted || base) at the
   sampled prefix.
2. GAE (gamma 0.99, lambda 0.95). Physical terminations do not bootstrap;
   timeouts bootstrap from the pre-reset terminal state. Advantages are
   centered, not scaled, so ``kl_beta`` keeps its meaning.
3. Two epochs of 2048-sample minibatches: clipped PPO on the joint four-token
   ratio plus ``kl_beta`` times the direct KL derivative; the critic regresses
   the GAE returns. Once a minibatch's approximate update KL exceeds 1.5x the
   target, the remaining actor steps of the iteration are skipped.

Only adapter and critic parameters are optimized.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from super_gpc.rl import prior as P
from super_gpc.rl.cola import ColaPolicy, categorical_statistics
from super_gpc.rl.cuda_graph import CudaGraphed
from super_gpc.rl.networks import Tracker
from super_gpc.tasks.steering.rl.cola_cfg import ColaCfg
from super_gpc.tasks.steering.rl.cola_env import PRIVILEGED_DIM, SteeringEnv
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.rl.stage2 import code_stats, file_sha256, tensor_digest
from super_gpc.tasks.tracking.rl.stage2_cfg import Stage2Cfg


def load_prior(path: Path) -> tuple[P.CodePrior, Tracker, dict]:
  """Frozen prior and decoder from a Stage 2 ``policy_state()`` file."""
  state = torch.load(path, map_location="cpu", weights_only=False)
  if state.get("stage") != "2":
    raise ValueError(f"{path} is not a Stage 2 policy checkpoint")
  # Only the architecture and action clip matter here; later Stage 2 code may
  # record config fields this checkout does not define.
  fields = Stage2Cfg.__dataclass_fields__.keys() | {"ff_size"}
  s2 = Stage2Cfg.from_dict({k: v for k, v in state["cfg"].items() if k in fields})
  prior = P.CodePrior(
    O.OBS_DIM, s2.d_model, s2.num_layers, s2.num_heads, s2.dropout, s2.norm_ema_decay, s2.norm_clamp
  )
  prior.load_state_dict(state["prior"])
  tracker = Tracker(O.ENCODER_DIM, O.HISTORY_DIM, O.ACTION_DIM)
  tracker.load_state_dict(state["tracker"])
  info = {
    "path": str(path),
    "sha256": file_sha256(path),
    "iteration": int(state["iteration"]),
    "transitions": int(state["transitions"]),
    "d_model": s2.d_model,
    "action_clip": s2.action_clip,
    "prior_digest": tensor_digest(prior.state_dict().values()),
    "tracker_digest": tensor_digest(tracker.state_dict().values()),
    "teacher": state["teacher"],
  }
  return prior, tracker, info


def gae(rewards, values, next_values, terminated, done, gamma, lam):
  """Timeouts bootstrap from the terminal state but never leak across resets."""
  adv = torch.zeros_like(rewards)
  carry = torch.zeros_like(rewards[0])
  for t in reversed(range(len(rewards))):
    delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
    carry = delta + gamma * lam * (~done[t]) * carry
    adv[t] = carry
  return adv, adv + values


@contextmanager
def tf32(enabled: bool):
  m, c = torch.backends.cuda.matmul, torch.backends.cudnn
  prev = (m.allow_tf32, c.allow_tf32)
  m.allow_tf32 = c.allow_tf32 = enabled
  try:
    yield
  finally:
    m.allow_tf32, c.allow_tf32 = prev


class ColaTrainer:
  """Joystick steering trainer. Other CoLA tasks subclass it and replace the
  environment (``make_env``), the task-input and privileged widths and the
  evaluation; collection, PPO and native state are shared."""

  @property
  def task_dim(self) -> int:
    return self.cfg.command_dim

  @property
  def privileged_dim(self) -> int:
    return PRIVILEGED_DIM

  def make_env(self, num_envs: int, seed: int, evaluation: bool):
    return SteeringEnv(self.cfg, num_envs, self.device, seed)

  def __init__(self, cfg: ColaCfg, prior_path: Path, device: str, evaluate: bool = True):
    cfg.validate()
    self.cfg, self.device = cfg, device
    torch.manual_seed(cfg.seed)
    prior, tracker, self.prior_info = load_prior(prior_path)
    if self.prior_info["action_clip"] != cfg.action_clip:
      raise ValueError("Stage 2 and RLFT action clips differ")
    self.policy = ColaPolicy(
      prior, tracker, cfg.action_clip, cfg.rank, cfg.alpha, self.task_dim
    ).to(device)
    self.policy.set_bf16(cfg.bf16_prior)
    # Compiled once per input shape; the rollout's statistics run inside its CUDA graph.
    self.stats_fn, self.ppo_logits = categorical_statistics, self.policy.logits
    if cfg.compile:
      self.stats_fn = torch.compile(categorical_statistics, dynamic=False)
      self.ppo_logits = torch.compile(self.policy.logits, dynamic=False)
    width, layers = O.OBS_DIM + self.task_dim + self.privileged_dim, []
    for _ in range(cfg.critic_layers):
      layers += [nn.Linear(width, cfg.critic_width), nn.ReLU()]
      width = cfg.critic_width
    self.critic = nn.Sequential(*layers, nn.Linear(width, 1)).to(device)
    fused = device.startswith("cuda")
    self.actor_opt = torch.optim.AdamW(
      self.policy.adapter_parameters(), lr=cfg.actor_lr, fused=fused
    )
    self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr, fused=fused)
    self.env = self.make_env(cfg.num_envs, cfg.seed + 2, evaluation=False)
    self.eval_env = self.make_env(cfg.eval_envs, cfg.eval_seed + 1, True) if evaluate else None
    self.sampler = torch.Generator(device=device).manual_seed(cfg.seed + 3)
    self.shuffle = torch.Generator(device=device).manual_seed(cfg.seed + 4)
    self.iteration = self.actor_steps = self.critic_steps = 0
    self.transitions = 0
    self.stats: dict[str, list[float]] = {}
    self.base_logp = torch.empty(
      cfg.rollout_steps, cfg.num_envs, P.NUM_TOKENS, P.VOCAB, device=device
    )
    d = prior.head.in_features
    shape = (cfg.rollout_steps, cfg.num_envs, P.NUM_TOKENS, d)
    self.layer0 = torch.empty(shape, device=device) if cfg.fused_base else None
    # On CUDA the policy step replays a graph per batch size and mode; its
    # large outputs are copied into the rollout buffers, not cloned.
    self.graphed_step = CudaGraphed(self._policy_step, clone=False)
    self.frozen_at_start = self.frozen_digest()

  # --- inference -------------------------------------------------------------
  def critic_input(self, inp: dict) -> torch.Tensor:
    clean = self.policy.prior.obs_norm(inp["clean"])  # frozen pretrained statistics
    return torch.cat((clean, inp["command"], inp["privileged"]), -1)

  def _policy_step(self, obs, command, history, uniform, base: bool = False):
    pol = self.policy
    first = None
    with tf32(self.cfg.tf32):
      if self.cfg.fused_base and not base:
        tokens, logits, ref, first = pol.rollout_step(obs, command, uniform)
      else:
        tokens, logits = pol.sample(obs, command, uniform, base=base)
        ref = logits if base else pol.logits(obs, command, tokens, base=True)
    ref = ref.float().log_softmax(-1)
    logp, entropy, kl = self.stats_fn(logits, tokens, ref)
    return pol.decode(tokens, history), tokens, logp, entropy, kl, ref, first

  @torch.no_grad()
  def act(
    self, inp: dict, generator: torch.Generator, base: bool = False, ref_out=None, layer0_out=None
  ):
    """(action, tokens, per-token logp, entropy sum, KL sum) for one step.
    ``ref_out`` receives the frozen base log-probabilities at the sampled
    prefix, which PPO reuses: the base is frozen and sees the same inputs.
    With ``fused_base``, ``layer0_out`` receives the frozen first block's
    outputs."""
    uniform = torch.rand(len(inp["obs"]), P.NUM_TOKENS, device=self.device, generator=generator)
    args = (inp["obs"], inp["command"], inp["history"], uniform)
    *small, ref, first = self.graphed_step(*args, base=base)
    action, tokens, logp, entropy, kl = (t.clone() for t in small)
    if ref_out is not None:
      ref_out.copy_(ref)
    if layer0_out is not None:
      layer0_out.copy_(first)
    return action, tokens, logp, entropy, kl

  def step_reward(self, out: dict) -> torch.Tensor:
    """Task reward of one environment step; tasks with a learned reward
    term (AMP) add it here."""
    return out["reward"]

  def start(self) -> None:
    self.env.reset()

  def _record(self, values: dict) -> None:
    for k, v in values.items():
      self.stats.setdefault(k, []).append(float(v))

  # --- collection ------------------------------------------------------------
  @torch.no_grad()
  def collect(self) -> dict[str, torch.Tensor]:
    cfg, env = self.cfg, self.env
    rows: dict[str, list[torch.Tensor]] = {}
    sums: dict[str, torch.Tensor] = {}
    lengths, returns, ends = [], [], torch.zeros(3, device=self.device)
    all_tokens = []
    inp = env.inputs()
    for step in range(cfg.rollout_steps):
      critic_in = self.critic_input(inp)
      value = self.critic(critic_in).squeeze(-1)
      action, tokens, logp, entropy, kl = self.act(
        inp,
        self.sampler,
        ref_out=self.base_logp[step],
        layer0_out=None if self.layer0 is None else self.layer0[step],
      )
      out = env.step(action)
      reward = self.step_reward(out)
      shaped = reward - cfg.kl_beta * kl
      nxt = env.inputs()  # terminal inputs before any reset
      next_value = self.critic(self.critic_input(nxt)).squeeze(-1)
      for k, v in dict(
        obs=inp["obs"],
        command=inp["command"],
        critic=critic_in,
        tokens=tokens,
        logp=logp.sum(-1),
        values=value,
        next_values=next_value,
        rewards=shaped,
        terminated=out["terminated"],
        done=out["done"],
      ).items():
        rows.setdefault(k, []).append(v)
      all_tokens.append(tokens)
      diag = {k: v.nanmean() for k, v in out["info"].items()} | {
        "reward/task": reward.mean(),
        "reward/shaped": shaped.mean(),
        "policy/entropy_sum": entropy.mean(),
        "policy/kl_base_sum": kl.mean(),
        "sim/max_nefc": out["max_nefc"].float(),
        "sim/nefc_overflow_fraction": out["nefc_overflow"],
      }
      for k, v in diag.items():
        sums[k] = sums[k] + v if k in sums else v.clone()
      ends += torch.stack(
        (out["done"].sum(), out["terminated"].sum(), out["timeout"].sum())
      ).float()
      finished = env.finish(out["done"])
      if finished:
        lengths.append(finished["length_s"])
        returns.append(finished["ret"])
      inp = nxt if not finished else env.inputs()
    batch = {k: torch.stack(v) for k, v in rows.items()}
    adv, ret = gae(
      batch["rewards"],
      batch["values"],
      batch["next_values"],
      batch["terminated"],
      batch["done"],
      cfg.gamma,
      cfg.gae_lambda,
    )
    batch["advantages"] = adv - adv.mean()
    batch["returns"] = ret
    keys = list(sums)
    vals = torch.stack([sums[k] for k in keys] + list(ends)).tolist()
    T = cfg.rollout_steps
    self._record({k: v / T for k, v in zip(keys, vals[: len(keys)], strict=True)})
    self._record(
      dict(
        zip(
          ("episodes/finished", "episodes/terminated", "episodes/timeouts"),
          vals[len(keys) :],
          strict=True,
        )
      )
    )
    if lengths:
      self.stats.setdefault("episodes/length_s", []).extend(torch.cat(lengths).tolist())
      self.stats.setdefault("episodes/return", []).extend(torch.cat(returns).tolist())
    toks = torch.cat(all_tokens)
    self._record(code_stats(P.tokens_to_codes(toks), toks, "rollout_codes"))
    self._record(
      {
        "values/mean": batch["values"].mean(),
        "values/return_mean": ret.mean(),
        "values/explained_variance": 1 - (ret - batch["values"]).var() / ret.var().clamp_min(1e-8),
      }
    )
    self.transitions += T * cfg.num_envs
    batch["base"] = self.base_logp
    if self.layer0 is not None:
      batch["layer0"] = self.layer0
    return {k: v.flatten(0, 1) for k, v in batch.items()}

  # --- optimization ----------------------------------------------------------
  def optimize(self, batch: dict[str, torch.Tensor]) -> None:
    cfg, pol = self.cfg, self.policy
    total = len(batch["tokens"])
    rows, skip_actor, actor_updates = [], False, 0
    pol.train()
    with tf32(cfg.tf32):
      for _ in range(cfg.mini_epochs):
        order = torch.randperm(total, device=self.device, generator=self.shuffle)
        for ids in order.split(cfg.minibatch_size):
          obs, cmd, tok = batch["obs"][ids], batch["command"][ids], batch["tokens"][ids]
          row = {}
          if not skip_actor:
            first = batch["layer0"][ids] if "layer0" in batch else None
            logits = self.ppo_logits(obs, cmd, tok, layer0=first)
            selected, entropy, kl = self.stats_fn(
              logits, tok, batch["base"][ids], entropy_grad=cfg.entropy_coefficient != 0
            )
            log_ratio = selected.sum(-1) - batch["logp"][ids]
            ratio = log_ratio.exp()
            a = batch["advantages"][ids]
            ppo = -torch.minimum(
              a * ratio, a * ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
            ).mean()
            ent = entropy.mean() / P.NUM_TOKENS
            # The KL penalty is a parameter-dependent reward: add its direct
            # derivative at the rollout prefixes (G1 gpc).
            loss = ppo - cfg.entropy_coefficient * ent + cfg.kl_beta * kl.mean()
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(pol.adapter_parameters(), cfg.gradient_clip)
            approx_kl = (ratio - 1 - log_ratio).mean().detach()
            skip_actor = bool(approx_kl > 1.5 * cfg.target_update_kl)
            if not skip_actor:  # do not apply an already-too-large update
              self.actor_opt.step()
              self.actor_steps += 1
              actor_updates += 1
            row |= {
              "actor/ppo_loss": ppo.detach(),
              "actor/loss": loss.detach(),
              "actor/entropy_per_token": ent.detach(),
              "actor/kl_base_sum": kl.mean().detach(),
              "actor/ratio_mean": ratio.mean().detach(),
              "actor/clip_fraction": ((ratio - 1).abs() > cfg.clip_ratio).float().mean(),
              "actor/approx_update_kl": approx_kl,
              "actor/grad_norm": gn.detach(),
            }
          v = self.critic(batch["critic"][ids]).squeeze(-1)
          vloss = (v - batch["returns"][ids]).square().mean()
          self.critic_opt.zero_grad(set_to_none=True)
          vloss.backward()
          cgn = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.gradient_clip)
          self.critic_opt.step()
          self.critic_steps += 1
          rows.append(row | {"critic/loss": vloss.detach(), "critic/grad_norm": cgn.detach()})
    pol.eval()
    keys = sorted({k for r in rows for k in r})
    for k in keys:
      vals = [r[k] for r in rows if k in r]
      self._record({k: torch.stack(vals).float().mean()})
    m = torch.stack([a.m.detach().norm() for a in pol.adapters])
    self._record(
      {
        "actor/updates": actor_updates,
        "actor/early_stopped": float(skip_actor),
        "adapter/m_norm_mean": m.mean(),
        "optim/actor_lr": self.actor_opt.param_groups[0]["lr"],
      }
    )

  def train_iteration(self) -> None:
    t0 = time.perf_counter()
    batch = self.collect()
    t1 = time.perf_counter()
    self.optimize(batch)
    if self.device.startswith("cuda"):
      torch.cuda.synchronize(self.device)
    t2 = time.perf_counter()
    self.iteration += 1
    n = self.cfg.num_envs * self.cfg.rollout_steps
    self._record(
      {
        "time/collect_s": t1 - t0,
        "time/optimize_s": t2 - t1,
        "time/iteration_s": t2 - t0,
        "throughput/env_steps_per_s": n / (t2 - t0),
        "throughput/collect_env_steps_per_s": n / (t1 - t0),
      }
    )

  def pop_stats(self) -> dict[str, float]:
    sums = ("episodes/finished", "episodes/terminated", "episodes/timeouts", "actor/updates")
    out = {k: (sum(v) if k in sums else sum(v) / len(v)) for k, v in self.stats.items() if v}
    if out.get("episodes/finished", 0) > 0:
      out["episodes/termination_fraction"] = out["episodes/terminated"] / out["episodes/finished"]
    out |= {
      "progress/iteration": float(self.iteration),
      "progress/transitions": float(self.transitions),
      "progress/actor_steps": float(self.actor_steps),
      "progress/critic_steps": float(self.critic_steps),
    }
    self.stats = {}
    return out

  # --- evaluation ------------------------------------------------------------
  def fingerprint(self) -> dict[str, str]:
    t = self.env.fingerprint_tensors() | {
      "rng_cpu": torch.get_rng_state(),
      "sampler": self.sampler.get_state(),
      "shuffle": self.shuffle.get_state(),
    }
    if self.device.startswith("cuda"):
      t["rng_cuda"] = torch.cuda.get_rng_state(self.device)
    out = {k: tensor_digest([v]) for k, v in t.items()}
    out["adapters"] = tensor_digest(self.policy.adapters.state_dict().values())
    out["critic"] = tensor_digest(self.critic.state_dict().values())
    out["frozen"] = self.frozen_digest()
    return out | {"next_episode_id": str(self.env.next_episode_id)}

  def frozen_digest(self) -> str:
    return tensor_digest(
      list(self.policy.prior.state_dict().values())
      + list(self.policy.tracker.state_dict().values())
    )

  def final_metrics(self, alive: torch.Tensor, info: dict) -> dict[str, float]:
    """Metrics of the last evaluation step (G1 gpc success thresholds)."""
    success = (
      alive
      & (info["task/fallen"] == 0)
      & (info["task/speed_error_mps"] < 0.3)
      & (info["task/velocity_error_mps"] < 0.4)
      & (info["task/facing_error_rad"] < 0.35)
    )
    return {"success_rate": float(success.float().mean())}

  @torch.no_grad()
  def evaluate(self, base: bool = False) -> tuple[dict[str, float], dict, dict]:
    """Seeded rollouts in the separate evaluation environment: every episode
    starts from reset with seeded commands and sampler draws; the training
    environment, RNG streams and parameters are fingerprinted around it."""
    cfg, env = self.cfg, self.eval_env
    before = self.fingerprint()
    devices = [torch.device(self.device)] if self.device.startswith("cuda") else []
    totals: dict[str, torch.Tensor] = {}
    weights: dict[str, torch.Tensor] = {}
    with torch.random.fork_rng(devices=devices):
      torch.manual_seed(cfg.eval_seed)
      env.rng.manual_seed(cfg.eval_seed + 1)
      gen = torch.Generator(device=self.device).manual_seed(cfg.eval_seed + 2)
      env.reset()
      n = env.num_envs
      alive = torch.ones(n, dtype=torch.bool, device=self.device)
      steps = torch.zeros(n, device=self.device)
      toks = []
      for _ in range(cfg.eval_steps):
        inp = env.inputs()
        action, tokens, _, entropy, kl = self.act(inp, gen, base=base)
        out = env.step(action)
        w = alive.float()
        steps += w
        values = out["info"] | {
          "reward/task": out["reward"],
          "policy/entropy_sum": entropy,
          "policy/kl_base_sum": kl,
        }
        for k, v in values.items():  # idle/* metrics are NaN outside idle steps
          wk = w * v.isfinite()
          totals[k] = totals.get(k, 0.0) + (v.nan_to_num() * wk).sum()
          weights[k] = weights.get(k, 0.0) + wk.sum()
        toks.append(tokens[alive])
        alive &= ~out["terminated"]
      final = self.final_metrics(alive, out["info"])
    prefix = "eval_base" if base else "eval"
    metrics = {f"{prefix}/{k}": float(v / weights[k]) for k, v in totals.items()}
    metrics |= {f"{prefix}/{k}": v for k, v in final.items()}
    metrics |= {
      f"{prefix}/survival_rate": float(alive.float().mean()),
      f"{prefix}/survival_s": float(steps.mean() / 50),
    }
    t = torch.cat(toks)
    metrics |= code_stats(P.tokens_to_codes(t), t, f"{prefix}_codes")
    after = self.fingerprint()
    metrics[f"{prefix}/training_state_preserved"] = float(before == after)
    metrics[f"{prefix}/frozen_unchanged"] = float(self.frozen_digest() == self.frozen_at_start)
    return metrics, before, after

  # --- native state ----------------------------------------------------------
  def policy_state(self) -> dict:
    """Self-contained inference state: adapters plus the frozen prior/decoder."""
    cpu = lambda m: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}  # noqa: E731
    return {
      "stage": "rlft-cola",
      "adapters": cpu(self.policy.adapters),
      "prior": cpu(self.policy.prior),
      "tracker": cpu(self.policy.tracker),
      "prior_info": self.prior_info,
      "iteration": self.iteration,
      "transitions": self.transitions,
      "cfg": self.cfg.to_dict(),
    }

  def resumable_state(self) -> dict:
    rng = {"cpu": torch.get_rng_state()}
    if self.device.startswith("cuda"):
      rng["cuda"] = torch.cuda.get_rng_state(self.device)
    return self.policy_state() | {
      "critic": self.critic.state_dict(),
      "actor_opt": self.actor_opt.state_dict(),
      "critic_opt": self.critic_opt.state_dict(),
      "actor_steps": self.actor_steps,
      "critic_steps": self.critic_steps,
      "rng": rng,
      "sampler": self.sampler.get_state(),
      "shuffle": self.shuffle.get_state(),
      "commands_rng": self.env.rng.get_state(),
      "next_episode_id": self.env.next_episode_id,
    }

  def load_resumable(self, state: dict) -> None:
    if state["prior_info"]["sha256"] != self.prior_info["sha256"]:
      raise ValueError("resume state was trained on a different Stage 2 prior")
    self.policy.adapters.load_state_dict(state["adapters"])
    self.critic.load_state_dict(state["critic"])
    self.actor_opt.load_state_dict(state["actor_opt"])
    self.critic_opt.load_state_dict(state["critic_opt"])
    self.iteration, self.transitions = state["iteration"], state["transitions"]
    self.actor_steps, self.critic_steps = state["actor_steps"], state["critic_steps"]
    torch.set_rng_state(state["rng"]["cpu"].cpu())
    if "cuda" in state["rng"] and self.device.startswith("cuda"):
      torch.cuda.set_rng_state(state["rng"]["cuda"].cpu(), self.device)
    self.sampler.set_state(state["sampler"].cpu())
    self.shuffle.set_state(state["shuffle"].cpu())
    self.env.rng.set_state(state["commands_rng"].cpu())
    # Live simulator episodes are not native state: restart them with fresh ids.
    self.env.next_episode_id = state["next_episode_id"]
    self.env.reset()
