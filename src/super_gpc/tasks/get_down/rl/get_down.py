"""Get-down CoLA RLFT: the shared categorical PPO on the frozen Stage 2 prior
with an AMP style reward.

The adapters see the five-frame task input, the frozen prior sees
proprioception, and the rollout reward is the task reward minus 0.05 times
KL(adapted || unconditioned prior). Collection, GAE, PPO, checkpoints and
resume are ``ColaTrainer``'s. This module adds the start sources, the
impact-weight ramp, the AMP discriminator (``rl.amp``) and the push
evaluation: from sampled pre-fall joystick-rollout
states, no push or a 0.6 or 1.0 m/s push forward, backward, left or right,
then the controller is switched on. The frozen prior and the damping
baseline are evaluated with the same seeds at iteration 0.

The style reward, ``amp_weight * max(0, 1 - 0.25 (D - 1)^2)``, is added to
every step's task reward with the discriminator of the collecting iteration.
After each PPO update the rollout's policy transitions enter a replay buffer
of recent ones, and the discriminator trains on draws from that buffer
against reference transitions.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from super_gpc.rl import prior as P
from super_gpc.rl.amp import AmpDiscriminator
from super_gpc.tasks.get_down import mdp as M
from super_gpc.tasks.get_down.rl.get_down_cfg import GetDownCfg
from super_gpc.tasks.get_down.rl.get_down_env import PRIVILEGED_DIM, GetDownEnv
from super_gpc.tasks.steering.rl.cola import ColaTrainer
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.mdp.motion_library import POLICY_HZ, MotionLibrary
from super_gpc.tasks.tracking.rl.stage2 import code_stats, tensor_digest


def push_conditions(magnitudes) -> list[tuple[str, float, float]]:
  """(label, forward, left) m/s: no push, then each direction and magnitude."""
  out = [("stand", 0.0, 0.0)]
  for name, fx, fy in (("fwd", 1, 0), ("back", -1, 0), ("left", 0, 1), ("right", 0, -1)):
    out += [(f"{name}_{m:g}", fx * m, fy * m) for m in magnitudes]
  return out


def load_bank(path: Path) -> dict:
  bank = torch.load(path, map_location="cpu", weights_only=False)
  if bank.get("kind") != "joystick-starts":
    raise ValueError(f"{path} is not a joystick start bank")
  return bank


def eval_starts(cfg: GetDownCfg, bank: dict) -> tuple[torch.Tensor, torch.Tensor]:
  """Fixed evaluation starts (eval_envs, 67) and their condition indices:
  pre-fall (standing or moving) bank states at heading zero, each paired
  across every push condition."""
  conditions = push_conditions(cfg.eval_push_mps)
  per = cfg.eval_envs // len(conditions)
  if per * len(conditions) != cfg.eval_envs:
    raise ValueError("eval_envs must be a multiple of the push conditions")
  labels = bank["labels"]
  pool = torch.nonzero(labels != M.LABELS.index("disturbed")).squeeze(-1)
  gen = torch.Generator().manual_seed(cfg.eval_seed)
  pick = pool[torch.randperm(len(pool), generator=gen)[:per]]
  states = bank["states"][pick]
  base = M.rotate_start(states, -M.heading(states))
  cond = torch.arange(cfg.eval_envs) // per
  forward = torch.tensor([conditions[c][1] for c in cond])
  left = torch.tensor([conditions[c][2] for c in cond])
  return M.push(base.repeat(len(conditions), 1), forward, left), cond


class GetDownTrainer(ColaTrainer):
  task_dim = M.TASK_DIM
  privileged_dim = PRIVILEGED_DIM
  # Push evaluation: per-trial peaks (metric -> info key, scale), means over
  # the final ``eval_lying_s`` (metric -> info key) and the settled metrics
  # also reported per push condition.
  peak_metrics = {
    "floor_force_kn": ("task/peak_floor_force_n", 1e-3),
    "body_force_kn": ("task/peak_body_force_n", 1e-3),
    "torso_force_kn": ("task/peak_torso_force_n", 1e-3),
    "wrist_force_kn": ("task/peak_wrist_force_n", 1e-3),
    "pelvis_down_speed_mps": ("task/pelvis_down_speed_mps", 1.0),
  }
  settled_metrics = {
    "lying_rate": "task/lying",
    "on_back_rate": "task/on_back",
    "torso_height_m": "task/torso_height_m",
    "pelvis_height_m": "real/pelvis_height_m",
    "pelvis_speed_mps": "task/pelvis_speed_mps",
  }
  push_settled = ("lying_rate", "on_back_rate")

  def __init__(
    self,
    cfg: GetDownCfg,
    prior_path: Path,
    motions: Path,
    starts: Path,
    device: str,
    evaluate: bool = True,
  ):
    self.library = MotionLibrary(motions, device)
    self.bank = load_bank(starts)
    self.bank_states = self.bank["states"].to(device)
    self.device = device
    self.conditions = push_conditions(cfg.eval_push_mps)
    starts_eval, self.eval_condition = eval_starts(cfg, self.bank)
    self.eval_assignment = starts_eval.to(device)
    super().__init__(cfg, prior_path, device, evaluate)
    reference = MotionLibrary(motions, device, clip_names=list(cfg.reference_clips))
    self.reference_info = {
      "clips": [
        {k: c[k] for k in ("name", "dataset", "frames", "offset")} for c in reference.clips
      ],
      "tail_s": cfg.reference_tail_s,
    }
    self.expert, ends = M.reference_transitions(
      reference, cfg.reference_tail_s, self.env.amp_kin, cfg
    )
    self.reference_info |= {
      "end_s": ends,
      "transitions": len(self.expert),
      "digest": tensor_digest([self.expert]),
    }
    self.disc = AmpDiscriminator(M.AMP_DIM, tuple(cfg.disc_hidden)).to(device)
    self.disc.norm.update(self.expert)
    self.disc_opt = torch.optim.AdamW(
      self.disc.parameters(), lr=cfg.disc_lr, weight_decay=cfg.disc_weight_decay
    )
    self.disc_gen = torch.Generator(device=device).manual_seed(cfg.seed + 5)
    self.disc_steps = 0
    # Ring buffer of recent policy transitions; the discriminator draws its
    # policy samples from it rather than from the latest rollout alone.
    self.replay = torch.zeros(cfg.disc_replay, M.AMP_DIM, device=device)
    self.replay_count = self.replay_next = 0
    self.amp_rows: list[torch.Tensor] | None = None

  def make_env(self, num_envs: int, seed: int, evaluation: bool) -> GetDownEnv:
    return GetDownEnv(self.cfg, num_envs, self.device, seed, self.library, self.bank_states)

  def start(self) -> None:
    super().start()
    self.env.stagger()

  def train_iteration(self) -> None:
    progress = self.iteration / max(self.cfg.impact_ramp_iterations, 1)
    self.env.impact_progress = progress
    body, torso, wrist = M.impact_weights(self.cfg, progress)
    self._record(
      {"reward_weights/body_impact": body, "reward_weights/torso_impact": torso}
      | {"reward_weights/wrist_impact": wrist}
      | {"starts/joystick_fraction": self.env.starts.source.float().mean()}
    )
    super().train_iteration()

  # --- AMP ---------------------------------------------------------------------
  def step_reward(self, out: dict) -> torch.Tensor:
    style = self.cfg.amp_weight * self.disc.style_reward(out["amp"])
    if self.amp_rows is not None:
      self.amp_rows.append(out["amp"])
    out["info"]["reward/amp_style"] = style
    return out["reward"] + style

  def collect(self) -> dict[str, torch.Tensor]:
    self.amp_rows = []
    try:
      batch = super().collect()
      batch["amp"] = torch.cat(self.amp_rows)
    finally:
      self.amp_rows = None
    return batch

  def optimize(self, batch: dict[str, torch.Tensor]) -> None:
    super().optimize(batch)
    self.update_discriminator(batch["amp"])

  def _expert_batch(self, n: int) -> torch.Tensor:
    ids = torch.randint(0, len(self.expert), (n,), device=self.device, generator=self.disc_gen)
    return self.expert[ids]

  def _store(self, policy: torch.Tensor) -> None:
    n, size = len(policy), len(self.replay)
    ids = (torch.arange(n, device=self.device) + self.replay_next) % size
    self.replay[ids] = policy
    self.replay_next = (self.replay_next + n) % size
    self.replay_count = min(self.replay_count + n, size)

  def update_discriminator(self, policy: torch.Tensor) -> None:
    cfg, disc = self.cfg, self.disc
    disc.norm.update(torch.cat((policy, self._expert_batch(len(policy)))))
    self._store(policy)
    rows = []
    disc.train()
    for _ in range(cfg.disc_epochs):
      for _ in range(-(-len(policy) // cfg.disc_minibatch)):
        ids = torch.randint(
          0, self.replay_count, (cfg.disc_minibatch,), device=self.device, generator=self.disc_gen
        )
        loss, stats = disc.loss(
          self.replay[ids],
          self._expert_batch(cfg.disc_minibatch),
          cfg.disc_grad_penalty,
          cfg.disc_logit_reg,
        )
        self.disc_opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(disc.parameters(), cfg.gradient_clip)
        self.disc_opt.step()
        self.disc_steps += 1
        rows.append(stats | {"disc/loss": loss.detach(), "disc/grad_norm": gn.detach()})
    disc.eval()
    for k in rows[0]:
      self._record({k: torch.stack([r[k] for r in rows]).float().mean()})
    with torch.no_grad():
      expert_reward = disc.style_reward(self._expert_batch(cfg.disc_minibatch)).mean()
    self._record({"disc/expert_style_reward": expert_reward})

  def pop_stats(self) -> dict[str, float]:
    out = super().pop_stats()
    out["progress/disc_steps"] = float(self.disc_steps)
    return out

  # --- evaluation ------------------------------------------------------------
  def _set_damping(self, env: GetDownEnv) -> None:
    """G1 damping mode: zero position gain, keep each actuator's (randomized)
    damping. Overwritten by the next reset's physics randomization."""
    m, c = env.env.sim.model, env.ctrl[None]
    e = torch.arange(env.num_envs, device=self.device)[:, None]
    m.actuator_gainprm[e, c, 0] = 0.0
    m.actuator_biasprm[e, c, 1] = 0.0

  @torch.no_grad()
  def _push_eval(self, prefix: str, mode: str) -> dict[str, float]:
    cfg, env = self.cfg, self.eval_env
    devices = [torch.device(self.device)] if self.device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
      torch.manual_seed(cfg.eval_seed)
      gen = torch.Generator(device=self.device).manual_seed(cfg.eval_seed + 2)
      env.impact_progress = 1.0  # always scored with the final impact weights
      env.starts.assignment = self.eval_assignment
      try:
        env.reset()
      finally:
        env.starts.assignment = None
      if mode == "damping":
        self._set_damping(env)
      n = env.num_envs
      alive = torch.ones(n, dtype=torch.bool, device=self.device)
      peak = {k: torch.zeros(n, device=self.device) for k in self.peak_metrics}
      tail = round(cfg.eval_lying_s * POLICY_HZ)
      settled = {k: torch.zeros(n, device=self.device) for k in self.settled_metrics}
      totals: dict[str, torch.Tensor] = {}
      toks = []
      for step in range(cfg.eval_steps):
        inp = env.inputs()
        if mode == "damping":
          action = torch.zeros(n, O.ACTION_DIM, device=self.device)
          kl = entropy = torch.zeros(n, device=self.device)
        else:
          action, tokens, _, entropy, kl = self.act(inp, gen, base=mode == "base")
          toks.append(tokens[alive])
        out = env.step(action)
        reward = self.step_reward(out)
        info, w = out["info"], alive.float()
        for k, (key, scale) in self.peak_metrics.items():
          peak[k] = torch.maximum(peak[k], info[key] * scale * w)
        if step >= cfg.eval_steps - tail:
          for k, key in self.settled_metrics.items():
            settled[k] += info[key] * w / tail
        values = info | {"reward/task": reward, "policy/kl_base_sum": kl}
        values["policy/entropy_sum"] = entropy
        for k, v in values.items():
          totals[k] = totals.get(k, 0.0) + (v * w).sum()
        totals["_w"] = totals.get("_w", 0.0) + w.sum()
        alive &= ~out["terminated"]
    wsum = totals.pop("_w")
    metrics = {f"{prefix}/{k}": float(v / wsum) for k, v in totals.items()}
    metrics |= {f"{prefix}/peak_{k}": float(v.mean()) for k, v in peak.items()}
    metrics |= {f"{prefix}/settled_{k}": float(v.mean()) for k, v in settled.items()}
    metrics[f"{prefix}/survival_rate"] = float(alive.float().mean())
    cond = self.eval_condition.to(self.device)
    for i, (label, _, _) in enumerate(self.conditions):
      sel = cond == i
      metrics[f"{prefix}_push/{label}_peak_floor_force_kn"] = float(
        peak["floor_force_kn"][sel].mean()
      )
      for k in self.push_settled:
        metrics[f"{prefix}_push/{label}_settled_{k}"] = float(settled[k][sel].mean())
    if toks:
      t = torch.cat(toks)
      metrics |= code_stats(P.tokens_to_codes(t), t, f"{prefix}_codes")
    return metrics

  @torch.no_grad()
  def evaluate(self, base: bool = False) -> tuple[dict[str, float], dict, dict]:
    """Seeded push evaluation in the separate environment; ``base`` also
    scores damping mode. The training environment, RNG streams and
    parameters are fingerprinted around it."""
    before = self.fingerprint()
    if base:
      metrics = self._push_eval("eval_base", "base") | self._push_eval("eval_damping", "damping")
    else:
      metrics = self._push_eval("eval", "adapted")
    after = self.fingerprint()
    p = "eval_base" if base else "eval"
    metrics[f"{p}/training_state_preserved"] = float(before == after)
    metrics[f"{p}/frozen_unchanged"] = float(self.frozen_digest() == self.frozen_at_start)
    return metrics, before, after

  # --- native state ----------------------------------------------------------
  def fingerprint(self) -> dict[str, str]:
    out = super().fingerprint()
    out["discriminator"] = tensor_digest(self.disc.state_dict().values())
    out["disc_gen"] = tensor_digest([self.disc_gen.get_state()])
    out["replay"] = tensor_digest([self.replay[: self.replay_count]])
    out["replay_pointers"] = f"{self.replay_count}/{self.replay_next}"
    return out

  def policy_state(self) -> dict:
    return super().policy_state() | {
      "stage": "rlft-get-down",
      "task_dim": M.TASK_DIM,
      "reference": self.reference_info,
    }

  def resumable_state(self) -> dict:
    return super().resumable_state() | {
      "discriminator": self.disc.state_dict(),
      "disc_opt": self.disc_opt.state_dict(),
      "disc_gen": self.disc_gen.get_state(),
      "disc_steps": self.disc_steps,
      "replay": self.replay[: self.replay_count].cpu(),
      "replay_next": self.replay_next,
    }

  def load_resumable(self, state: dict) -> None:
    if state["reference"]["digest"] != self.reference_info["digest"]:
      raise ValueError("resume state was trained on different reference transitions")
    self.disc.load_state_dict(state["discriminator"])
    self.disc_opt.load_state_dict(state["disc_opt"])
    self.disc_gen.set_state(state["disc_gen"].cpu())
    self.disc_steps = state["disc_steps"]
    replay = state["replay"]
    self.replay_count, self.replay_next = len(replay), state["replay_next"]
    self.replay[: self.replay_count] = replay.to(self.device)
    super().load_resumable(state)  # restores the start RNG, then resets every env
    self.env.stagger()
