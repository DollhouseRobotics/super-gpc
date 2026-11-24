"""Stage 1: model-based differentiable imagined-return tracker training.

SuperMimic Algorithm A.1 with a 16x7 FSQ tracker. Each iteration:

1. Collect 2 or 3 vector steps (UTD 0.4) with the current tracker plus
   Gaussian action noise (sigma 0.1); store executed actions in replay.
2. Fit the world model on 10 batches of 2048 eight-step autoregressive
   replay windows (pose and velocity loss after analytic integration).
3. Imagine 4096 rollouts of 32 steps from replay states through the frozen
   model and maximize the survival-weighted, critic-bootstrapped return
   (SuperMimic Eq. 2) through the encoder, FSQ straight-through estimator and decoder.
4. Fit the critic to fixed lambda-return targets of those rollouts (SuperMimic Eq. 3),
   5 epochs of 4 shuffled minibatches.

The first 64 vector steps only collect; there is no model-only warm-up.

Throughput (engineering, after BlinkMimic/InstantTrack/XMimic): on CUDA each
model update and each actor+critic update replays one CUDA graph, sampling
included, so an update costs one host launch instead of thousands. Replay
sampling reads only device-side bounds and metrics stay on the device until
they are logged, so nothing in an iteration waits for the host.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import mujoco
import torch

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1
from super_gpc.rl import state_increments as SI
from super_gpc.rl.networks import Critic, Tracker, WorldModel
from super_gpc.rl.optim import MuonAdam
from super_gpc.rl.replay import Replay
from super_gpc.tasks.tracking.mdp import jitter as J
from super_gpc.tasks.tracking.mdp import kinematics as K
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.mdp import rewards as R
from super_gpc.tasks.tracking.mdp.commands import ReturnSampler
from super_gpc.tasks.tracking.mdp.events import CONTEXT_DIM, normalize_context
from super_gpc.tasks.tracking.mdp.kinematics import Kinematics
from super_gpc.tasks.tracking.mdp.motion_library import POLICY_HZ, MotionLibrary
from super_gpc.tasks.tracking.rl.real_env import RealEnv, reference_window
from super_gpc.tasks.tracking.rl.stage1_cfg import Stage1Cfg

NA = G1.NUM_ACTIONS
CRITIC_DIM = O.LOCAL_STATE_DIM + CONTEXT_DIM + NA + O.ENCODER_DIM + 6 + 1


def build_kinematics(device: str) -> tuple[Kinematics, Kinematics]:
  mj = mujoco.MjModel.from_xml_path(str(G1.mjcf_path()))
  kin = Kinematics(mj, G1.ACTION_JOINT_NAMES, G1.TRACKED_BODIES).to(device)
  torso = Kinematics(mj, G1.ACTION_JOINT_NAMES, (G1.ANCHOR_BODY,)).to(device)
  return kin, torso


def evaluation_clips(lib_dir: Path, count: int) -> list[str]:
  """Deterministic clip selection by name: half validation and half training
  clips for a corpus with a validation split, else all from the training clips."""
  clips = json.loads((Path(lib_dir) / "clips.json").read_text())
  out = []
  has_val = any(c["split"] == "val" for c in clips)
  groups = (("val", count - count // 2), ("train", count // 2)) if has_val else (("train", count),)
  for split, n in groups:
    names = sorted(c["name"] for c in clips if c["split"] == split and c["frames"] > POLICY_HZ)
    stride = max(1, len(names) // max(n, 1))
    out += names[::stride][:n]
  return out


class GraphedUpdate:
  """Run a zero-argument training update ``fn`` (sampling, forward, backward,
  clipping and optimizer step) from one CUDA graph.

  The first ``warmup`` calls run eagerly on a side stream (they compile the
  kernels and allocate optimizer state); the next call captures ``fn`` and
  every call replays it. Random draws advance per replay. Outputs are cloned
  out of the graph's static buffers."""

  def __init__(self, fn, warmup: int = 3):
    self.fn, self.warmup, self.calls = fn, warmup, 0
    self.graph: torch.cuda.CUDAGraph | None = None
    self.out = None

  def __call__(self):
    self.calls += 1
    if self.calls <= self.warmup:
      side = torch.cuda.Stream()
      side.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(side):
        out = self.fn()
      torch.cuda.current_stream().wait_stream(side)
      return out
    if self.graph is None:
      torch.cuda.synchronize()
      torch.cuda.empty_cache()  # return warm-up blocks before the graph takes its pool
      self.graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(self.graph):
        self.out = self.fn()
    self.graph.replay()
    return torch.utils._pytree.tree_map(torch.clone, self.out)


class Stage1Trainer:
  graphed = False  # CUDA-graph updates (set by __init__; subclasses with own setup stay eager)
  amp_dtype: torch.dtype | None = None

  def __init__(self, cfg: Stage1Cfg, motions: Path, device: str):
    self.cfg, self.device = cfg, device
    torch.manual_seed(cfg.seed)
    cuda = device.startswith("cuda")
    self.graphed = bool(cfg.cuda_graphs and cfg.compile and cuda)
    self.amp_dtype = torch.bfloat16 if (cfg.bf16 and cuda) else None
    self.kin, self.kin_torso = build_kinematics(device)
    for k in (self.kin, self.kin_torso):
      k.requires_grad_(False)
    self.lib = MotionLibrary(motions, device, split="train")
    self.sampler = ReturnSampler(
      self.lib.num_bins,
      device,
      alpha=cfg.sampler_alpha,
      recent_share=cfg.recent_failure_share,
      probe_share=cfg.probe_share,
      bin_clip=self.lib.bin_clip,
      focus_seed=cfg.seed + 10000,
    )
    steps = round(cfg.episode_seconds * POLICY_HZ)
    self.episode_steps = steps
    shared = dict(reference_offsets=cfg.reference_offsets)
    # cuda_graphs: encoder features, rewards and reset-time set_const replay
    # from CUDA graphs, bitwise equal to eager.
    self.train_env = RealEnv(
      cfg.num_envs,
      device,
      self.lib,
      self.sampler,
      self.kin,
      self.kin_torso,
      cfg.gamma,
      steps,
      cuda_graphs=cfg.cuda_graphs,
      reward_scale=cfg.reward_scale,
      reward_anchor=cfg.reward_anchor,
      **shared,
    )
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
      **shared,
    )
    self.eval_envs = {"": self.eval_env}
    # Module-specific initialization keeps unchanged networks identical across
    # widths/configurations without advancing collection's random stream.
    devices = [torch.device(device)] if cuda else []
    with torch.random.fork_rng(devices=devices):
      torch.manual_seed(cfg.seed + 100)
      self.tracker = Tracker(O.ENCODER_DIM, O.HISTORY_DIM, O.ACTION_DIM).to(device)
      torch.manual_seed(cfg.seed + 200)
      self.world = WorldModel(
        O.LOCAL_STATE_DIM, NA, CONTEXT_DIM, SI.INCREMENT_DIM, SI.STATE_ERROR_DIM
      ).to(device)
      torch.manual_seed(cfg.seed + 300)
      self.critic = Critic(CRITIC_DIM).to(device)
    self.critic.VALUE_SCALE = 100.0 * cfg.reward_scale
    self.ema = (
      copy.deepcopy(self.tracker).eval().requires_grad_(False) if cfg.policy_ema_decay else None
    )
    self.ema_updates = 0
    self.lr_decay_at: int | None = None
    # Graphed updates need fused, capturable Adam with the learning rate held
    # in a device tensor that the schedule fills (same Adam arithmetic).
    self.init_optimizers()
    self.replay = Replay(cfg.replay_capacity, cfg.num_envs, device)
    self.default_q = self.train_env.default_q
    self.action_scale = torch.tensor(
      [G1.ACTION_SCALE[n] for n in G1.ACTION_JOINT_NAMES], device=device
    )
    self.soft_lo, self.soft_hi = self.train_env.soft_lo, self.train_env.soft_hi
    self.noise = O.noise_scale(device)
    self.iteration = 0
    self.vector_steps = 0
    self.normalizers_fitted = False
    self.stats: dict[str, list] = {}
    self.optim_rng: dict[str, dict] = {}
    self.init_reference_contacts()
    self.probe_env = None
    self.probe_state = None
    self.probe_rng = torch.Generator(device=device).manual_seed(cfg.seed + 20000)
    if cfg.probe_share:
      if self.ema is None:
        raise ValueError("EMA failure probes require policy_ema_decay")
      self.probe_tiles = torch.cat(
        [
          torch.stack((torch.full_like(f, c), f), -1)
          for c, length in enumerate(self.lib.lengths.tolist())
          for f in [torch.arange(0, length - 1, 300, device=device)]
        ]
      )
      self.probe_policy = copy.deepcopy(self.ema)
      with torch.random.fork_rng(devices=devices):
        self.probe_env = RealEnv(
          len(self.probe_tiles),
          device,
          self.lib,
          None,
          self.kin,
          self.kin_torso,
          cfg.gamma,
          751,
          reset_perturbation=False,
          nominal_physics=True,
          cuda_graphs=cfg.cuda_graphs,
          sensor_noise=False,
          **shared,
        )
    # Fusing the imagined step removes the launch overhead of the unrolled FK
    # (about 6 FK evaluations per step); numerics are checked against eager.
    compile_fn = torch.compile if cfg.compile else (lambda f, **_: f)
    self._step_fn = compile_fn(self._imagine_step, dynamic=False)
    self._terminal_fn = compile_fn(self._terminal_features, dynamic=False)
    self._rollout_fn = compile_fn(self.model_rollout, dynamic=False)
    if self.graphed:
      self._world_step = GraphedUpdate(self._world_update)
      self._policy_step = GraphedUpdate(self._policy_update)
    else:
      self._world_step, self._policy_step = self._world_update, self._policy_update
    self._log_clock, self._log_iteration = time.perf_counter(), 0

  # --- precision and learning-rate plumbing --------------------------------
  def init_reference_contacts(self) -> None:
    self.ref_contact = None
    if self.cfg.contact_weight:
      contacts = []
      for c, length in enumerate(self.lib.lengths.tolist()):
        frame = torch.arange(length, device=self.device)
        clip = torch.full_like(frame, c)
        contacts.append(R.calibrated_reference_contact(self.kin(self.lib.state(clip, frame))))
      self.ref_contact = torch.cat(contacts)

  def init_optimizers(self) -> None:
    """CQ optimizers, shared with refinement so checkpoint moments stay compatible."""
    cfg = self.cfg
    kw = dict(fused=self.device.startswith("cuda"), capturable=self.graphed)
    self.opt_world = MuonAdam(self.world, [self.world.dynamics], cfg.lr, betas=(0.9, 0.999), **kw)
    self.opt_actor = MuonAdam(
      self.tracker,
      [self.tracker.encoder, self.tracker.decoder],
      cfg.lr,
      betas=cfg.actor_betas,
      **kw,
    )
    self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self._lr_value(cfg.lr), **kw)

  def _lr_value(self, lr: float):
    return torch.tensor(lr, device=self.device) if self.graphed else lr

  def _amp(self):
    """bf16 autocast around the decoder and dynamics MLPs inside training
    updates only; everything else, collection, evaluation and export stay FP32."""
    if self.amp_dtype is None:
      return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=self.amp_dtype)

  def _device_lr(self) -> None:
    """Keep graphed optimizers' learning rates as device tensors (after a load)."""
    if not self.graphed:
      return
    for opt in (self.opt_world, self.opt_actor, self.opt_critic):
      for g in opt.param_groups:
        if not torch.is_tensor(g["lr"]) or g["lr"].device != torch.device(self.device):
          g["lr"] = torch.tensor(float(g["lr"]), device=self.device)

  # --- shared calculations ------------------------------------------------
  def reward_terms(self, s_next, ref_next, action, prev_action):
    return R.tracking_reward(
      self.kin(s_next),
      self.kin(ref_next),
      action,
      prev_action,
      s_next[..., 13:40],
      self.soft_lo,
      self.soft_hi,
      anchor=self.cfg.reward_anchor,
    )

  def remaining(self, lib, clip, frame) -> torch.Tensor:
    """Clip frames left, saturated and scaled to [0, 1] (critic boundary input)."""
    left = (lib.lengths[clip] - 1 - frame).clamp(0, self.cfg.value_horizon_frames)
    return (left.float() / self.cfg.value_horizon_frames)[..., None]

  def critic_features(self, state, prev_action, ctx_n, remaining, ref_now, encoder):
    return torch.cat(
      (
        O.local_state(state),
        ctx_n,
        prev_action,
        encoder,
        O.anchor_errors(self.kin_torso, state, ref_now),
        remaining,
      ),
      -1,
    )

  def _imagine_step(
    self, s, hist, prev, ctx_n, bias, ref_now, fut, ref_next, contact, remaining, noise
  ):
    """One differentiable imagined policy step (compiled as a unit)."""
    cfg = self.cfg
    enc = O.encoder_features(self.kin_torso, s, ref_now, fut)
    feat = self.critic_features(s, prev, ctx_n, remaining, ref_now, enc).detach()
    # The encoder and FSQ stay FP32 so rounding assigns the same codes as the
    # FP32 deployment graph; bf16 moves pre-round values across code boundaries.
    latent, codes, bounded = self.tracker.encode(enc)
    with self._amp():
      action = self.tracker.decode(latent, hist.reshape(hist.shape[0], -1))
    action = action.float()
    clipped = (action.abs() > cfg.action_clip).float().mean()
    action = action.clamp(-cfg.action_clip, cfg.action_clip)
    with self._amp():
      inc = self.world.increment(O.local_state(s), action, ctx_n)
    s_next = SI.compose(s, inc.float())
    r = self.reward_terms(s_next, ref_next, action, prev)["total"]
    if cfg.contact_weight:
      r = r + R.contact_bonus(self.kin(s_next), contact, cfg.contact_weight)
    r = r * cfg.reward_scale
    p_fail = R.soft_failure(s_next, ref_next)
    obs = O.actor_observation(s_next, action, bias, self.default_q) + noise
    hist = torch.cat((hist[:, 1:], obs[:, None]), 1)
    return feat, codes.detach(), bounded.detach(), clipped, action, s_next, r, p_fail, hist

  def _terminal_features(self, s, prev, ctx_n, ref_now, fut, remaining):
    enc = O.encoder_features(self.kin_torso, s, ref_now, fut)
    return self.critic_features(s, prev, ctx_n, remaining, ref_now, enc)

  def _record(self, **values) -> None:
    """Accumulate metrics without reading the device; ``pop_stats`` syncs once."""
    for k, v in values.items():
      self.stats.setdefault(k, []).append(
        v.detach().float().reshape(-1) if torch.is_tensor(v) else v
      )

  def _record_samples(self, key: str, values: torch.Tensor) -> None:
    """Per-episode samples (averaged over episodes, not steps)."""
    self.stats.setdefault(key, []).append(values.detach().float().reshape(-1))

  # --- real collection ----------------------------------------------------
  @torch.no_grad()
  def collect(self, steps: int, noise_std: float) -> None:
    env = self.train_env
    for _ in range(steps):
      _, features, history = env.policy_inputs()
      action = self.tracker(features, history)
      if noise_std > 0:
        action = action + noise_std * torch.randn_like(action)
      clipped = action.abs() > self.cfg.action_clip
      action = action.clamp(-self.cfg.action_clip, self.cfg.action_clip)
      out = env.step(action)
      self.replay.add(**out["transition"])
      self.vector_steps += 1
      info, tr = out["info"], out["transition"]
      t = info["terms"]
      self._record(
        **{
          "real/reward": info["reward"].mean(),
          "real/failure_rate": tr["failure"].float().mean(),
          "real/err_body_pos_m": t["err_body_pos_m"].mean(),
          "real/err_body_ori_rad": t["err_body_ori_rad"].mean(),
          "real/err_body_lin_vel": t["err_body_lin_vel"].mean(),
          "real/err_anchor_pos_m": t["err_anchor_pos_m"].mean(),
          "real/err_anchor_ori_rad": t["err_anchor_ori_rad"].mean(),
          "real/reward_action_rate": t["action_rate"].mean(),
          "real/reward_joint_limit": t["joint_limit"].mean(),
          "real/actuator_saturation": info["saturation"].mean(),
          "real/action_clip_fraction": clipped.float().mean(),
          "real/action_abs_mean": action.abs().mean(),
          "sim/max_nefc": info["max_nefc"],
          "sim/diverged_envs": info["diverged"],
          "sim/nefc_overflow_fraction": info["nefc_overflow"],
          "sim/contacts_per_world": info["contacts_per_world"],
        }
      )
      ep = info["episodes"]
      if ep:
        self._record(**{"episodes/finished": len(ep["length"])})
        self._record_samples("episodes/length_s", ep["length"] / POLICY_HZ)
        self._record_samples("episodes/normalized_return", ep["normalized"])
        self._record_samples("episodes/diagnostic_normalized_return", ep["diagnostic_normalized"])
        self._record_samples("episodes/return", ep["ret"])
        done_fail = tr["failure"].sum()
        self._record(
          **{
            "episodes/failures": done_fail,
            "episodes/timeouts": tr["timeout"].sum(),
            "episodes/clip_ends": tr["clip_end"].sum(),
          }
        )

  # --- normalization --------------------------------------------------------
  @torch.no_grad()
  def fit_normalizers(self, samples: int = 32768) -> None:
    st = self.replay.sample_starts(samples, O.HISTORY)
    clip, frame = st["clip"].long(), st["frame"].long()
    ref_now, fut = reference_window(self.lib, clip, frame, self.cfg.reference_offsets)
    enc = O.encoder_features(self.kin_torso, st["state"], ref_now, fut)
    self.tracker.encoder_norm.fit(enc)
    hist = st["history"]
    self.tracker.history_norm.fit(hist.reshape(samples, -1))
    # Previous-action channels are already normalized and their scale grows
    # with training; keep them unscaled.
    hn = self.tracker.history_norm
    for f in range(O.HISTORY):
      sl = slice(f * O.OBS_DIM + O.PREV_ACTION.start, f * O.OBS_DIM + O.PREV_ACTION.stop)
      hn.mean[sl], hn.std[sl] = 0.0, 1.0
    t, e = self.replay.sample_windows(samples, 1)
    w = self.replay.window(t, e, 1)
    s, s1 = w["state"][:, 0], w["next_state"][:, 0]
    self.world.local_norm.fit(O.local_state(s))
    self.world.increment_norm.fit(SI.increment(s, s1), min_std=1e-4)
    self.world.error_norm.fit(SI.state_error(s, s1), min_std=1e-4)
    ctx_n = normalize_context(st["context"])
    prev = hist[:, -1, O.PREV_ACTION]
    rem = self.remaining(self.lib, clip, frame)
    cf = self.critic_features(st["state"], prev, ctx_n, rem, ref_now, enc)
    self.critic.norm.fit(cf)
    cn = self.critic.norm
    a, b = O.LOCAL_STATE_DIM, O.LOCAL_STATE_DIM + CONTEXT_DIM + NA
    cn.mean[a:b], cn.std[a:b] = 0.0, 1.0
    self.normalizers_fitted = True

  # --- world model ----------------------------------------------------------
  def model_rollout(self, w: dict, length: int):
    """Autoregressive prediction over a replay window; returns per-step errors."""
    s = w["state"][:, 0]
    ctx_n = normalize_context(w["context"][:, 0])
    std = self.world.error_norm.std
    state_loss, preds = 0.0, []
    for j in range(length):
      with self._amp():
        inc = self.world.increment(O.local_state(s), w["action"][:, j], ctx_n)
      s = SI.compose(s, inc.float())
      target = w["next_state"][:, j]
      err = SI.state_error(target, s) / std
      state_loss = state_loss + (err * err).mean(-1)
      preds.append(s)
    return state_loss, torch.stack(preds, 1)

  def _world_update(self) -> dict[str, torch.Tensor]:
    """One model batch: sample windows, fit, step (graph-capturable)."""
    cfg = self.cfg
    t, e, valid = self.replay.sample_windows_fixed(cfg.model_batch_size, cfg.model_window)
    fields = ("state", "next_state", "action", "context")
    w = self.replay.window(t, e, cfg.model_window, fields)
    state_loss, _ = self._rollout_fn(w, cfg.model_window)
    m = valid.float()
    n = m.sum().clamp_min(1)
    loss = (state_loss * m).sum() / n
    self.opt_world.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(self.world.parameters(), cfg.grad_clip_norm)
    self.opt_world.step()
    return {
      "model/loss": loss.detach(),
      "model/state_loss": (state_loss.detach() * m).sum() / n,
      "model/grad_norm": gn.detach(),
      "model/valid_window_fraction": m.mean(),
    }

  def fit_world_model(self) -> None:
    for _ in range(self.cfg.model_batches):
      out = self._world_step()
    self._record(**out)  # last batch, as before

  # --- imagination ------------------------------------------------------------
  def imagine(self):
    """One actor update on imagined rollouts; returns detached critic data."""
    stats, data = self._imagine_update()
    codes0 = stats.pop("fsq/_codes")
    self._record(**stats)
    self._record(**{"fsq/unique_code_fraction": unique_fraction(codes0)})
    return data

  def _imagine_update(self):
    """Actor update without host reads: (device metrics, critic data)."""
    cfg, lib = self.cfg, self.lib
    B, H, gamma = cfg.actor_batch, cfg.horizon, cfg.gamma
    st = self.replay.sample_starts(B, O.HISTORY)
    s, hist = st["state"], st["history"]
    ctx = st["context"]
    ctx_n, bias = normalize_context(ctx), ctx[:, 5 : 5 + NA]
    clip, frame = st["clip"].long(), st["frame"].long()
    length = lib.lengths[clip]
    prev = hist[:, -1, O.PREV_ACTION]
    survival = torch.ones(B, device=self.device)
    valid = torch.ones(B, device=self.device)
    ret = torch.zeros(B, device=self.device)
    feats, rewards, dones, valids, clip_frac = [], [], [], [], []
    step = self._step_fn
    for t in range(H):
      ref_now, fut = reference_window(lib, clip, frame, cfg.reference_offsets)
      next_frame = torch.minimum(frame + 1, length - 1)
      contact = (
        self.ref_contact[lib.offsets[clip] + next_frame]
        if getattr(self, "ref_contact", None) is not None
        else torch.zeros(B, 2, device=self.device)
      )
      noise = (2.0 * torch.rand(B, O.OBS_DIM, device=self.device) - 1.0) * self.noise
      feat, codes, bounded, clipped, action, s_next, r, p_fail, hist = step(
        s,
        hist,
        prev,
        ctx_n,
        bias,
        ref_now,
        fut,
        lib.state(clip, frame + 1),
        contact,
        self.remaining(lib, clip, frame),
        noise,
      )
      if t == 0:
        codes0, bounded0 = codes, bounded
      feats.append(feat)
      clip_frac.append(clipped)
      end = (frame + 1 >= length - 1).float()
      d = end + (1 - end) * p_fail
      ret = ret + (gamma**t) * survival * r
      rewards.append(r.detach())
      dones.append(d.detach())
      valids.append(valid)
      survival = survival * (1 - d)
      valid = valid * (1 - end)
      prev, s = action, s_next
      frame = torch.minimum(frame + 1, length - 1)
    ref_now, fut = reference_window(lib, clip, frame, cfg.reference_offsets)
    cf = self._terminal_fn(s, prev, ctx_n, ref_now, fut, self.remaining(lib, clip, frame))
    feats.append(cf.detach())
    value_h = self.critic(cf)
    ret = ret + (gamma**H) * survival * value_h
    loss = -ret.mean() / H
    self.opt_actor.zero_grad(set_to_none=True)
    self.backward_actor(loss)
    enc_gn = _grad_norm_t(self.tracker.encoder.parameters())
    dec_gn = _grad_norm_t(self.tracker.decoder.parameters())
    gn = torch.nn.utils.clip_grad_norm_(self.tracker.parameters(), cfg.grad_clip_norm)
    torch.nn.utils.clip_grad_value_(self.tracker.parameters(), cfg.actor_grad_clip_value)
    self.opt_actor.step()
    stats = {
      "actor/loss": loss.detach(),
      "actor/imagined_return": ret.detach().mean(),
      "actor/imagined_reward": torch.stack(rewards).mean(),
      "actor/final_survival": survival.detach().mean(),
      "actor/grad_norm": gn.detach(),
      "actor/encoder_grad_norm": enc_gn,
      "actor/decoder_grad_norm": dec_gn,
      "actor/action_clip_fraction": torch.stack(clip_frac).detach().mean(),
      **fsq_stats(codes0, bounded0),
      "fsq/_codes": codes0,
    }
    data = (torch.stack(feats), torch.stack(rewards), torch.stack(dones), torch.stack(valids))
    return stats, data

  def backward_actor(self, loss: torch.Tensor) -> None:
    """Backpropagate the actor objective."""
    loss.backward()

  def fit_critic(self, feats, rewards, dones, valids) -> None:
    self._record(**self._critic_update(feats, rewards, dones, valids))

  def _critic_update(self, feats, rewards, dones, valids) -> dict[str, torch.Tensor]:
    cfg = self.cfg
    H, B = rewards.shape
    with torch.no_grad():
      values = self.critic(feats)
      target = torch.empty_like(rewards)
      nxt = values[H]
      for t in reversed(range(H)):
        nxt = rewards[t] + cfg.gamma * (1 - dones[t]) * (
          (1 - cfg.lam) * values[t + 1] + cfg.lam * nxt
        )
        target[t] = nxt
    x, y, m = feats[:H].reshape(H * B, -1), target.reshape(-1), valids.reshape(-1)
    n = H * B
    for _ in range(cfg.critic_epochs):
      perm = torch.randperm(n, device=self.device)
      for chunk in perm.chunk(cfg.critic_minibatches):
        err = (self.critic(x[chunk]) - y[chunk]) ** 2
        loss = (err * m[chunk]).sum() / m[chunk].sum().clamp_min(1)
        self.opt_critic.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
        self.opt_critic.step()
    return {
      "critic/loss": loss.detach(),
      "critic/grad_norm": gn.detach(),
      "critic/value_mean": values[:H].mean(),
      "critic/target_mean": target.mean(),
    }

  def _policy_update(self) -> dict[str, torch.Tensor]:
    """Actor update on imagined rollouts, then the critic fit (graph-capturable)."""
    self.world.requires_grad_(False)
    self.critic.requires_grad_(False)
    stats, data = self._imagine_update()
    self.world.requires_grad_(True)
    self.critic.requires_grad_(True)
    return stats | self._critic_update(*data)

  # --- loop -------------------------------------------------------------------
  @contextlib.contextmanager
  def optimization_rng(self, phase: str):
    """Model/policy draws must not advance live collection's random stream."""
    if not hasattr(self, "optim_rng"):
      self.optim_rng = {}
    cuda = self.device.startswith("cuda")
    with torch.random.fork_rng(devices=[torch.device(self.device)] if cuda else []):
      state = self.optim_rng.get(phase)
      if state is None:
        torch.manual_seed(self.cfg.seed + (30000 if phase == "model" else 40000))
      else:
        torch.set_rng_state(state["cpu"].cpu())
        if cuda:
          torch.cuda.set_rng_state(state["cuda"].cpu(), self.device)
      try:
        yield
      finally:
        self.optim_rng[phase] = {"cpu": torch.get_rng_state()}
        if cuda:
          self.optim_rng[phase]["cuda"] = torch.cuda.get_rng_state(self.device)

  def set_lr(self) -> float:
    lr = self.cfg.lr_at(self.iteration)
    cfg = self.cfg
    if cfg.actor_lr_multiplier != 1.0 and self.lr_decay_at is None:
      if self.iteration >= cfg.actor_lr_decay_by:
        self.lr_decay_at = self.iteration
    mult = cfg.actor_lr_multiplier
    if self.lr_decay_at is not None:
      fraction = min(1.0, (self.iteration - self.lr_decay_at) / cfg.actor_lr_decay_steps)
      mult += (1 - mult) * fraction
    for opt in (self.opt_world, self.opt_actor, self.opt_critic):
      for g in opt.param_groups:
        value = lr * (mult if opt is self.opt_actor else 1.0)
        if torch.is_tensor(g["lr"]):
          g["lr"].fill_(value)
        else:
          g["lr"] = value
    return lr

  def initial_fill(self) -> None:
    """Collection-only start; a fine-tune keeps its loaded normalizers."""
    self.train_env.reset()
    self.collect(self.cfg.initial_fill_steps, self.cfg.action_noise_std)
    if not self.normalizers_fitted:
      self.fit_normalizers()
      if getattr(self, "ema", None) is not None:
        self.ema.load_state_dict(self.tracker.state_dict())

  def train_iteration(self) -> None:
    cfg = self.cfg
    # Phase timings need device syncs; take them on one iteration per log
    # interval so the others never wait for the GPU.
    timed = self.iteration % cfg.log_interval == 0
    sync = torch.cuda.synchronize if (timed and self.device.startswith("cuda")) else (lambda: None)
    sync()
    t0 = time.perf_counter()
    lr = self.set_lr()
    steps = cfg.steps_for_iteration(self.iteration)
    self.collect(steps, cfg.action_noise_std)
    sync()
    t1 = time.perf_counter()
    self.world.train()
    with self.optimization_rng("model"):
      self.fit_world_model()
    sync()
    t2 = time.perf_counter()
    with self.optimization_rng("policy"):
      if self.graphed:
        stats = self._policy_step()
        codes0 = stats.pop("fsq/_codes")
        self._record(**stats)
        if timed:
          self._record(**{"fsq/unique_code_fraction": unique_fraction(codes0)})
        t3 = time.perf_counter()
      else:
        self.world.requires_grad_(False)
        self.critic.requires_grad_(False)
        data = self.imagine()
        self.world.requires_grad_(True)
        self.critic.requires_grad_(True)
        sync()
        t3 = time.perf_counter()
        self.fit_critic(*data)
    self.sampler.accumulate_history()
    ema_start = time.perf_counter()
    self.update_ema()
    if timed:
      sync()
      self._record(**{"time/ema_s": time.perf_counter() - ema_start})
    sync()
    t4 = time.perf_counter()
    self.iteration += 1
    if cfg.probe_share:
      self.probe_tick()
    self._record(**{"optim/lr": lr})
    if timed:
      self._record(
        **{
          "time/collect_s": t1 - t0,
          "time/model_s": t2 - t1,
          "time/actor_s": t3 - t2,  # actor and critic together when graphed
          "time/critic_s": t4 - t3,
          "time/iteration_s": t4 - t0,
          "throughput/env_steps_per_s": steps * cfg.num_envs / max(t1 - t0, 1e-9),
        }
      )

  @torch.no_grad()
  def update_ema(self) -> None:
    if getattr(self, "ema", None) is None:
      return
    decay = self.cfg.policy_ema_decay
    for target, live in zip(self.ema.parameters(), self.tracker.parameters(), strict=True):
      target.lerp_(live, 1 - decay)
    # Fixed fitted affine normalizers are inference state, not trainable weights.
    for target, live in zip(self.ema.buffers(), self.tracker.buffers(), strict=True):
      target.copy_(live)
    self.ema_updates += 1

  @torch.no_grad()
  def probe_tick(self) -> None:
    """EMA snapshot sweep, twenty side steps per update; apply only a full sweep.

    Separate simulator and forked RNG preserve continuing training episodes.
    The sweep is interleaved, not concurrently streamed: its full cost is timed.
    """
    cfg, env = self.cfg, self.probe_env
    if env is None:  # refinement has its own role-specific sampler
      return
    clock = time.perf_counter()
    devices = [torch.device(self.device)] if self.device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
      if self.probe_state is None:
        if self.iteration < cfg.probe_start or self.iteration % cfg.probe_every:
          return
        self.probe_policy.load_state_dict(self.ema.state_dict())
        clips, tiles = self.probe_tiles.unbind(-1)
        lead = int(torch.randint(200, 451, (1,), device=self.device, generator=self.probe_rng))
        start = (tiles - lead).clamp_min(0)
        env.reset(assignment=(clips, start))
        stop = torch.minimum(tiles + 300, self.lib.lengths[clips] - 1)
        self.probe_state = dict(
          active=torch.ones_like(clips, dtype=torch.bool),
          failed=torch.zeros_like(clips, dtype=torch.bool),
          frames=torch.full_like(clips, -1),
          stop=stop,
          began=self.iteration,
        )
      p = self.probe_state
      for _ in range(20):
        _, features, history = env.policy_inputs()
        action = self.probe_policy(features, history).clamp(-cfg.action_clip, cfg.action_clip)
        out = env.step(action, auto_reset=False)
        hit = p["active"] & out["transition"]["failure"]
        p["frames"][hit] = env.cmd.frame[hit]
        p["failed"] |= hit
        p["active"] &= ~hit & (env.cmd.frame < p["stop"])
        if not bool(p["active"].any()):
          self.sampler.record_probes(self.probe_tiles[:, 0], p["frames"], p["failed"])
          self._record(
            **{
              "probe/failure_fraction": p["failed"].float().mean(),
              "probe/snapshot_lag_updates": self.iteration - p["began"],
              "probe/worlds": len(self.probe_tiles),
            }
          )
          self.probe_state = None
          break
    self._record(**{"time/probe_s": time.perf_counter() - clock})

  def pop_stats(self) -> dict[str, float]:
    out: dict[str, float] = {}
    sums = ("episodes/finished", "episodes/failures", "episodes/timeouts", "episodes/clip_ends")
    keys, reduced = [], []  # device reductions, read back in one transfer
    for k, v in self.stats.items():
      ts = [x.to(self.device) for x in v if torch.is_tensor(x)]
      fs = [float(x) for x in v if not torch.is_tensor(x)]
      if ts:
        if fs:
          ts.append(torch.tensor(fs, device=self.device))
        vals = torch.cat(ts)
        if vals.numel():
          keys.append(k)
          reduced.append(vals.sum() if k in sums else vals.mean())
      elif fs:
        out[k] = float(sum(fs)) if k in sums else float(sum(fs) / len(fs))
    if reduced:
      out |= dict(zip(keys, torch.stack(reduced).cpu().tolist(), strict=True))
    if self.cfg.actor_lr_multiplier != 1.0 and self.lr_decay_at is None:
      if out.get("episodes/diagnostic_normalized_return", -math.inf) >= self.cfg.actor_lr_trigger:
        self.lr_decay_at = self.iteration
    out["optim/actor_lr_decay_at"] = float(self.lr_decay_at if self.lr_decay_at is not None else -1)
    out["optim/ema_updates"] = float(getattr(self, "ema_updates", 0))
    now = time.perf_counter()
    last_it, last_clock = getattr(self, "_log_iteration", None), getattr(self, "_log_clock", now)
    if last_it is not None and self.iteration > last_it:
      out["throughput/iterations_per_s"] = (self.iteration - last_it) / max(now - last_clock, 1e-9)
    self._log_clock, self._log_iteration = now, self.iteration
    ends = sum(
      out.get(k, 0.0) for k in ("episodes/failures", "episodes/timeouts", "episodes/clip_ends")
    )
    if ends > 0:
      out["episodes/failure_fraction"] = out.get("episodes/failures", 0.0) / ends
      out["episodes/timeout_fraction"] = out.get("episodes/timeouts", 0.0) / ends
      out["episodes/clip_end_fraction"] = out.get("episodes/clip_ends", 0.0) / ends
    s = self.sampler
    p = s.current()
    out["sampler/visited_bin_fraction"] = float((s.visits > 0).float().mean())
    out["sampler/estimate_mean"] = (
      float(s.estimate[s.visits > 0].mean()) if (s.visits > 0).any() else 0.0
    )
    out["sampler/current_entropy_ratio"] = float(-(p * (p + 1e-12).log()).sum() / math.log(len(p)))
    out["progress/iteration"] = float(self.iteration)
    out["progress/vector_steps"] = float(self.vector_steps)
    out["progress/transitions"] = float(self.vector_steps * self.cfg.num_envs)
    self.stats = {}
    return out

  # --- evaluation -------------------------------------------------------------
  def training_fingerprint(self) -> dict[str, str]:
    """Digests of everything a side rollout must leave untouched."""
    env = self.train_env
    sim = env.env.sim
    tensors = {
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
        (
          env.ep_len.float(),
          env.ep_return,
          env.ep_disc_return,
          env.ep_diagnostic_return,
          env.episode.float(),
        )
      ),
      "sampler": torch.cat([v.float() for v in self.sampler.state_dict().values()]),
      "rng_cpu": torch.get_rng_state(),
    }
    if self.device.startswith("cuda"):
      tensors["rng_cuda"] = torch.cuda.get_rng_state(self.device)
    return {
      k: hashlib.sha256(v.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]
      for k, v in tensors.items()
    } | {"replay_count": str(self.replay.count), "next_episode_id": str(env.next_episode_id)}

  @torch.no_grad()
  def evaluate(self) -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    """Deterministic held-out/training-clip rollouts in the separate eval envs.

    Starts every eval clip at frame 0 with training physics randomization and
    sensor noise, no reset perturbation and no action noise; runs to failure,
    clip completion or the 10 s timeout. The global RNG is forked so training
    randomness is unchanged, and the training-state fingerprint is compared.
    """
    before = self.training_fingerprint()
    devices = [torch.device(self.device)] if self.device.startswith("cuda") else []
    metrics: dict[str, float] = {}
    with torch.random.fork_rng(devices=devices):
      for tag, env in self.eval_envs.items():
        torch.manual_seed(self.cfg.eval_seed)
        metrics |= self._eval_rollout(env, tag)
        if self.ema is not None:
          torch.manual_seed(self.cfg.eval_seed)
          metrics |= self._eval_rollout(env, tag + "_ema", policy=self.ema)
    after = self.training_fingerprint()
    metrics["eval/training_state_preserved"] = float(before == after)
    if before != after:
      raise RuntimeError("Evaluation changed continuing training state")
    return metrics, before, after

  def _eval_rollout(self, env: RealEnv, tag: str, policy=None) -> dict[str, float]:
    cfg = self.cfg
    n = env.num_envs
    clips = torch.arange(n, device=self.device)
    starts = torch.zeros_like(clips)
    env.reset(assignment=(clips, starts))
    alive = torch.ones(n, dtype=torch.bool, device=self.device)
    failed = torch.zeros_like(alive)
    acc: dict[str, torch.Tensor] = {}
    traj: dict[str, list[torch.Tensor]] = {
      k: []
      for k in (
        "state",
        "action",
        "next_state",
        "context",
        "clip",
        "frame",
        "alive",
        "reward",
        "prev_action",
      )
    }
    steps = torch.zeros(n, device=self.device)
    for _ in range(cfg.eval_steps):
      prev = env.prev_action.clone()
      _, features, history = env.policy_inputs()
      action = (self.tracker if policy is None else policy)(features, history).clamp(
        -cfg.action_clip, cfg.action_clip
      )
      out = env.step(action, auto_reset=False)
      tr, info = out["transition"], out["info"]
      for k in ("state", "action", "next_state", "context", "clip", "frame"):
        traj[k].append(tr[k])
      traj["alive"].append(alive.clone())
      traj["reward"].append(info["reward"])
      traj["prev_action"].append(prev)
      m = alive.float()
      steps += m
      vals = {
        "reward": info["reward"],
        "saturation": info["saturation"],
        **{k: v for k, v in info["terms"].items() if k.startswith("err_")},
      }
      for k, v in vals.items():
        acc[k] = acc.get(k, torch.zeros(n, device=self.device)) + v * m
      failed |= alive & tr["failure"]
      alive &= ~info["done"]
      if not alive.any():
        break
    traj_t = {k: torch.stack(v) for k, v in traj.items()}
    # Physical target rates of the deterministic eval actions (jitter.py).
    live = traj_t["alive"][1:] & traj_t["alive"][:-1]
    live[: J.SETTLE_STEPS] = False
    actions = traj_t["action"]
    rates = J.target_rate_sums(actions * self.action_scale, live, POLICY_HZ)
    available = torch.minimum(
      self.eval_lib.lengths - 1 - starts, torch.full_like(self.eval_lib.lengths, cfg.eval_steps)
    ).float()
    metrics: dict[str, float] = {}
    for split in ("val", "train"):
      sel = torch.tensor([sp == split for sp in self.eval_split], device=self.device)
      if not sel.any():
        continue
      st = steps[sel].clamp_min(1)
      metrics[f"eval_{split}{tag}/success_rate"] = float((~failed[sel]).float().mean())
      metrics[f"eval_{split}{tag}/failure_rate"] = float(failed[sel].float().mean())
      metrics[f"eval_{split}{tag}/episode_length_s"] = float(steps[sel].mean() / POLICY_HZ)
      metrics[f"eval_{split}{tag}/normalized_return"] = float(
        (acc["reward"][sel] / (R.MAX_REWARD * available[sel])).mean()
      )
      metrics[f"eval_{split}{tag}/survived_fraction"] = float((steps[sel] / available[sel]).mean())
      for k, v in acc.items():
        metrics[f"eval_{split}{tag}/{k}"] = float((v[sel] / st).mean())
      metrics |= J.target_rate_metrics(
        {k: v[sel] for k, v in rates.items()}, G1.ACTION_JOINT_NAMES, f"eval_{split}{tag}/"
      )
    # Eval rollouts never enter replay; without a validation split the
    # model error is measured on all of them.
    val = torch.tensor([sp == "val" for sp in self.eval_split], device=self.device)
    metrics |= self.heldout_model_errors(traj_t, val if bool(val.any()) else ~val, tag)
    return metrics

  @torch.no_grad()
  def heldout_model_errors(
    self, traj: dict, envs: torch.Tensor, tag: str = "", max_windows: int = 4096
  ):
    """World-model error on held-out real eval transitions (never in replay)."""
    out: dict[str, float] = {}
    T = traj["state"].shape[0]
    alive = traj["alive"] & envs[None]
    for k in (1, 8, 32):
      if T < k:
        continue
      ok = torch.ones_like(alive[: T - k + 1])
      for j in range(k):
        ok &= alive[j : T - k + 1 + j]
      idx = ok.nonzero(as_tuple=False)
      if len(idx) == 0:
        continue
      idx = idx[torch.randperm(len(idx), device=self.device)[:max_windows]]
      t0, e = idx[:, 0], idx[:, 1]
      s = traj["state"][t0, e]
      ctx_n = normalize_context(traj["context"][t0, e])
      for j in range(k):
        a = traj["action"][t0 + j, e]
        inc = self.world.increment(O.local_state(s), a, ctx_n)
        s = SI.compose(s, inc)
      true = traj["next_state"][t0 + k - 1, e]
      clip, frame = traj["clip"][t0 + k - 1, e].long(), traj["frame"][t0 + k - 1, e].long()
      ref = self.eval_lib.state(clip, frame + 1)
      a, prev = traj["action"][t0 + k - 1, e], traj["prev_action"][t0 + k - 1, e]
      r_pred = R.tracking_reward(
        self.kin(s), self.kin(ref), a, prev, s[:, 13:40], self.soft_lo, self.soft_hi
      )["total"]
      r_true = traj["reward"][t0 + k - 1, e]
      bp, bt = self.kin(s, velocities=False).pos, self.kin(true, velocities=False).pos
      out |= {
        f"model_heldout{tag}/pelvis_pos_err_m_{k}step": float(
          (s[:, :3] - true[:, :3]).norm(dim=-1).mean()
        ),
        f"model_heldout{tag}/pelvis_rot_err_rad_{k}step": float(
          K.quat_angle_sq(s[:, 3:7], true[:, 3:7]).sqrt().mean()
        ),
        f"model_heldout{tag}/joint_pos_rmse_rad_{k}step": float(
          (s[:, 13:40] - true[:, 13:40]).pow(2).mean().sqrt()
        ),
        f"model_heldout{tag}/joint_vel_rmse_{k}step": float(
          (s[:, 40:67] - true[:, 40:67]).pow(2).mean().sqrt()
        ),
        f"model_heldout{tag}/body_pos_err_m_{k}step": float((bp - bt).norm(dim=-1).mean()),
        f"model_heldout{tag}/reward_abs_err_{k}step": float((r_pred - r_true).abs().mean()),
        f"model_heldout{tag}/reward_true_{k}step": float(r_true.mean()),
        f"model_heldout{tag}/windows_{k}step": float(len(idx)),
      }
    return out

  # --- native state ---------------------------------------------------------
  def policy_state(self) -> dict:
    return {
      "tracker": self.tracker.state_dict(),
      "iteration": self.iteration,
      "vector_steps": self.vector_steps,
      "cfg": self.cfg.to_dict(),
      "ema": self.ema.state_dict() if getattr(self, "ema", None) is not None else None,
      "ema_updates": getattr(self, "ema_updates", 0),
      "lr_decay_at": getattr(self, "lr_decay_at", None),
    }

  def resumable_state(self) -> dict:
    rng = {"cpu": torch.get_rng_state()}
    if self.device.startswith("cuda"):
      rng["cuda"] = torch.cuda.get_rng_state(self.device)
    return {
      **self.policy_state(),
      "world": self.world.state_dict(),
      "critic": self.critic.state_dict(),
      "opt_world": self.opt_world.state_dict(),
      "opt_actor": self.opt_actor.state_dict(),
      "opt_critic": self.opt_critic.state_dict(),
      "sampler": self.sampler.state_dict(),
      "replay": self.replay.state_dict(),
      "rng": rng,
      "next_episode_id": self.train_env.next_episode_id,
      "normalizers_fitted": self.normalizers_fitted,
      "probe_rng": self.probe_rng.get_state() if hasattr(self, "probe_rng") else None,
      "optim_rng": getattr(self, "optim_rng", {}),
    }

  def load_resumable(self, state: dict) -> None:
    self.tracker.load_state_dict(state["tracker"])
    if getattr(self, "ema", None) is not None:
      self.ema.load_state_dict(state["ema"])
      self.ema_updates = state["ema_updates"]
    self.lr_decay_at = state.get("lr_decay_at")
    self.optim_rng = state.get("optim_rng", {})
    if state.get("probe_rng") is not None:
      self.probe_rng.set_state(state["probe_rng"].cpu())
    self.world.load_state_dict(state["world"])
    self.critic.load_state_dict(state["critic"])
    self.opt_world.load_state_dict(state["opt_world"])
    self.opt_actor.load_state_dict(state["opt_actor"])
    self.opt_critic.load_state_dict(state["opt_critic"])
    self._device_lr()
    self.sampler.load_state_dict(state["sampler"])
    self.replay.load_state_dict(state["replay"])
    self.iteration, self.vector_steps = state["iteration"], state["vector_steps"]
    self._log_clock, self._log_iteration = time.perf_counter(), self.iteration
    self.normalizers_fitted = state["normalizers_fitted"]
    # torch.load(map_location=device) moves the RNG ByteTensors too; the
    # generator setters require CPU tensors.
    torch.set_rng_state(state["rng"]["cpu"].cpu())
    if "cuda" in state["rng"] and self.device.startswith("cuda"):
      torch.cuda.set_rng_state(state["rng"]["cuda"].cpu(), self.device)
    # Live simulator episodes are not part of the native state: restart them
    # with fresh episode ids so no replay window crosses the gap.
    self.train_env.next_episode_id = state["next_episode_id"]
    self.train_env.reset()


def _grad_norm(params) -> float:
  return float(_grad_norm_t(params))


def _grad_norm_t(params) -> torch.Tensor:
  sq = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
  return torch.stack(sq).sum().sqrt() if sq else torch.zeros(())


@torch.no_grad()
def fsq_stats(codes: torch.Tensor, bounded: torch.Tensor) -> dict[str, torch.Tensor]:
  """Code usage and saturation over one imagination batch (device tensors;
  the unique-code fraction needs a host read and is ``unique_fraction``)."""
  levels = torch.arange(-3, 4, device=codes.device, dtype=codes.dtype)
  hist = (codes[..., None] == levels).float().mean(0)  # (16, 7)
  entropy = -(hist * (hist + 1e-12).log()).sum(-1) / math.log(7)
  return {
    "fsq/level_entropy_ratio": entropy.mean(),
    "fsq/min_level_entropy_ratio": entropy.min(),
    "fsq/extreme_code_fraction": (codes.abs() == 3).float().mean(),
    "fsq/tanh_saturation_fraction": (bounded.abs() > 2.97).float().mean(),
    "fsq/levels_used_mean": (hist > 0).float().sum(-1).mean(),
  }


@torch.no_grad()
def unique_fraction(codes: torch.Tensor) -> float:
  return torch.unique(codes, dim=0).shape[0] / codes.shape[0]
