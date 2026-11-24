# super-gpc training specification

This repository trains generative controllers for the Dollhouse Robotics
27-actuator G1 (`g1_body25_head2`) on MJLab. It contains four trainings:

1. **Stage 1 tracker** (`gpc-train-stage1`): SuperMimic learned-dynamics
   policy optimization with an FSQ encoder/decoder tracker, using the CQ
   recipe with an imagination-only contact reward.
2. **Stage 2 prior** (`gpc-train-stage2`): an autoregressive transformer over
   the tracker's grouped FSQ codes, trained on frozen-tracker rollouts.
3. **Joystick RLFT** (`gpc-train-joystick`): CoLA DoRA/FiLM adapters on the
   frozen prior, with idle gaze.
4. **Get-down RLFT** (`gpc-train-get-down`): CoLA adapters trained with an AMP
   style reward, a pelvis-height reward and floor-impact penalties, starting
   from a joystick-rollout bank.

This document describes what the code does with its current defaults. The
configuration dataclasses are the authoritative values:
`tasks/tracking/rl/stage1_cfg.py`, `tasks/tracking/rl/stage2_cfg.py`,
`tasks/steering/rl/cola_cfg.py` and `tasks/get_down/rl/get_down_cfg.py`
(all under `src/super_gpc/`).

## Sources and conventions

| Tag | Source |
| --- | --- |
| **SM** | SuperMimic (arXiv:2610.28480v1): Sections 3.1–3.3, Appendix A, Tables A.1–A.5, Algorithm A.1, Appendix D.1 |
| **FSQ** | Mentzer et al. 2023, finite scalar quantization (arXiv:2309.15505) |
| **GPC-code** | ProtoMotions original-GPC prior experiment, pinned at `NVlabs/ProtoMotions@b43fc6f`, and its inherited optimizer and trainer defaults |
| **G1 gpc** | The Dollhouse G1 gpc steering implementation (`DollhouseRobotics/g1-gpc`, `src/dollhouse_g1/steering.py`, `SteeringConfig`, single-GPU recipe) |
| **BM** | BeyondMimic `whole_body_tracking@9db9158`: tracking configuration, reward alignment and G1 actuation |
| **MJLab** | `mujocolab/mjlab@dcdf6ce`, the pinned simulation dependency |
| **AMP** | Peng et al. 2021, adversarial motion priors |
| **ER** | Extreme-RGMT (arXiv:2607.20110v1), Table II: four additional dynamics randomization ranges |

Only Extreme-RGMT's four dynamics randomization ranges are used; its
broader friction range, pushes, command perturbations and refinement stage
are not. BlinkMimic and InstantTrack are cited only where
an engineering choice follows their code.

Every value is one of three kinds, and comments in the code use the same
words:

- **Paper / Spec**: a published setting (SM, FSQ, AMP, ER) or an accepted
  specification decision derived from it.
- **Source / GPC-code**: an upstream implementation default adopted where the
  papers are silent (GPC-code for Stage 2, G1 gpc for joystick RLFT).
- **Choice / Decision**: our engineering choice, recorded in this document.

SuperMimic uses a continuous latent. We keep its model-based optimization
and replace the latent with a discrete FSQ code, so that Stage 2 can model
the tracker's latent as tokens.

## Motion data

`gpc-prepare-motions` downloads pinned private Hugging Face dataset revisions,
checks every file against the dataset's `SHA256SUMS`, and converts them to a
prepared directory (`qpos.npy`, `qvel.npy`, `clips.json`, `manifest.json`).

| Key | Dataset |
| --- | --- |
| `bones-50h` (default) | `dollhouserobotics/bones-seed-50hrs-dollhouse-g1@7dae6da` |
| `bones-15h-basic` | `dollhouserobotics/dollhouse-g1-bones-15hrs-basic@940f717` |
| `bones-advanced` | `dollhouserobotics/dollhouse-g1-bones-advanced@230a558` |
| `bones-floor` | `dollhouserobotics/dollhouse-g1-bones-floor@164c84a` |
| `g1-getups` | `dollhouserobotics/dollhouse-g1-getups@0079f8c` |
| `g1-crawl` | `dollhouserobotics/dollhouse-g1-crawl@f13d344` |

Several `--datasets` keys build one union; clip names must be unique and
categories are prefixed with the dataset key. The pinned revisions are the
cleaned ones: clips with involuntary convulsing, acted shaking or one-frame
joint snaps of at least 1.5 rad were removed upstream.

**Source-format probe.** Every dataset stores packed 30 Hz ProtoMotions
tensors for the 27-joint robot:

- `dps` holds joint positions in canonical MJCF tree order (head pan/tilt
  after the waist); they are reordered by name into action order (neck last).
- `grs` holds XYZW quaternions; they are stored as WXYZ.
- `gts[:, 0]` is the pelvis origin. Forward kinematics from root and joints
  reproduces the stored body positions to 2.4e-7 m.
- The stored velocities are forward differences that do not equal link-origin
  velocities (errors up to several m/s or rad/s).

We therefore resample generalized coordinates to the 50 Hz policy rate (linear
interpolation for positions and joints, slerp for orientation) and derive all
velocities from them: central differences for linear and joint velocities and
body-frame quaternion-log differences for angular velocity. Body poses come
from forward kinematics of the same coordinates.

**Split.** Every clip is a training clip (`VAL_PERCENT = 0`); every stage
trains on all data. The source-group hash rule (a take and its `_M` mirror
share a group) is kept so a nonzero percentage would still keep pairs
together. Evaluation clips, switcher references and standing poses come from
the training clips.

**Sampling unit.** The motion library divides each clip into one-second bins
(50 frames); a start draws a uniform frame within a bin. Reference lookups past
a clip end hold the final pose with zero velocity (encoder windows only).

## Robot and environment

### Robot and actuation

The canonical MJCF comes from the pinned `dollhouse-robots` package
(`DollhouseRobotics/robots@62aebbf`), verified by SHA-256. Placeholder
actuators are replaced by MJLab position actuators in the canonical
27-actuator order, which is also the action order. Canonical passive joint
damping (0.05) and friction loss (0.2) are kept.

| Joint group | Kp | Kd | Effort (N m) | Action scale (rad) | Source |
| --- | --- | --- | --- | --- | --- |
| Hip pitch/yaw, waist yaw | 40.179 | 2.558 | 88 | 0.5475 | BM |
| Hip roll, knees | 99.098 | 6.309 | 139 | 0.3507 | BM |
| Ankle pitch/roll, waist pitch/roll | 28.501 | 1.814 | 50 | 0.4386 | BM |
| Shoulders, elbows, wrist roll | 14.251 | 0.907 | 25 | 0.4386 | BM |
| Head pan, head tilt | 20 | 1.068 | 8 | 0.1 | Choice (provisional, needs hardware identification) |

BM gains use a 10 Hz natural frequency, damping ratio 2 and the BM motor-group
armatures; the action scale is 0.25 × effort / Kp. The joint target is
`default_pose + action_scale × action`; the default pose is BeyondMimic's
knees-bent pose (zero for the neck). Soft joint limits are 0.9 of each
physical range (MJLab G1 default).

Foot contact uses MuJoCo's default critically damped contact (solref 0.02 s,
damping ratio 1), condim 3 and priority 1 on the seven foot capsules per
foot; other collisions use condim 1. An underdamped mapping of BeyondMimic's
restitution made standing feet chatter and slide, so no restitution target is
used (Choice).

### Simulation

| Setting | Value |
| --- | --- |
| Physics / policy rate | 200 Hz (dt 0.005 s) / 50 Hz, decimation 4 |
| Solver | 10 iterations, 20 line-search iterations |
| Contact and constraint budgets | `nconmax` 64 per world (pooled), `njmax` 512 |
| Terrain | Flat plane |
| Environment | MJLab `ManagerBasedRlEnvCfg` with no reward, observation or termination terms, `auto_reset=False`, `scale_rewards_by_dt=False` |

MJLab owns the scene, simulation, actuators, action application, reset
events and the motion-command clock. Observations, rewards and failures are
stateless functions evaluated by the trainer on the clean generalized state,
so simulator and imagined rollouts share one implementation.

**Reward units.** MJLab's reward manager multiplies rewards by the control dt
by default. Our tracking reward and all CoLA task rewards are per-step
weighted sums without a dt factor; the environments declare no reward terms
and disable dt scaling.

### Physical domain randomization

All parameters are drawn independently and uniformly and stay fixed within an
episode. Foot friction and torso COM are drawn once per world when the
environment is constructed; all others are drawn at every episode reset
(BlinkMimic lifetimes, Choice).

| Parameter | Scope | Range | Source |
| --- | --- | --- | --- |
| Foot friction (absolute, all foot geoms) | per env, per world | [0.3, 1.2] | SM Table A.3 |
| Joint calibration bias | per actuator, per reset | [−0.01, 0.01] rad | SM Table A.3 |
| Torso COM offset (torso axes) | per env, per world | x [−0.025, 0.025], y/z [−0.05, 0.05] m | SM Table A.3 |
| Torso added mass | per env, per reset | [−3, 6] kg, inertia scaled by the mass ratio | ER Table II |
| Motor strength scale | per actuator, per reset | [0.8, 1.2] | ER Table II |
| PD gain scale (common to Kp and Kd) | per actuator, per reset | [0.8, 1.2] | ER Table II |
| Armature scale | per actuator, per reset | [1.0, 1.05] | ER Table II |
| Mid-episode pushes | — | none | SM |

MuJoCo position actuators compute `clamp(kp (u − q) − kd q̇, ±F)`. Writing
`s·g·kp`, `s·g·kd` and `±s·F` yields `s · clamp(g kp (u − q) − g kd q̇, ±F)`:
the randomized-gain PD torque clipped to the nominal effort limit, then scaled
by strength. Action scales stay nominal. Calibration bias enters the PD error
through MJLab's `JointPositionAction` and the sensed joint positions;
rewards, failures and replay use true coordinates.

The realized parameters form a 113-scalar context (friction, added mass, COM,
then bias, strength, gain and armature per actuator), normalized to [−1, 1]
over the ranges. The world model and critic receive it; the policy does not.

### Observation and action contract

Each 50 Hz sensor frame has 87 scalars, in this order:

| Channels | Width | Additive uniform noise half width | Source |
| --- | --- | --- | --- |
| Joint position (true + calibration bias) − default pose | 27 | 0.01 rad | SM Table A.1 |
| Joint velocity | 27 | 0.5 rad/s | SM Table A.1 |
| Pelvis IMU angular velocity | 3 | 0.2 rad/s | SM Table A.1 |
| Gravity in the pelvis frame | 3 | 0.05 | SM Table A.1 |
| Previous executed normalized action | 27 | none | SM Table A.1 |

There is no base linear velocity or position: no policy input needs state
estimation. Noise is drawn fresh per channel and step and kept
unchanged when the frame moves into the history. The decoder history is the
last 10 frames, oldest first (870 scalars); at reset it is the first frame
repeated with zero previous action. The previous action is the executed
command after collection noise and clipping.

The action is 27 normalized joint setpoints, clipped to ±5 in collection,
imagination and deployment alike as a numerical guard (Choice). The neck's
±5 clip at action scale 0.1 limits head pan from the policy to ±0.5 rad.

Generalized state (67): env-local pelvis position, WXYZ quaternion, world
linear velocity, pelvis-frame angular velocity, 27 joint positions and 27
joint velocities in action order.

### Reference window and offsets

The encoder sees ten future reference frames at policy offsets
`0, 5, …, 45` (0–900 ms; CQ recipe, SuperMimic public-code spacing). Each
frame has 69 scalars (690 total): torso position (3), torso orientation as the
first two rotation-matrix columns (6), torso linear and angular velocity
(3 + 3), reference joint positions and velocities (27 + 27).

The window uses BeyondMimic alignment on `torso_link`: one transform from the
current robot and reference torso poses (robot torso XY, reference torso Z,
yaw-only rotation of robot-anchor × inverse reference-anchor) is applied to
every frame, not recentred per frame, and the result is expressed relative to
the actual torso in robot-yaw axes. Applying the alignment to a future window
is our adaptation. Reward and failure targets are always frame +1.

### Resets, failures and timeouts

Tracking resets write the reference state at a sampled start, perturbed
uniformly (BM): base x/y ±0.05 m, z ±0.01 m, roll/pitch ±0.1 rad, yaw
±0.2 rad, joints ±0.1 rad (clamped to limits), linear velocity x/y ±0.5,
z ±0.2 m/s, angular velocity roll/pitch ±0.52, yaw ±0.78 rad/s.

| Boundary | Rule | Bootstrap |
| --- | --- | --- |
| Physical failure (SM) | \|pelvis z − reference z\| > 0.25 m, or \|gravity-z difference in the pelvis frames\| > 0.8, or a non-finite state | No |
| Clip completion | The next frame is the last clip frame | No (finite task) |
| Timeout (BM) | 10 s from the reset | Yes, from the pre-reset state |

Failure takes precedence over clip end, and clip end over timeout. The three
are stored separately in replay. Evaluation and side rollouts use separate
environment instances and never touch training episodes.

## Stage 1 tracker

`Stage1Trainer` implements SM Algorithm A.1 with a 16×7 FSQ tracker. Each
iteration:

1. Collect 2 or 3 vector steps (UTD 0.4, alternating) with the current
   tracker plus Gaussian action noise (σ 0.1), storing executed actions.
2. Fit the world model on 10 batches of 2048 eight-step autoregressive replay
   windows.
3. Imagine 4096 rollouts of 32 steps from replay states through the frozen
   world model and maximize the survival-weighted, critic-bootstrapped return
   through the decoder, FSQ straight-through estimator and encoder.
4. Fit the critic to fixed λ-return targets of those rollouts: 5 epochs of 4
   shuffled minibatches.

The first 64 vector steps only collect; normalizers are fitted on them. There
is no model-only warm-up.

### Architecture

| Network | Structure | Source |
| --- | --- | --- |
| Encoder | 690 → 2048 → 1024 → 512 → 256 → 16, ELU | SM Table A.4 widths; 16-scalar latent is Choice |
| FSQ | `round(3 tanh(e))`, integers −3…3, straight-through; decoder input `codes / 3` | FSQ with 7 levels; scaling is Choice |
| Decoder | (870 history + 16 latent) → 2048 → 1024 → 512 → 256 → 27, ELU; output layer scaled by 0.01, zero bias | SM Table A.4; init Choice |
| Dynamics | (64 local state + 27 action + 113 context) → 3 × 2048, LayerNorm + SiLU → 33 | SM Table A.4; activation Choice |
| Critic | 901 → 256 → 128 → 1, ELU; output × 100 × `reward_scale` | SM Table A.4; scale Choice |

Activation, normalization and initialization are not published; the choices
above (ELU for encoder, decoder and critic; LayerNorm and SiLU for dynamics;
PyTorch default initialization otherwise) are ours. Rounding is
half-to-even built from Floor/compare operations, bit-identical to
`torch.round`, because the Miniverse ONNX runtime rejects `Round`.

Inputs use fixed affine normalizers (mean/std, std floor 1e-2) fitted once on
the initial fill: encoder features, decoder history (previous-action channels
unscaled), dynamics local state, velocity deltas and full-state errors (std
floor 1e-4), and critic features (context and previous action unscaled).

The encoder and FSQ stay FP32 so rounding assigns the same codes as the
deployment graph.

### World model

Choice following SuperMimic's public code: the dynamics MLP predicts 33
velocity deltas (pelvis linear and angular velocity in the current pelvis
frame, then 27 joint velocities). Root position, quaternion and joint
positions integrate analytically with the current velocity plus 0.625 of the
delta at dt 0.02 s (`(decimation + 1) / (2 decimation)`); angular velocity is
transported into the next pelvis frame. No pose correction is learned. The
loss is the per-step mean squared 66-channel pose/velocity error, normalized
by the fitted full-state error std and summed over the 8-step window. Windows
never cross a reset or a collection gap.

Rewards, observations and termination are analytic, not learned. Termination
in imagination is BlinkMimic's soft failure probability:
`1 − (1 − σ((dz − 0.25) / 0.02)) (1 − σ((dg − 0.8) / 0.04))`.

### Imagination and critic

- Sensor noise is drawn outside the model and appended to the carried real
  history; old noise is never redrawn.
- An imagined trajectory ends at clip completion (termination weight 1, later
  steps masked); 10 s timeouts are not modeled.
- Actor loss: negative mean imagined return divided by the horizon.
- Critic input (901): local state (64), normalized context (113), previous
  action (27), encoder features (690), original-global torso anchor
  position/orientation errors in robot-yaw axes (6) and clip frames remaining,
  saturated at 500 and scaled to [0, 1].
- Critic targets: SM Eq. 3 λ-returns (γ 0.99, λ 0.95) computed before fitting
  and held fixed.

### Rewards

SM Table A.2 per step, no dt factor, maximum unpenalized value 5:

| Term | Weight | Kernel |
| --- | --- | --- |
| Body position (15 bodies, aligned) | 1.0 | exp(−mean ‖Δp‖² / 0.3²) |
| Body orientation (aligned) | 1.0 | exp(−mean angle² / 0.4²) |
| Body linear velocity (world) | 1.0 | exp(−mean ‖Δv‖² / 1.0²) |
| Body angular velocity (world) | 1.0 | exp(−mean ‖Δω‖² / 3.14²) |
| Anchor position (global) | 0.5 | exp(−‖Δp‖² / 0.3²) |
| Anchor orientation (global) | 0.5 | exp(−angle² / 0.4²) |
| Action rate | −0.1 | ‖a − a_prev‖² (normalized actions) |
| Joint limits | −2.0 | Σ soft-limit violation |

The 15 tracked link origins are pelvis, torso, head tilt link, and left/right
hip roll, knee, ankle roll, shoulder roll, elbow and wrist hand links. There
are no contact, energy, torque or alive terms in this base reward.

CQ recipe choices:

- The reward aligns and anchors on the pelvis (`reward_anchor = 0`); encoder
  alignment stays on the torso.
- Real collection uses `0.02 ×` the base reward. Imagination uses
  `0.02 × (base + contact bonus)`.
- **Contact reward** (imagination only, BlinkMimic-derived, weight 0.25):
  `0.25 × mean_feet(1 − (c_sim − c_ref)² + c_ref exp(−‖v_xy‖² / 0.3²))` with
  `c_sim = σ((0.045 − z_ankle) / 0.005)` on the ankle-roll link origins.
  Reference foot contact is calibrated per complete clip: ankle height below
  the clip's 5th percentile + 0.015 m and planar speed below 0.5 m/s.
- The return sampler normalizes the collection reward by its scale, so its
  units are those of the unscaled reward. The actor learning-rate trigger uses
  the unscaled torso-anchored reward as a diagnostic
  (`episodes/diagnostic_normalized_return`).

### Optimization

| Setting | Value | Source |
| --- | --- | --- |
| Envs / replay | 4096 per GPU / 1000 steps per env, FIFO | SM |
| Model window, batches, batch size | 8 / 10 / 2048 | SM Table A.5 |
| Actor batch / horizon | 4096 / 32 | SM Table A.5 |
| Critic fitting | 5 epochs × 4 minibatches | SM Table A.5 |
| Collection noise / UTD | σ 0.1 / 0.4 | SM Table A.5 |
| World model and actor optimizer | Muon on hidden Linear weights, Adam on the rest | CQ, BlinkMimic `optim.py` |
| Critic optimizer | Adam | SM |
| Base LR | 4e-4, cosine to 0.03 × over 250,000 iterations, then constant | CQ; SM schedule |
| Actor LR multiplier | 4×, decaying linearly to 1× over 300 updates once diagnostic normalized return ≥ 0.75 or at update 1400 | CQ |
| Adam betas | actor (0.9, 0.95); world and critic (0.9, 0.999); eps 1e-8, no weight decay | SM / Choice |
| Muon | momentum 0.95 (Nesterov), 5 Newton–Schulz steps, no weight decay, `match_rms_adamw` LR scaling | CQ |
| Gradient clipping | global norm 1.0 everywhere; actor also elementwise ±0.1 | SM Table A.5 |
| Policy EMA | decay 0.995 per update; used for inference, export and as the Stage 2 teacher | CQ |
| γ / λ | 0.99 / 0.95 | SM Table A.5 |
| Length | 320,000 iterations (`--iterations` sets the run budget) | SM Table A.5 |

Muon excludes each MLP's first and last Linear weight.

### Reset-start sampling

SM Appendix D.1 bin sampler, with unpublished details chosen by us: a bin's
estimate is the EMA (rate 0.25) of the normalized discounted return of
episodes that started in it, divided by the largest return attainable over
that episode's available horizon; unvisited bins start at 0 (hardest). The
base mixture is 50% uniform over bins, 40% softmax(−25 × estimate), 10%
accumulated history of that softmax.

CQ focus: 30% recent-failure focus is mixed into the base, then 15% EMA-probe
focus into the result (59.5 / 25.5 / 15% once both have failures; empty focus
falls back to its input).

- Recent failures fade by 0.999 per nonempty batch of endings and update hit
  bins with 0.9 old + 0.1 mean(failed); sampling shifts one bin earlier within
  the clip.
- Probe sweeps start at update 1000 and repeat every 125 updates: the frozen
  EMA policy, nominal physics and no sensor noise in a separate environment,
  one tile per 300 reference frames with a random lead of 200–450 frames,
  20 side steps per training update. Counts fade by 0.7 per completed sweep
  and spread over the failure bin and ten preceding bins. The probe
  environment has one world per tile, so it grows with the corpus.

### Engineering throughput choices

- The imagined step is fused with `torch.compile`.
- With `cuda_graphs`, each model update and each actor+critic update replays
  one CUDA graph (sampling included); per-step encoder features, rewards and
  reset-time `set_const` replay from bitwise-equal graphs.
- The decoder and dynamics MLPs run under BF16 autocast inside training
  updates only; collection, evaluation and export are FP32.
- Model and policy updates draw from their own forked RNG streams, so
  collection randomness does not depend on them.

### Evaluation

In-training evaluation every 500 iterations runs 512 clips from frame 0 for
up to 500 steps in a separate environment (training randomization and sensor
noise, no reset perturbation, no action noise) under a forked RNG. It reports
tracking errors, returns, failures, one-, 8- and 32-step world-model errors on
these rollouts, and joint-target jitter (RMS physical target rate, and after
a second-order 5 Hz Butterworth high-pass, excluding 0.5 s after reset). A
digest of the training state before and after is logged as
`eval/training_state_preserved`. `gpc-evaluate-stage1` repeats this from a
run's `resume.pt` after seeding nonzero episode progress, or runs the
standing check alone (`--standing`).

At every policy checkpoint a standing check holds five quiet upright poses
for 10 s in native MuJoCo, clean and with sensor noise, and reports falls,
drift, joint error and target rates. These are diagnostics, not rewards.

### Checkpoints

Every 1000 iterations: a numbered `checkpoints/policy_*.pt` (EMA and live
tracker, config) that is exported and published, and `resume.pt`
(networks, normalizers, optimizers, sampler, replay, RNG), overwritten in
place. Exports run in 2 background processes. On resume, live simulator
episodes restart with fresh segment ids and replay windows break across the
gap.

## Stage 2 prior

### Stage 2 prior: accepted decisions

Precedence: decisions here, then GPC-code for missing details. The Stage 1 task contracts (robot, sensing,
randomization, alignment) are unchanged.

**Teacher.** A Stage 1 tracker `policy_*.pt` file (EMA weights), frozen:
its encoder labels the aligned reference window, its decoder drives the
robot. `--teacher-status` is `smoke` or `stage1` and is checked against the
file; a smoke tracker cannot run as `stage1`. A run directory refuses a
different teacher on resume.

**Tokens.** The 16 seven-level codes are packed into 4 tokens of 4
consecutive scalars, little-endian mixed radix over `code + 3`
(GPC-code `FSQTokenization`): token `g = Σ_i (code[4g + i] + 3) · 7^i`,
vocabulary 7⁴ = 2401. Unpacking uses float Floor arithmetic so it exports to
ONNX.

**Prior.**

| Setting | Value | Source |
| --- | --- | --- |
| Width / feed-forward | 1024 / 4096 (the only accepted width) | Spec, GPC Table 10 |
| Layers / heads | 3 / 8 | Choice |
| Context | One current 87-scalar sensor frame; no past-step tokens | Spec |
| Observation normalization | Running mean/variance, statistics EMA 0.999, clamp ±5, eps 1e-5, float64 buffers; updated per training minibatch only | GPC-code |
| Observation encoder | Linear(87, 1024) → GELU → Linear(1024, 1024) | GPC-code |
| Token embedding | Biased linear over one-hot tokens (column lookup) | GPC-code |
| Sequence | `[context, token0, token1, token2]` + learned positions; position k predicts token k | GPC-code |
| Transformer | Post-LayerNorm causal encoder, GELU, dropout 0.1 in training | GPC-code |
| Sampling | Temperature 1.0, nucleus top-p 0.9, renormalize, discrete draw, token by token | Spec, GPC-code |

The nucleus keeps a token when the probability mass ranked strictly before it
is at most top-p (ties rank the lower index first). Each draw inverts the kept
CDF in vocabulary order with an explicit uniform, so the Miniverse controller
owns the randomness. The export ranks with `TopK` over 2400 tokens instead of
a full sort (exact for top-p < 1 − 1/2401).

**Optimization and collection.**

| Setting | Value | Source |
| --- | --- | --- |
| Optimizer | AdamW, LR 1e-4 constant (no warm-up, not scaled with GPUs), betas (0.9, 0.999), eps 1e-8, weight decay 0.01 | GPC-code |
| Loss | Teacher-forced cross-entropy over the 4 positions, label smoothing 0.01 | GPC-code |
| Gradient clipping | Global norm 100 | GPC-code |
| Model-weight EMA | None | GPC-code |
| Envs / minibatch | 1024 / 1024 per GPU | GPC-code example |
| Rollout | 64 fresh real-simulator steps per iteration, one shuffled pass (64 minibatches) | GPC-code |
| Collection noise | σ 1e-4 on the teacher action (normalized units) | GPC-code |
| Reset sampling | Uniform over one-second bins (the teacher does not adapt) | Choice |
| Evaluation interval | 200 iterations | GPC-code |
| Latest interval | 1000 iterations: `resume.pt` and `checkpoints/latest.pt` (GPC-code uses 10) | Choice |
| Numbered checkpoint interval | 1000 iterations | GPC-code |
| Transition ceiling | 10¹³, a launcher ceiling, never a budget; `--iterations` is the budget | GPC-code |

The prior input is the newest frame of the teacher's decoder history: the
noisy frame with calibration bias and the executed previous action. Episodes
continue across iterations; each iteration's buffer is discarded after its
pass. There is no replay, world model, critic or PPO loss.

**Execution.**

- Collection runs serially in the trainer process: collect one rollout, then
  optimize one pass. Exports run inline in the trainer.
- Multi-GPU training is plain `torchrun` data parallelism (DDP): one rank per
  GPU, 1024 environments and 1024-example minibatches per rank, averaged
  gradients, observation statistics from all-reduced minibatch moments.
- Throughput choices that change only rounding, never the recipe:
  - `tf32_prior`: TF32 matmuls in prior optimization only; teacher,
    simulator-side code, evaluation and export stay FP32.
  - `bf16_linears`: the transformer feed-forward, output head and
    observation-encoder Linears run in BF16 in optimization only; attention,
    LayerNorm, residuals, embedding and loss stay FP32/TF32; weights,
    optimizer state and checkpoints stay FP32.
  - `attention_math`: training computes the 4-position attention directly in
    exact FP32 instead of the fused SDPA kernel.
  - `compile_prior`: `torch.compile` of the training loss.
  - `cuda_graphs`: bitwise-equal CUDA-graph replay of per-step environment
    calculations.
  - `compile_rewards`: `torch.compile` of the diagnostic-only real reward
    terms.

### Prior evaluation

Every 200 iterations, in a separate environment under a forked RNG:

1. Teacher-driven rollouts without action noise (64 clips, 500 steps) give
   held-out (frame, token) pairs: the prior's eval-mode cross-entropy,
   perplexity and accuracy per position.
2. Closed-loop generation through the frozen decoder: survival, fall rate
   (pelvis below 0.4 m), SuperMimic reference failures and tracking errors as
   diagnostics, code agreement with the teacher and code statistics.

**Quality evaluation.** A prior is judged by closed-loop unconditional
sampling, with GPC's push impulses (Table 9, Appendix C.3),
not by token loss. Every `quality_interval` (1000) iterations, and with
`gpc-evaluate-stage2-quality` for any checkpoints, 32 start clips × 8 samples
run 20 s from frame 0 with nominal physics, free and after 2.4 and 9.8 m/s
pushes at 4 s, over seeds 0…n−1 (`quality_seeds` 2) with push directions
identical across checkpoints. It reports pose and root APD (first 10 s), the
fallen-and-stuck rate (pelvis below 0.45 m or uprightness below 0.6 for 50
frames while moving less than 0.2 m), survival, normalized jerk (whole body
and feet, 0.4 s windows) and mean acceleration. These are measurements, not
acceptance gates.

`gpc-evaluate-stage2` repeats the in-training evaluation from `resume.pt` or
a policy file after seeding nonzero episode progress, and optionally builds
and checks a local export.

## Joystick RLFT

Joystick RLFT trains CoLA adapters on the frozen Stage 2 prior and its frozen
Stage 1 decoder, porting the G1 gpc steering task and trainer to MJLab and our
prior. Values are G1 gpc single-GPU settings unless marked.

### Policy

- Every transformer layer gets a DoRA residual whose rank-space activations
  are FiLM-modulated by the task input:
  `layer(x) + m · normalize(scale · FiLM(x A, c) B)`, rank 32, α 64
  (scale α/rank), `A ~ N(0, 1/d)`, `B ~ N(0, 1e-8)`, FiLM γ = 1, β = 0 at
  start (upstream ProtoMotions `TransformerLayerWithDoRA`, reimplemented).
  `m` starts at zero, so the initial policy is exactly the base prior.
- The prior's observation encoder, embeddings, positions, head and statistics,
  and the decoder, never change; only adapters and critic are optimized.
- Tokens are drawn from the full-support temperature-1 categorical, not the
  Stage 2 nucleus, so PPO ratios are exact. Draws invert the vocabulary-order
  CDF with an explicit uniform.
- The adapter input is the 7-D local command (below). The prior input is the
  newest noisy 87-scalar frame, as in Stage 2.
- Critic: four 1024-wide ReLU layers on the prior-normalized clean frame, the
  command and 7 privileged features (local planar and vertical base velocity,
  pelvis height, fallen-step fraction, command and gaze clocks).

### Task

| Component | Value | Source |
| --- | --- | --- |
| Reset | Default standing pose at 0.76 m, uniform yaw, ±0.02 rad joint noise, Stage 1 randomization | G1 gpc |
| Command | Velocity (x forward, y left) in the commanded facing frame plus facing; expressed in robot-yaw axes | G1 gpc |
| Command sampling | Speed U[0.2, 1.5] m/s, uniform direction; facing changes with probability 0.5 to a uniform angle and anchors at the current yaw; moving commands last 100–250 steps | G1 gpc |
| Stop probability | 0.3 (G1 gpc 0.1) | Decision (idle gaze) |
| Episodes | 1000 steps (20 s), bootstrapped timeout | G1 gpc |
| Termination | 150 consecutive fallen steps (pelvis below 0.45 m or uprightness below 0.6), a joint 1 rad beyond its range, or divergence | G1 gpc |
| KL | Rollout reward minus 0.1 × the prefix-summed KL(adapted ‖ base) at sampled prefixes; PPO adds its direct derivative | G1 gpc |

Reward per step, summed without dt scaling:

| Term | Weight | Form |
| --- | --- | --- |
| Facing | 1.0 | exp(cos(facing error) − 1) |
| Direction | 1.0 | exp(cos(velocity direction error) − 1), zero when commanded or measured speed < 0.1 m/s |
| Speed | 1.0 | exp(−0.5 ((speed − commanded) / 0.5)²) |
| Action rate | −0.1 | ‖physical joint-target change‖ (rad) |
| Joint limits | −10 | Σ soft-limit violation |
| Contact | −0.1 | count of non-foot bodies above 1 N net contact |
| Gaze, still, foot slip, arm pose | see Idle gaze | Decision |

Evaluation every 100 iterations: 128 envs for 400 steps from seeded resets in
a separate environment, G1 gpc success thresholds (alive, not fallen, speed
error < 0.3 m/s, velocity error < 0.4 m/s, facing error < 0.35 rad), plus
idle metrics such as `idle/offcenter_look_error_rad` (world gaze error on
look targets more than 0.3 rad off centre). The frozen prior is evaluated
with the same seeds at iteration 0.

### Idle gaze

Decision: at zero commanded speed the robot stands still and looks around.
Idle gaze is always on.

- **Command.** The local command is 7-D: velocity (2) and facing (cos, sin),
  plus the gaze direction's horizontal (cos, sin) in robot-yaw axes and its
  pitch (rad, positive up). The gaze is a yaw offset from the commanded facing
  and a pitch.
- **Sampling.** A zero-speed command lasts 150–500 steps (3–10 s). Every new
  command looks straight ahead. While stopped, a new look target is drawn
  every 75–200 steps: centred with probability 0.25, else yaw U[−1.0, 1.0] rad
  and pitch U[−0.3, 0.2] rad. Moving commands look straight ahead.
- **Slew.** The gaze command moves toward each new target at 1.5 rad/s, in
  training and in the controller.
- **Neck.** The frozen decoder cannot turn the neck from a standing state
  (measured: its head-pan target stays near zero for any codes, although the
  corpus has neck motion). The two neck joints therefore follow the gaze
  command analytically: pan and tilt that point the head link's x axis along
  the commanded direction, solved through the pelvis orientation and the
  waist yaw, roll and pitch joints, clamped to pan ±1.5 rad and tilt ±0.6 rad.
  These targets replace the decoder's neck outputs in training and in the
  exported model. The CoLA observes the neck at the pose its own previous
  neck action commanded, at rest, so prior and decoder never see neck poses
  outside their corpus.

| Term | Weight | Form | When |
| --- | --- | --- | --- |
| Gaze | 1.0 | exp((cos(gaze error) − 1) / 0.25), head x axis vs. world target | Every step |
| Still | 1.0 | exp(−(pelvis speed / 0.1)²) | Stopped and facing within 0.35 rad |
| Foot slip | −1.0 | per m/s of summed planar ankle-roll link speed | Same |
| Arm pose | −0.05 | per rad² of summed shoulder, elbow and wrist offsets from the default pose | Every step |

Turning in place toward a new facing is not penalized for stepping.

### PPO and the fast recipe

| Setting | Value | Source |
| --- | --- | --- |
| Envs | 4096 | G1 gpc |
| Rollout / minibatch | 16 steps / 2048 (32 updates per epoch) | Fast recipe (G1 gpc 32 / 4096) |
| Epochs | 2 | G1 gpc |
| γ / λ / clip | 0.99 / 0.95 / 0.2 | G1 gpc |
| Advantages | Centred, not scaled, so `kl_beta` keeps its meaning | G1 gpc |
| Entropy bonus | 0 | G1 gpc |
| Actor (adapter) LR | 1e-3 AdamW | Fast recipe (G1 gpc 1e-4) |
| Critic LR | 1e-4 AdamW | G1 gpc |
| Gradient clipping | Global norm 25 | G1 gpc |
| Update-KL early stop | Actor steps stop for the iteration after a minibatch's approximate KL exceeds 1.5 × 0.02 | G1 gpc |
| Budget | 500 M transitions as a cap; `--iterations` is the run budget | G1 gpc |

Physical terminations do not bootstrap; timeouts bootstrap from the pre-reset
state.

**Fast recipe throughput settings** (Choice; change only rounding or
scheduling):

- `tf32`: TF32 transformer matmuls in rollout and PPO; decoder and simulator
  side stay FP32.
- `bf16_prior`: the frozen transformer and head Linears run from BF16 weight
  copies with FP32 accumulation, in rollout and PPO alike. Adapters,
  LayerNorm, attention scores, residuals, critic, decoder, checkpoints and
  export stay FP32.
- `fused_base`: the rollout evaluates the frozen base in the same
  key/value-cached pass as the adapted policy; PPO reuses the shared first
  block's output.
- `compile`: `torch.compile` of PPO's teacher-forced logits and the token
  statistics. The CUDA policy step replays from a CUDA graph.
- Attention over at most four positions is a plain matmul/softmax instead of
  PyTorch's fused kernels, and sampling uses a per-layer key/value cache.

### Checkpoints

Every 100 iterations: `resume.pt` (adapters, critic, both optimizers,
counters, RNG and generators) and an exported, published policy; numbered
`policy_*.pt` files are kept every 500. One background export process.
`gpc-render-joystick` renders an export through its own controller with a
scripted joystick sequence.

## Get-down

Get-down trains CoLA adapters on the frozen prior so the robot gets onto its
back on the floor slowly and naturally from wherever it is. It uses the shared
CoLA trainer and fast-recipe PPO except where listed. The start mixture,
task input, KL weight, impact and regularizer terms, kernels, gates,
schedules and the AMP and pelvis terms are our choices.

### Starts

- `gpc-collect-joystick-starts` runs a trained joystick CoLA checkpoint
  (4096 envs, 1500 steps, training commands with stop probability 0.25,
  Stage 1 randomization and sensor noise), adds random planar pushes
  (probability 0.01 per env-step, 0.2–1.0 m/s) and records, every 25 steps
  after a 25-step warm-up, the states of robots that have not fallen.
  Labels: `disturbed` within 1 s of a push, else `standing` when command and
  pelvis are near rest, else `moving`. The bank names a few representative
  states for the Miniverse start switcher.
- At reset, with probability 0.5 (Choice) the env starts from a uniform bank
  state turned to a uniform heading; otherwise from a uniform one-second bin
  of the motion archive with the Stage 1 reset perturbations. The archive
  supplies starts only, never a reference.
- After a reset of every environment the episode clocks are staggered
  uniformly so timeouts do not synchronize.

### Task input and critic

The adapter input is the five newest task frames (30 scalars), padded with
the first frame at reset. A frame is torso orientation as gravity in the
torso frame (3) and torso angular velocity in the torso frame (3), computed
from the pelvis IMU and joint encoders through forward kinematics. No input
uses base position, height or linear velocity, so none needs state
estimation. The critic additionally gets the episode clock, torso forward-z
and up-z, torso, pelvis and head heights and the pelvis linear velocity in the
heading frame (9 privileged features).

### Reward

Per step, without dt scaling:

| Term | Value | Source |
| --- | --- | --- |
| Whole-body impact | −w Σ_bodies max(0, F − 327 N), w ramps 0.0002 → 0.007 | Choice |
| Torso impact | −w max(0, F − 150 N), w ramps 0.0004 → 0.014 | Choice |
| Wrist impact | −w Σ max(0, F − 100 N), w ramps 0.0004 → 0.014 | Choice |
| Action rate | −0.1 ‖physical joint-target change‖ | Choice |
| Joint limits | −10 Σ soft-limit violation | Choice |
| Pelvis height | 0.5 × clamp(1 − max(h − 0.10, 0) / (0.78 − 0.10), 0, 1) × clamp(chest-up, 0, 1) | Choice |
| AMP style | 1.0 × max(0, 1 − 0.25 (D(s, s′) − 1)²) | AMP, Choice |
| KL | −0.05 × KL(adapted ‖ unconditioned prior) | Choice |

- F is each body's largest net floor-contact force over the four physics
  substeps of the step. The impact weights ramp linearly over 500 iterations
  (Choice); impact terms are masked on the first step after a reset.
- The pelvis term is linear from the frozen prior's standing pelvis height
  (0.78 m) to the reference's lying height (0.10 m), scaled by the world-z
  component of the torso chest axis, so a squat or a face-down pose earns
  little of it.
- Lying gate (Choice): torso below 0.2 m with |torso x · world z| > 0.8;
  on the back when the chest faces up.

### AMP

- Reference: five bones-floor `spread_out_R_001` takes (`A428_M`, `A429`,
  `A429_M`, `A430`, `A430_M`) that squat, sit and roll onto the back
  smoothly, each cropped from its first frame to 1.5 s before the end of its
  first on-back stretch. `A432` (fast descent) and `A431_M` (legs kick up) are
  excluded. `--motions` must contain bones-floor.
- Frame (85): pelvis height; pelvis forward and up axes in the heading frame;
  pelvis linear velocity in the heading frame; pelvis angular velocity in the
  pelvis frame; joint positions and velocities; torso, head, hand and foot
  positions relative to the pelvis in the heading frame. The heading comes
  from the pelvis lateral axis, which stays horizontal lying on the back.
- Discriminator on consecutive-frame transitions (170): ReLU MLP 1024 → 512 →
  1, inputs standardized by running statistics of reference and policy
  transitions and clipped to ±5. LSGAN targets +1/−1, gradient penalty 10 on
  reference samples (AMP paper value), output-layer weight penalty 0.05, AdamW
  1e-4 with weight decay 1e-4, global-norm clip 25.
- Each step's style reward uses the discriminator of the collecting
  iteration. After the PPO update the rollout's policy transitions enter a
  replay of the latest 1,048,576; one epoch of 4096-sample minibatches (as
  many as the rollout has transitions) draws policy samples from it against
  as many reference draws.

### Episodes, evaluation and budget

- Episodes are 400 steps (8 s) with a bootstrapped timeout; the reference
  lies down in about 3 s. Termination only for a joint 1 rad beyond its range
  or divergence.
- Evaluation: 144 envs = 9 push conditions (none, or 0.6 and 1.0 m/s forward,
  backward, left or right) × 16 pre-fall bank states at heading zero, 400
  steps, impacts scored at the final weights. Reports peak floor, body, torso and wrist forces, peak downward
  pelvis speed, and settled lying rate, on-back rate, torso and pelvis height
  and pelvis speed over the last 1 s. The frozen prior and an approximate
  damping mode are evaluated with the same seeds at iteration 0.
- Transition cap 2000 × 4096 × 32; `--iterations` is the run budget.

## Checkpoints and Miniverse publication

Every training run, including smoke runs, is a local Miniverse run
(`miniverse-sdk` 0.9.2). The run directory holds:

- `config.json` (resolved configuration; authoritative on resume),
  `miniverse.json` (run binding, provenance: code and dependency revisions,
  dataset manifest, robot asset identity, teacher or prior identity);
- `run.json`, `base.mini` (the run's source bundle, from `bundle/source.mini`)
  and one Rerun metric file per process start under `metrics/`, plus
  `metrics.jsonl`;
- `preservation.jsonl`: training-state digests around every evaluation;
- `resume.pt` and native `checkpoints/policy_*.pt` (Stage 2 also
  `checkpoints/latest.pt`);
- `exports/iter_*` bundles with their check records, and published
  checkpoints `checkpoints/step-<N>/` with the ledger `checkpoints.json`.

Re-running the same command resumes the same run directory and Miniverse run;
a different teacher or prior is refused. View runs with
`miniverse serve` on the parent directory.

Every saved policy is exported to ONNX at the same step. The graph reuses the
training functions, so deployment and training share one numeric path. Each
export is checked for native/ONNX Runtime parity on random inputs (32 for the
prior and CoLA exports) and in a native-MuJoCo closed loop that runs the
bundle's own controller and model while recomputing the action with the
training code. The checkpoint replaces only the base bundle's ONNX model;
native resumable state stays separate. Deployment differs from training only
in nominal physics, clean sensing and no calibration bias.

| Task | Model | Controls |
| --- | --- | --- |
| Stage 1 | `tracker`: torso FK, aligned window, encoder, FSQ, decoder, action-to-target map | Motion switcher over named reference clips (`example_clips`, default chosen per category); live total and high-passed target rates |
| Stage 2 | `prior`: one step of nucleus sampling and the frozen decoder in one graph | Start-pose switcher and sampler-seed switcher (8 seeds) |
| Joystick | `policy`: adapted prior, analytic neck targets and frozen decoder | Movement joystick (≤ 1.5 m/s), facing scalar, look stick and auto-look toggle |
| Get-down | `policy`: in-graph task frames, adapted prior and frozen decoder | Start switcher over named bank states, push switcher (the evaluation's conditions), controller switcher (get down / approximate damping) |

Controllers read native MuJoCo state, keep the past observations and the
executed previous action, and draw the four uniforms per step, so they own
the sampling randomness.
