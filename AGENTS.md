# Agent guidance

## Specification and scope

Read [the specification](docs/spec.md) before changing training behavior.
Preserve accepted decisions and distinguish paper settings, upstream-code
defaults and our adaptations. Record new agreed decisions there. The README
describes the stages; it is not a second hyperparameter source.

Implement only the work requested in the current task. Do not launch full
training, provision paid compute or modify shared infrastructure without the
required authorization and budget.

## Downstream MJLab architecture

Depend on MJLab as an external, pinned dependency. Do not vendor or fork its
simulation stack, shadow its `mjlab` namespace, or edit installed dependencies.

| Location | Responsibility |
| --- | --- |
| `pyproject.toml` and `uv.lock` | Package/dependency configuration, pinned MJLab version, CLI and task-plugin entry points |
| `src/super_gpc/asset_zoo/robots/dollhouse_g1/` | Dollhouse robot configuration on the MJCF from the pinned `dollhouse-robots` package: joint/body mapping, nominal actuation and asset provenance |
| `src/super_gpc/tasks/tracking/tracking_env_cfg.py` | Shared tracking environment construction from MJLab dataclass configs and manager terms |
| `src/super_gpc/tasks/tracking/config/dollhouse_g1/` | Dollhouse train/play environment variants and task registration, following MJLab's `env_cfgs.py` convention |
| `src/super_gpc/tasks/tracking/mdp/` | Task observations, motion commands/sampling, rewards, terminations, reset/randomization events and metrics |
| `src/super_gpc/tasks/tracking/rl/` | Stage 1 tracker and Stage 2 prior configuration, training, evaluation, export and run lifecycle |
| `src/super_gpc/tasks/steering/` | Joystick CoLA environment, commands and rewards (`mdp.py`), training, export and run lifecycle (`rl/`) |
| `src/super_gpc/tasks/get_down/` | Get-down CoLA environment and MDP terms (`mdp.py`), training, export, start bank and run lifecycle (`rl/`) |
| `src/super_gpc/rl/` | Learning components shared across stages, such as FSQ, networks, the code prior, CoLA adapters, learned dynamics and replay |
| `src/super_gpc/scripts/` | Thin train, collect, evaluate and render entry points calling package code |
| `docs/` | Authoritative spec, design decisions, dataset manifests and small validation reports |
| `runs/`, `checkpoints/`, `data/` | Untracked run outputs, checkpoints and local datasets; use durable artifact storage for evidence |

Follow the separation in MJLab's
[architecture overview](https://github.com/mujocolab/mjlab/blob/dcdf6ce519abfd2ea45e9503ae6fb650b75bb869/docs/source/architecture_overview.rst)
and [tracking task](https://github.com/mujocolab/mjlab/tree/dcdf6ce519abfd2ea45e9503ae6fb650b75bb869/src/mjlab/tasks/tracking):

- MJLab owns simulation, scenes, entities, actuators, sensors and manager
  lifecycles. Compose its `ManagerBasedRlEnvCfg` rather than duplicating the
  environment loop.
- Put stateless MDP calculations in functions. Use manager-term classes only
  for state or cached setup, with explicit reset and lifetime behavior.
- Keep robot configuration separate from task configuration. Resolve joint
  and body names once; do not spread positional indexing assumptions across
  the trainer.
- Keep environment terms independent of optimizer and network internals.
  Trainers consume explicit observations, state, reference and boundary data.
- Use MJLab's `mjlab.tasks` plugin entry-point group and task registry where
  compatible. Its default RSL-RL PPO runner is not our Stage 1 algorithm. Use
  a project-owned training entry point when needed; do not force model-based
  training or prior supervision into a PPO-shaped adapter.
- Keep CLI files thin and configuration explicit. Reuse upstream helpers when
  they satisfy the selected contract. Do not build a general framework, add
  pass-through wrappers or create catch-all utility modules.
- Verify port semantics. MJLab's reward manager applies timestep scaling by
  default, while our selected tracking reward does not. Preserve spec units,
  coordinate frames, history/noise behavior and timeout semantics explicitly.

## Validation, not unit-test code

Do not implement, add or commit unit tests in this repository. Do not create
a `tests/` tree, pytest fixtures, mock-based suites, test-only dependencies or
CI that runs a unit-test suite. Do not disguise a unit-test suite as validation
scripts. This policy applies to all stages and to future bug fixes.

Agents must instead provide irrefutable evidence for implementation claims.
That means reproducible execution and measured outcomes, not confidence,
static inspection, a clean import or an attractive recording alone.

- Prefer short, bounded training-validation runs on the actual MJLab simulator
  whenever practical and possible. Show that optimization advances and that
  real rollout behavior improves, not just imagined return or training loss.
- Exercise the affected runtime paths and failure/boundary cases. For a bug
  fix, reproduce the failure before the change and run the same scenario
  afterward where possible. Use matched seeds and controls when comparing
  algorithms, widths, checkpoints or performance.
- For each validation, record the code/dependency revisions, exact command,
  resolved config, dataset manifest, seed, hardware, run budget, expected
  result and measured result. Preserve logs and relevant curves, clips,
  checkpoint identifiers or state snapshots as reviewable evidence.
- Validate stage-specific behavior. Check real/model prediction agreement,
  encoder/FSQ gradients and code usage for Stage 1; held-out token prediction
  and closed-loop generation/recovery for Stage 2; command tracking, idle gaze
  and falls for the joystick CoLA; settling on the back, impacts and pushes for
  get-down. Validate checkpoint resume and export when those paths change.
- Use scratch probes or assertions to investigate invariants, then remove
  temporary code before committing. Reusable training, evaluation and export
  commands belong in the normal runtime package, not a parallel test system.
  Commit concise reports or manifests when useful, not generated test code,
  large checkpoints, raw motion data or secrets.
- Run relevant formatting, linting, typing and configuration checks too
  (`uvx ruff check src`, `uvx ruff format --check src`). They support runtime
  evidence; they do not replace it. Do not add a unit test framework just to
  validate a change.
- If hardware, data or budget prevents a required run, state exactly what
  remains unverified. Do not claim correctness or full validation, fabricate
  results, or spend unapproved compute to hide the limitation.

## Required Miniverse integration

Log metrics and publish checkpoints with `miniverse-sdk` 0.9, pinned in
`pyproject.toml`. It installs the `miniverse` Python package and the
`miniverse` executable. Versions 0.8 and earlier were an uploader for
miniverse.bot's retired cloud training API (`miniverse_sdk`, `TrainingRun`,
`/api/v1/training-runs`, `bundle upload`, hosted test sessions,
`MINIVERSE_API_TOKEN`). Those endpoints are gone; do not use or restore them.
Before changing the integration, run `uv run miniverse agent-help` and read its
`run-directory`, `metrics`, `bundles` and `hosting` topics, and the
[miniverse skill](https://github.com/DollhouseRobotics/miniverse/tree/main/skills/miniverse).

Every training run, including smoke and validation runs, is a local Miniverse
run. Its `--run-dir` is the Miniverse run directory
(`src/super_gpc/tasks/tracking/rl/telemetry.py`). `miniverse.init` keeps
`run.json`, `base.mini` (the run's source bundle) and one metric file per
process start under `metrics/`. Each published checkpoint is
`checkpoints/step-<N>/`. Resume reuses the same directory and base. Record
the resolved configuration, seeds, dependency/code revisions, dataset manifest
and robot asset identity in the run directory (`config.json`, `miniverse.json`).

- While training, serve the runs and give the user a URL they can open.
  `miniverse serve` binds to loopback and has no access control; never expose
  it publicly. In an orb, run it as a supervised service and share the portal
  URL it prints:
  `amp orb service start miniverse --command 'uv run miniverse serve runs --port $PORT' --portal`.
  `serve` lists the run directories directly under the directory it is given.
  When training runs on a remote GPU host, run `miniverse serve` there, forward
  its port with `ssh -N -L`, and expose the forwarded port with
  `amp orb portal <port>`.
- Log optimization, real rollout, evaluation and throughput metrics throughout
  training. Stage 1 must expose world-model prediction error, actor/critic
  losses, real tracking errors and returns, failure/episode-length statistics,
  FSQ usage/saturation and actuator saturation. `run.log` streams metrics to
  disk as they are logged; they are also appended to `metrics.jsonl`.
- Publish a Miniverse checkpoint (`run.checkpoint`) for every saved policy
  checkpoint, using the exact exported inference models at the same training
  step. A checkpoint replaces only the base bundle's named ONNX models. The base
  fixes the robot, controller, reference examples, commands and model interface
  for the whole run; changing any of them needs a new run directory. Keep native
  resumable checkpoints separately; inference exports do not replace
  optimizer/replay/RNG state.
- Build task-appropriate interactive controls. Trackers need a working motion
  switcher over several named references. The Stage 2 prior needs start-pose
  and sampler-seed controls. Joystick policies need joystick controls.
  Get-down needs a start switcher. Include frozen companion models when
  needed, such as the action decoder. Videos, static previews and stale or
  placeholder weights do not satisfy this.
- Verify export parity against native inference. Then open the published
  checkpoint in the served training view, exercise its controls and rollouts,
  and save the view's RRD recording into the run's `recordings/` as evidence.
  Record checkpoint steps, export identities and validation results, and show
  evidence from the actual published checkpoint rather than a different local
  model.
- Pushing a run to miniverse.bot is optional and only on request:
  `miniverse push RUN_DIR [--watch]` with a personal access token in
  `MINIVERSE_TOKEN`, under the Dollhouse Robotics organization
  (`--organization`) on first push. Source bundles, checkpoints and reference
  examples must remain private unless public sharing is explicitly authorized.
  Package only a small representative reference set, not the private full
  corpus, in the interactive demonstration.
- Keep credentials out of Git, logs and committed files. A pushed run's
  `.miniverse/push.json` holds a write token for that hosted run.

Keep telemetry in the training lifecycle and Miniverse export/controller code
in the export path. Do not couple stateless MDP terms to network I/O. Observe
the training-state preservation rules below during checkpoint demonstrations.

## Preserve training episodes during side rollouts

Evaluation, recording and diagnostics must never discard in-progress training
episodes. Use a separate evaluation environment or snapshot the complete
training environment and motion IDs/times, restore them in a `finally` block,
disable training-sampler side effects, and leave episode counters and sampler
slots unchanged. Preserve relevant observation history and per-environment
randomization state too.

Break replay sequences across an interrupted collection gap. Do not reset all
training environments, finish sampling or zero counters afterward. Demonstrate
preservation with a runtime validation scenario that seeds nonzero episode
progress, motion clocks, rewards and length counters, invokes the side rollout
and compares before/after state. Inspect episode-length metrics around the
first evaluation. Do not commit that scenario as unit-test code.

## Delivery

Before reporting completion, provide the executed validation commands and
decisive results, link the evidence and disclose limitations. State whether
work is local, committed or pushed. Keep generated run output out of Git and
never claim a successful training run from documentation-only changes.
