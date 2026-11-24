# super-gpc

Generative Pretrained Controller training for Dollhouse Robotics' custom
27-actuator G1, built as a downstream project of
[MJLab](https://github.com/mujocolab/mjlab). MJLab supplies the manager-based
environment and MuJoCo Warp simulation; this project owns the robot-specific
tasks and learning algorithms.

The specification is [docs/spec.md](docs/spec.md). It records sources,
accepted decisions and engineering choices for every stage.

## Training stages

- **Stage 1: tracker.** SuperMimic's learned-dynamics, differentiable
  imagined-return training of a reference encoder, a 16-scalar seven-level
  finite scalar quantization (FSQ) bottleneck and a sensor-conditioned
  action decoder, using the CQ recipe
  (contact reward, policy EMA, recent-failure and probe sampling).
- **Stage 2: generative prior.** Freeze the Stage 1 tracker and learn a
  proprioception-conditioned autoregressive distribution over its grouped
  FSQ codes from fresh expert rollouts: transformer width 1024, 3 layers,
  8 heads, nucleus sampling at top-p 0.9 and temperature 1.0.
- **Joystick CoLA.** Task-conditioned DoRA adapters with FiLM command
  modulation on the frozen Stage 2 prior, trained with PPO from a joystick
  command (velocity and facing) plus an idle gaze command.
- **Get-down CoLA.** Adapters on the same frozen prior that lower the robot
  onto its back from wherever it is, with an AMP style reward on floor
  reference clips, a pelvis-height reward and floor-impact penalties. Half
  of its resets come from a bank of joystick-policy rollout states.

## Quick start

On a bare CUDA image, first install the EGL/GL libraries MuJoCo imports and the
Python headers PyTorch's Triton kernels compile against:
`apt-get install libegl1 libgl1 libglu1-mesa python3.12-dev build-essential`.

```bash
uv sync --extra cu130        # or --extra cpu
export HF_TOKEN=...          # the motion corpora are private Hugging Face datasets
uv run gpc-prepare-motions --source data/src --output data/motions \
  --datasets bones-15h-basic bones-advanced bones-floor g1-getups

# Stage 1 tracker
uv run gpc-train-stage1 --motions data/motions --run-dir runs/stage1 --iterations <N>

# Stage 2 prior on the frozen tracker (one rank per GPU under torchrun)
uv run gpc-train-stage2 --motions data/motions --teacher runs/stage1/checkpoints/policy_<N>.pt \
  --teacher-status stage1 --run-dir runs/stage2 --iterations <M>

# Joystick CoLA, its start bank, then get-down CoLA
uv run gpc-train-joystick --prior runs/stage2/checkpoints/policy_<M>.pt \
  --run-dir runs/joystick --iterations <K>
uv run gpc-collect-joystick-starts --cola runs/joystick/checkpoints/policy_<K>.pt \
  --out data/joystick_starts.pt
uv run gpc-train-get-down --prior runs/stage2/checkpoints/policy_<M>.pt \
  --motions data/motions --starts data/joystick_starts.pt --run-dir runs/get-down --iterations <L>
```

Without `--datasets`, `gpc-prepare-motions` prepares the 50-hour corpus alone.
Get-down takes its AMP reference clips from `bones-floor`, so its motion
directory must include that dataset.

Every training command resumes when re-run with the same `--run-dir`; the
stored `config.json` is authoritative. Each script's `--help` documents its
arguments, and `--set key=value` overrides configuration fields when a run is
created. Evaluation commands: `gpc-evaluate-stage1`, `gpc-evaluate-stage2`,
`gpc-evaluate-stage2-quality` and `gpc-render-joystick`.

## Metrics and checkpoint Miniverses

All training runs, including quick validation runs, log metrics and publish
checkpoints with the `miniverse` package from
[`miniverse-sdk`](https://pypi.org/project/miniverse-sdk/) 0.9. Each `--run-dir`
is a local Miniverse run directory that `miniverse serve` shows live, with
metric graphs and checkpoints that run in the browser:

```bash
uv run miniverse serve runs
```

`serve` binds to loopback and lists the run directories directly under the
directory it is given. See [AGENTS.md](AGENTS.md) for running it as a
supervised service in an orb or forwarding it from a remote GPU host.

Every saved policy checkpoint is exported to ONNX, checked for parity against
native inference and published into the run directory as a Miniverse
checkpoint with an interactive demonstration: a motion switcher for Stage 1,
start-pose and sampler-seed controls for Stage 2, joystick and look controls
for the joystick CoLA, and a start switcher for get-down. Native resumable
checkpoints are kept separately. `miniverse push RUN_DIR` mirrors a run to a
private miniverse.bot run, only on request.

## Working on this project

Read [AGENTS.md](AGENTS.md) for the package layout, ownership boundaries and
validation policy. Do not add or commit unit tests. Validate through
reproducible execution and short training or evaluation runs, and provide the
commands, configurations, metrics and artifacts that support each claim.

## License

Apache License 2.0. See [LICENSE](LICENSE). The code and any trained models are provided as is, without warranty of any kind. Nothing here has been validated as safe on a physical robot; take every precaution when deploying a policy on hardware.
