# ruff: noqa: F821
# Miniverse controller template for the Stage 2 generative prior. The exporter
# prepends constants (START_LABELS, RESET_STATES, JOINT_NAMES, FREE_JOINT,
# OBS_DIM, ACTION_DIM, HISTORY, NUM_TOKENS); Miniverse supplies np, ControllerInitialState
# and StepResult. Not imported by the package.
#
# Each 50 Hz step runs the exported "prior" model once: it samples the four
# grouped FSQ tokens from the current sensor frame with nucleus sampling
# (top-p 0.9, temperature 1.0) using four uniforms drawn here, and the frozen
# teacher decoder inside the same graph maps those codes and the ten-frame
# history to joint targets. The controller reads native MuJoCo state, owns the per-episode
# sampler RNG (seeded from the selected seed and start pose), keeps the nine
# past observations and the executed previous action.


class Policy:
  physics_hz = 200
  policy_hz = 50
  publish_hz = 50
  control_loop = "policy-then-decimation"

  def initialize(self, context):
    self.model = context.models["prior"]
    self.last_time = -1.0
    self.last_key = None
    self.step_index = 0
    self.rng = np.random.default_rng(0)
    self.past = np.zeros((1, (HISTORY - 1) * OBS_DIM), dtype=np.float32)
    self.prev = np.zeros((1, ACTION_DIM), dtype=np.float32)
    self.last_uniform = np.zeros((1, NUM_TOKENS), dtype=np.float32)
    self.last_tokens = np.zeros((1, NUM_TOKENS), dtype=np.float32)

  def _choice(self, commands, key, count):
    value = int(float(commands[key][0]))
    if value < 0 or value >= count:
      raise ValueError(key + " out of range")
    return value

  def _state(self, index):
    return ControllerInitialState(root_body_id="pelvis", **RESET_STATES[index])

  def initial_state(self):
    return self._state(0)

  def reset_state(self, commands):
    return self._state(self._choice(commands, "start", len(RESET_STATES)))

  def robot_state(self, data):
    positions = np.asarray(data.joint_positions, dtype=np.float64).reshape(-1)
    velocities = np.asarray(data.joint_velocities, dtype=np.float64).reshape(-1)
    root = positions[data.model.joint_slice(FREE_JOINT, velocity=False)]
    twist = velocities[data.model.joint_slice(FREE_JOINT, velocity=True)]
    if root.size != 7 or twist.size != 6:
      raise ValueError("native MuJoCo free-joint state required")
    q, qd = [], []
    for name in JOINT_NAMES:
      q.append(positions[data.model.joint_slice(name, velocity=False)][0])
      qd.append(velocities[data.model.joint_slice(name, velocity=True)][0])
    state = np.concatenate((root, twist, np.asarray(q), np.asarray(qd)))
    if not np.isfinite(state).all():
      raise ValueError("nonfinite generalized state")
    return state.astype(np.float32).reshape(1, 67)

  def step(self, step):
    data = step.sim_data
    start = self._choice(step.commands, "start", len(RESET_STATES))
    seed = int(float(step.commands["seed"][0]))
    key = (start, seed)
    time = float(data.simulation_time)
    reset = step.policy_index == 0 or time <= self.last_time or key != self.last_key
    if reset:
      self.prev = np.zeros((1, ACTION_DIM), dtype=np.float32)
      self.rng = np.random.default_rng([seed, start])
      self.step_index = 0
    state = self.robot_state(data)
    reset_flag = np.asarray([[1.0 if reset else 0.0]], dtype=np.float32)
    uniform = self.rng.random((1, NUM_TOKENS)).astype(np.float32)
    outputs = self.model.run(
      {
        "robot_state": state,
        "past_obs": self.past,
        "prev_action": self.prev,
        "reset": reset_flag,
        "uniform": uniform,
      }
    )
    codes = np.asarray(outputs["fsq_codes"], dtype=np.float32).reshape(1, 16)
    target = np.asarray(outputs["position_targets"], dtype=np.float32).reshape(27)
    action = np.asarray(outputs["action"], dtype=np.float32).reshape(1, ACTION_DIM)
    history = np.asarray(outputs["history"], dtype=np.float32).reshape(1, HISTORY * OBS_DIM)
    if not (np.isfinite(target).all() and np.isfinite(action).all()):
      raise ValueError("nonfinite policy output")
    self.past = history[:, OBS_DIM:].copy()
    self.prev = action.copy()
    self.last_uniform = uniform
    self.last_tokens = np.asarray(outputs["tokens"], dtype=np.float32).reshape(1, NUM_TOKENS)
    self.last_time, self.last_key = time, key
    self.step_index += 1
    tokens = self.last_tokens[0]
    return StepResult(
      actuation=target,
      metrics={
        "start_index": float(start),
        "sampler_seed": float(seed),
        "generated_step": float(self.step_index),
        "pelvis_height_m": float(state[0, 2]),
        "token_0": float(tokens[0]),
        "token_1": float(tokens[1]),
        "token_2": float(tokens[2]),
        "token_3": float(tokens[3]),
        "code_abs_mean": float(np.mean(np.abs(codes))),
        "action_rms": float(np.sqrt(np.mean(action * action))),
      },
    )
