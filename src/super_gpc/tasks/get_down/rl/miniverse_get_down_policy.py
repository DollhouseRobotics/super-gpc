# ruff: noqa: F821
# Miniverse controller template for the get-down CoLA policy. The exporter
# prepends constants (START_LABELS, RESET_STATES, PUSH_LABELS, PUSHES,
# CONTROLLERS, JOINT_NAMES, FREE_JOINT, OBS_DIM, ACTION_DIM, HISTORY, NUM_TOKENS,
# PAST_TASK_DIM, TASK_FRAME_DIM); Miniverse supplies np, ControllerInitialState
# and StepResult. Not imported by the package.
#
# An episode starts from the selected joystick-rollout state with the selected
# push added to the pelvis velocity in its heading frame (as in evaluation:
# push, then switch the controller on). Each 50 Hz step of the get-down
# controller runs the exported "policy" model once: the task-adapted prior
# samples four tokens from the current sensor frame and the five-frame task
# input with four uniforms drawn here, and the frozen decoder in the same
# graph maps the codes and the ten-frame history to joint targets. The
# approximate damping controller holds each joint at its current position
# every step, so the PD actuators act as dampers between policy steps.


class Policy:
  physics_hz = 200
  policy_hz = 50
  publish_hz = 50
  control_loop = "policy-then-decimation"

  def initialize(self, context):
    self.model = context.models["policy"]
    self.last_time = -1.0
    self.last_key = None
    self.rng = np.random.default_rng(0)
    self.past = np.zeros((1, (HISTORY - 1) * OBS_DIM), dtype=np.float32)
    self.past_task = np.zeros((1, PAST_TASK_DIM), dtype=np.float32)
    self.prev = np.zeros((1, ACTION_DIM), dtype=np.float32)
    self.last_uniform = np.zeros((1, NUM_TOKENS), dtype=np.float32)
    self.last_tokens = np.zeros((1, NUM_TOKENS), dtype=np.float32)

  def _choice(self, commands, key, count):
    value = int(float(commands[key][0]))
    if value < 0 or value >= count:
      raise ValueError(key + " out of range")
    return value

  def _state(self, start, push):
    # Only builtins the policy sandbox provides: no dict() or list().
    state = {k: v for k, v in RESET_STATES[start].items()}
    x, y, z, w = state["root_rotation"]
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    forward, left = PUSHES[push]
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    v = state["root_linear_velocity"]
    state["root_linear_velocity"] = [
      v[0] + c * forward - s * left,
      v[1] + s * forward + c * left,
      v[2],
    ]
    return ControllerInitialState(root_body_id="pelvis", **state)

  def initial_state(self):
    return self._state(0, 0)

  def reset_state(self, commands):
    start = self._choice(commands, "start", len(RESET_STATES))
    return self._state(start, self._choice(commands, "push", len(PUSHES)))

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
    push = self._choice(step.commands, "push", len(PUSHES))
    controller = self._choice(step.commands, "controller", len(CONTROLLERS))
    key = (start, push, controller)
    time = float(data.simulation_time)
    reset = step.policy_index == 0 or time <= self.last_time or key != self.last_key
    if reset:
      self.prev = np.zeros((1, ACTION_DIM), dtype=np.float32)
      self.rng = np.random.default_rng([start, push])
    state = self.robot_state(data)
    self.last_time, self.last_key = time, key
    metrics = {
      "start_index": float(start),
      "push_index": float(push),
      "controller_index": float(controller),
      "pelvis_height_m": float(state[0, 2]),
    }
    if controller == 1:
      return StepResult(actuation=state[0, 13:40].copy(), metrics=metrics)
    uniform = self.rng.random((1, NUM_TOKENS)).astype(np.float32)
    outputs = self.model.run(
      {
        "robot_state": state,
        "past_obs": self.past,
        "prev_action": self.prev,
        "reset": np.asarray([[1.0 if reset else 0.0]], dtype=np.float32),
        "uniform": uniform,
        "past_task": self.past_task,
      }
    )
    target = np.asarray(outputs["position_targets"], dtype=np.float32).reshape(27)
    action = np.asarray(outputs["action"], dtype=np.float32).reshape(1, ACTION_DIM)
    history = np.asarray(outputs["history"], dtype=np.float32).reshape(1, HISTORY * OBS_DIM)
    task = np.asarray(outputs["task_history"], dtype=np.float32).reshape(1, -1)
    if not (np.isfinite(target).all() and np.isfinite(action).all()):
      raise ValueError("nonfinite policy output")
    self.past = history[:, OBS_DIM:].copy()
    self.past_task = task[:, TASK_FRAME_DIM:].copy()
    self.prev = action.copy()
    self.last_uniform = uniform
    self.last_tokens = np.asarray(outputs["tokens"], dtype=np.float32).reshape(1, NUM_TOKENS)
    metrics |= {
      "action_rms": float(np.sqrt(np.mean(action * action))),
    }
    return StepResult(actuation=target, metrics=metrics)
