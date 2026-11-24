# ruff: noqa: F821
# Miniverse controller template for the joystick CoLA policy. The exporter
# prepends constants (INITIAL_STATE, JOINT_NAMES, FREE_JOINT, OBS_DIM, ACTION_DIM, HISTORY,
# NUM_TOKENS, SPEED_MAX, GAZE); Miniverse supplies np, ControllerInitialState
# and StepResult. Not imported by the package.
#
# Each 50 Hz step runs the exported "policy" model once: the command-adapted
# prior samples four tokens from the current sensor frame and the local
# command with four uniforms drawn here, and the frozen decoder in the same
# graph maps the codes and the ten-frame history to joint targets. The
# joystick gives the velocity in the commanded facing frame (UI [right,
# forward] -> [forward, left], m/s, at most SPEED_MAX); a changed facing
# command anchors at the current pelvis yaw, as in training.
#
# Gaze: while moving, and for the first look of every stop, the robot looks
# straight ahead along the commanded facing. At zero speed the "look" stick
# (right = look right, up = look up) aims the gaze; when it is centred and
# "auto_look" is on, the controller samples look targets from the training
# distribution (GAZE), so a standing robot looks around by itself. The gaze
# command slews toward the chosen look target at GAZE["rate"] rad/s, as in
# training; the exported model points the neck along it.


def _command(commands, key, default):
  # Miniverse's policy sandbox has no KeyError or tuple builtins.
  return commands[key] if key in commands else default


WAIST_YAW = JOINT_NAMES.index("waist_yaw_joint")
HEAD_PAN = JOINT_NAMES.index("head_pan_joint")


class Policy:
  physics_hz = 200
  policy_hz = 50
  publish_hz = 50
  control_loop = "policy-then-decimation"

  def initialize(self, context):
    self.model = context.models["policy"]
    self.last_time = -1.0
    self.rng = np.random.default_rng(0)
    self.past = np.zeros((1, (HISTORY - 1) * OBS_DIM), dtype=np.float32)
    self.prev = np.zeros((1, ACTION_DIM), dtype=np.float32)
    self.anchor = 0.0
    self.facing = 0.0
    self.last_uniform = np.zeros((1, NUM_TOKENS), dtype=np.float32)
    self.last_tokens = np.zeros((1, NUM_TOKENS), dtype=np.float32)
    self.last_command = np.zeros((1, 7), dtype=np.float32)
    self.gaze_rng = np.random.default_rng(1)
    self.gaze = (0.0, 0.0)  # look target: yaw offset from facing, pitch (up)
    self.slewed = (0.0, 0.0)  # gaze command moving toward the look target
    self.gaze_left = GAZE["min_steps"]
    self.last_gaze_world = (0.0, 0.0)

  def initial_state(self):
    return ControllerInitialState(root_body_id="pelvis", **INITIAL_STATE)

  def reset_state(self, commands):
    return self.initial_state()

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

  def command(self, commands, state, reset):
    w, x, y, z = (float(v) for v in state[0, 3:7])
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    facing = float(commands["facing"][0])
    stick = np.asarray(commands["movement"], dtype=np.float64).reshape(2)
    velocity = np.asarray([stick[1], -stick[0]])
    if not np.isfinite(facing) or not np.isfinite(velocity).all():
      raise ValueError("nonfinite steering command")
    speed = float(np.linalg.norm(velocity))
    velocity = velocity * min(1.0, SPEED_MAX / max(speed, 1e-12))
    delta = float(np.arctan2(np.sin(facing - self.facing), np.cos(facing - self.facing)))
    if reset or abs(delta) > 1e-6:
      self.anchor, self.facing = yaw, facing
      self.gaze, self.gaze_left = (0.0, 0.0), GAZE["min_steps"]  # a new command looks ahead
    goal = self.look(commands, speed * min(1.0, SPEED_MAX / max(speed, 1e-12)))
    if reset:
      self.slewed = (0.0, 0.0)
    step = GAZE["rate"] / self.policy_hz
    (yaw0, pitch0), (yaw1, pitch1) = self.slewed, goal
    self.slewed = (
      float(yaw0 + np.clip(yaw1 - yaw0, -step, step)),
      float(pitch0 + np.clip(pitch1 - pitch0, -step, step)),
    )
    gaze_yaw, pitch = self.slewed
    angle = self.anchor + self.facing - yaw
    c, s = np.cos(angle), np.sin(angle)
    g = angle + gaze_yaw
    local = [
      c * velocity[0] - s * velocity[1],
      s * velocity[0] + c * velocity[1],
      c,
      s,
      np.cos(g),
      np.sin(g),
      pitch,
    ]
    self.last_gaze_world = (self.anchor + self.facing + gaze_yaw, pitch)
    return np.asarray([local], dtype=np.float32), yaw

  def look(self, commands, speed):
    if speed >= GAZE["near_zero_speed"]:
      self.gaze, self.gaze_left = (0.0, 0.0), GAZE["min_steps"]
      return self.gaze
    stick = np.asarray(_command(commands, "look", [0.0, 0.0]), dtype=np.float64).reshape(2)
    if not np.isfinite(stick).all():
      raise ValueError("nonfinite look command")
    stick = np.clip(stick, -1.0, 1.0)
    if np.abs(stick).max() > 0.05:  # manual aim; auto-look restarts centred
      up = GAZE["pitch_max"] if stick[1] > 0 else -GAZE["pitch_min"]
      self.gaze_left = GAZE["min_steps"]
      return (-float(stick[0]) * GAZE["yaw_max"], float(stick[1]) * up)
    if float(np.asarray(_command(commands, "auto_look", [1.0])).reshape(-1)[0]) < 0.5:
      self.gaze = (0.0, 0.0)
      return self.gaze
    self.gaze_left -= 1
    if self.gaze_left <= 0:
      r = self.gaze_rng.random(4)
      if r[2] < GAZE["center_probability"]:
        self.gaze = (0.0, 0.0)
      else:
        lo, hi = GAZE["pitch_min"], GAZE["pitch_max"]
        self.gaze = ((2 * r[0] - 1) * GAZE["yaw_max"], lo + (hi - lo) * r[1])
      self.gaze_left = GAZE["min_steps"] + int(r[3] * (GAZE["max_steps"] - GAZE["min_steps"] + 1))
    return self.gaze

  def step(self, step):
    data = step.sim_data
    time = float(data.simulation_time)
    reset = step.policy_index == 0 or time <= self.last_time
    if reset:
      self.prev = np.zeros((1, ACTION_DIM), dtype=np.float32)
      self.rng = np.random.default_rng(0)
      self.gaze_rng = np.random.default_rng(1)
    state = self.robot_state(data)
    command, yaw = self.command(step.commands, state, reset)
    uniform = self.rng.random((1, NUM_TOKENS)).astype(np.float32)
    outputs = self.model.run(
      {
        "robot_state": state,
        "past_obs": self.past,
        "prev_action": self.prev,
        "reset": np.asarray([[1.0 if reset else 0.0]], dtype=np.float32),
        "uniform": uniform,
        "command": command,
      }
    )
    target = np.asarray(outputs["position_targets"], dtype=np.float32).reshape(27)
    action = np.asarray(outputs["action"], dtype=np.float32).reshape(1, ACTION_DIM)
    history = np.asarray(outputs["history"], dtype=np.float32).reshape(1, HISTORY * OBS_DIM)
    if not (np.isfinite(target).all() and np.isfinite(action).all()):
      raise ValueError("nonfinite policy output")
    self.past = history[:, OBS_DIM:].copy()
    self.prev = action.copy()
    self.last_uniform, self.last_command = uniform, command
    self.last_tokens = np.asarray(outputs["tokens"], dtype=np.float32).reshape(1, NUM_TOKENS)
    self.last_time = time
    vx, vy = float(state[0, 7]), float(state[0, 8])
    c, s = np.cos(yaw), np.sin(yaw)
    return StepResult(
      actuation=target,
      metrics={
        "command_forward_mps": float(command[0, 0]),
        "command_left_mps": float(command[0, 1]),
        "facing_error_rad": float(np.arctan2(command[0, 3], command[0, 2])),
        "gaze_target_yaw_rad": float(np.arctan2(command[0, 5], command[0, 4])),
        "gaze_target_pitch_rad": float(command[0, 6]),
        "look_yaw_rad": float(state[0, 13 + WAIST_YAW] + state[0, 13 + HEAD_PAN]),
        "velocity_forward_mps": c * vx + s * vy,
        "velocity_left_mps": -s * vx + c * vy,
        "pelvis_height_m": float(state[0, 2]),
        "action_rms": float(np.sqrt(np.mean(action * action))),
      },
    )
