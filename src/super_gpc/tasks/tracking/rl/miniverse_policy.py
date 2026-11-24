# ruff: noqa: F821
# Miniverse controller template for the Stage 1 tracker. The exporter prepends
# constants (MOTION_LABELS, RESET_STATES, JOINT_NAMES, FREE_JOINT, LENGTHS,
# OBS_DIM, ACTION_DIM, HISTORY, HIGHPASS); Miniverse supplies np, ControllerInitialState
# and StepResult. Not imported by the package.
#
# The exported graph owns every numeric step of the trained inference path:
# torso FK, BeyondMimic window alignment, encoder, 16x7 FSQ, decoder and the
# action-to-joint-target mapping. This controller only reads the native MuJoCo
# generalized state, keeps the decoder's past-observation history and the
# executed previous action, and counts policy frames since the episode start.
# It also reports the physical joint-target rate, total and after the same
# 5 Hz high-pass as training's jitter metrics (HIGHPASS = b0, b1, b2, a1, a2),
# both starting at rest on each episode's first target.


class Policy:
  physics_hz = 200
  policy_hz = 50
  publish_hz = 50
  control_loop = "policy-then-decimation"

  def initialize(self, context):
    self.model = context.models["tracker"]
    self.frame = 0
    self.last_time = -1.0
    self.last_motion = -1
    self.past = np.zeros((1, (HISTORY - 1) * OBS_DIM), dtype=np.float32)
    self.prev = np.zeros((1, ACTION_DIM), dtype=np.float32)

  def _state(self, index):
    if index < 0 or index >= len(RESET_STATES):
      raise ValueError("motion index out of range")
    return ControllerInitialState(root_body_id="pelvis", **RESET_STATES[index])

  def initial_state(self):
    return self._state(0)

  def reset_state(self, commands):
    return self._state(int(float(commands["motion"][0])))

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
    # pelvis pos, WXYZ quaternion, world linear velocity, body angular velocity
    state = np.concatenate((root, twist, np.asarray(q), np.asarray(qd)))
    if not np.isfinite(state).all():
      raise ValueError("nonfinite generalized state")
    return state.astype(np.float32).reshape(1, 67)

  def step(self, step):
    data = step.sim_data
    motion = int(float(step.commands["motion"][0]))
    if motion < 0 or motion >= len(RESET_STATES):
      raise ValueError("motion index out of range")
    time = float(data.simulation_time)
    reset = step.policy_index == 0 or time <= self.last_time or motion != self.last_motion
    if reset:
      self.frame = 0
      self.prev = np.zeros((1, ACTION_DIM), dtype=np.float32)
      self.hp_x = None
    state = self.robot_state(data)
    outputs = self.model.run(
      {
        "robot_state": state,
        "past_obs": self.past,
        "prev_action": self.prev,
        "reset": np.asarray([[1.0 if reset else 0.0]], dtype=np.float32),
        "motion": np.asarray([motion], dtype=np.int64),
        "frame": np.asarray([self.frame], dtype=np.int64),
      }
    )
    target = np.asarray(outputs["position_targets"], dtype=np.float32).reshape(27)
    action = np.asarray(outputs["action"], dtype=np.float32).reshape(1, ACTION_DIM)
    history = np.asarray(outputs["history"], dtype=np.float32).reshape(1, HISTORY * OBS_DIM)
    reference = np.asarray(outputs["reference"], dtype=np.float32).reshape(67)
    if not (np.isfinite(target).all() and np.isfinite(action).all()):
      raise ValueError("nonfinite policy output")
    if self.hp_x is None:
      self.hp_x = [target.copy(), target.copy()]
      self.hp_y = [np.zeros(27), np.zeros(27)]
    b0, b1, b2, a1, a2 = HIGHPASS
    fast = (
      b0 * target + b1 * self.hp_x[0] + b2 * self.hp_x[1] - a1 * self.hp_y[0] - a2 * self.hp_y[1]
    )
    rate = (target - self.hp_x[0]) * self.policy_hz
    fast_rate = (fast - self.hp_y[0]) * self.policy_hz
    self.hp_x = [target.copy(), self.hp_x[0]]
    self.hp_y = [fast, self.hp_y[0]]
    self.past = history[:, OBS_DIM:].copy()
    self.prev = action.copy()
    self.last_time, self.last_motion = time, motion
    clip_frame = min(self.frame, LENGTHS[motion] - 1)
    self.frame += 1
    joint_err = state[0, 13:40] - reference[13:40]
    return StepResult(
      actuation=target,
      metrics={
        "motion_index": float(motion),
        "reference_frame": float(clip_frame),
        "clip_complete": 1.0 if clip_frame >= LENGTHS[motion] - 1 else 0.0,
        "pelvis_height_m": float(state[0, 2]),
        "pelvis_height_err_m": float(state[0, 2] - reference[2]),
        "pelvis_xy_err_m": float(np.sqrt(np.sum((state[0, 0:2] - reference[0:2]) ** 2))),
        "joint_pos_rmse_rad": float(np.sqrt(np.mean(joint_err * joint_err))),
        "action_rms": float(np.sqrt(np.mean(action * action))),
        "target_rate_rms_rad_s": float(np.sqrt(np.mean(rate * rate))),
        "target_rate_fast_rms_rad_s": float(np.sqrt(np.mean(fast_rate * fast_rate))),
      },
    )
