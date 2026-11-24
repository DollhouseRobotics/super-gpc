"""Render an exported joystick CoLA checkpoint to MP4 in native MuJoCo.

Drives the export's own ``policy.py`` controller and ONNX model (the files
published to Miniverse) through a scripted joystick sequence with auto look
on, via the same closed loop as the export check, so every step also
recomputes tokens with the training-code path. A yellow ray from the head
marks the controller's gaze target. Prints the closed-loop report as JSON.

Example:
  MUJOCO_GL=egl gpc-render-joystick --export runs/joystick/exports/iter_0002500 \\
    --native runs/joystick/checkpoints/policy_0002500.pt --out idle.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from super_gpc.tasks.steering.rl import cola_export as X
from super_gpc.tasks.steering.rl.cola_run import _policy_from_state

# (seconds, movement [right, forward] m/s, facing rad)
SCRIPT = (
  (14.0, (0.0, 0.0), 0.0),
  (5.0, (0.0, 1.0), 0.0),
  (10.0, (0.0, 0.0), 0.0),
  (4.0, (0.6, 0.0), 0.0),
  (12.0, (0.0, 0.0), 1.5),
)


def caption(stick, facing, metrics) -> str:
  if stick == (0.0, 0.0):
    yaw, pitch = (
      np.degrees(metrics["gaze_target_yaw_rad"]),
      np.degrees(metrics["gaze_target_pitch_rad"]),
    )
    look = f"auto look: yaw {yaw:+.0f} deg, pitch {pitch:+.0f} deg"
    return f"stick idle, facing {facing:+.1f} rad | {look}"
  return f"walk: right {stick[0]:+.1f}, forward {stick[1]:+.1f} m/s, facing {facing:+.1f} rad"


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("--export", type=Path, required=True, help="exports/iter_N bundle directory")
  p.add_argument("--native", type=Path, required=True, help="matching native policy_*.pt")
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--width", type=int, default=1280)
  p.add_argument("--height", type=int, default=720)
  p.add_argument("--every", type=int, default=2, help="render every Nth 50 Hz step")
  p.add_argument("--threads", type=int, default=8, help="torch and ONNX Runtime CPU threads")
  args = p.parse_args()
  torch.set_num_threads(args.threads)  # also sizes the ONNX Runtime session
  policy = _policy_from_state(torch.load(args.native, map_location="cpu", weights_only=False))
  frames, state = [], {"i": 0, "seg": 0, "n": 0, "lookat": None}
  renderer, cam, head = None, mujoco.MjvCamera(), None
  font = ImageFont.load_default(size=24)
  bounds = np.cumsum([round(s * 50) for s, _, _ in SCRIPT])

  def on_step(model, data, ctrl, metrics):
    nonlocal renderer, head
    i = state["i"]
    state["i"] += 1
    if i % args.every:
      return
    if renderer is None:
      model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
      model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
      model.vis.headlight.ambient[:] = 0.45  # display only; the robot meshes are dark
      model.vis.headlight.diffuse[:] = 0.6
      plane = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
      model.geom_rgba[plane] = (0.62, 0.64, 0.68, 1.0)
      renderer = mujoco.Renderer(model, args.height, args.width)
      head = model.body("head_tilt_link").id
    pelvis = data.qpos[:3].copy()
    look = pelvis if state["lookat"] is None else 0.9 * state["lookat"] + 0.1 * pelvis
    state["lookat"] = look
    cam.lookat[:] = (look[0], look[1], 0.75)
    cam.distance, cam.elevation, cam.azimuth = 2.6, -10.0, 150.0
    renderer.update_scene(data, cam)
    scene = renderer.scene
    yaw, pitch = ctrl.last_gaze_world
    origin = data.xpos[head]
    tip = origin + 1.2 * np.array(
      [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)]
    )
    if scene.ngeom + 2 <= scene.maxgeom:
      rgba = np.array([1.0, 0.85, 0.1, 0.9], dtype=np.float32)
      g = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.04, 0, 0], tip, np.eye(3).ravel(), rgba
      )
      scene.ngeom += 1
      g = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba
      )
      mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008, origin, tip)
      scene.ngeom += 1
    seg = int(np.searchsorted(bounds, i, side="right"))
    _, stick, facing = SCRIPT[min(seg, len(SCRIPT) - 1)]
    image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, args.width, 44), fill=(0, 0, 0))
    text = f"t={i / 50:5.1f} s   {caption(stick, facing, metrics)}"
    draw.text((14, 10), text, fill=(255, 255, 255), font=font)
    frames.append(np.asarray(image))

  report = X.closed_loop_check(args.export, policy, script=SCRIPT, on_step=on_step)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(args.out, frames, fps=50 // args.every, codec="libx264", quality=8)
  print(json.dumps(report | {"video": str(args.out), "frames": len(frames)}))


if __name__ == "__main__":
  main()
