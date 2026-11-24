"""Get-down CoLA run lifecycle: the joystick CoLA run layout with the
get-down trainer, bundle and checks.

``miniverse.json`` also binds the motion-archive manifest, the joystick-start
bank and the AMP reference clips; resuming with a different prior, archive or
bank is refused.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

from super_gpc.rl import prior as P
from super_gpc.rl.cola import ColaPolicy
from super_gpc.rl.networks import Tracker
from super_gpc.tasks.get_down import mdp as M
from super_gpc.tasks.get_down.rl import get_down_export as X
from super_gpc.tasks.get_down.rl.get_down import GetDownTrainer, load_bank, push_conditions
from super_gpc.tasks.get_down.rl.get_down_cfg import GetDownCfg
from super_gpc.tasks.steering.rl.cola_run import ColaRun
from super_gpc.tasks.tracking.mdp import observations as O
from super_gpc.tasks.tracking.rl import export as E
from super_gpc.tasks.tracking.rl.stage2_cfg import Stage2Cfg


def policy_from_state(state: dict) -> ColaPolicy:
  """Adapted policy of a saved get-down state."""
  cfg = state["cfg"]
  if state.get("task_dim") != M.TASK_DIM:
    raise ValueError(f"checkpoint task input is not the {M.TASK_DIM}-D get-down frame")
  d = state["prior_info"]["d_model"]
  s2 = Stage2Cfg(d_model=d)
  prior = P.CodePrior(O.OBS_DIM, d, s2.num_layers, s2.num_heads, s2.dropout)
  prior.load_state_dict(state["prior"])
  tracker = Tracker(O.ENCODER_DIM, O.HISTORY_DIM, O.ACTION_DIM)
  tracker.load_state_dict(state["tracker"])
  policy = ColaPolicy(prior, tracker, cfg["action_clip"], cfg["rank"], cfg["alpha"], M.TASK_DIM)
  policy.adapters.load_state_dict(state["adapters"])
  return policy.eval()


def export_checkpoint(run_dir: str, bundle_id: str, state: dict, native: str, sha: str) -> dict:
  """Export, parity-check and closed-loop-check one saved policy state."""
  binding = json.loads((Path(run_dir) / "miniverse.json").read_text())
  bank = load_bank(Path(binding["provenance"]["starts"]["path"]))
  pushes = push_conditions(state["cfg"]["eval_push_mps"])
  policy = policy_from_state(state)
  graph = X.GetDownPolicyGraph(policy)
  it = state["iteration"]
  out = Path(run_dir) / "exports" / f"iter_{it:07d}"
  info = X.build_bundle(
    out,
    graph,
    bundle_id,
    {"iteration": it, "native_sha256": sha},
    {"iteration": it},
    bank,
    pushes,
    embodiment_from=Path(run_dir) / "bundle" / "source" / "embodiment",
  )
  onnx = Path(info["onnx"])
  samples = X.parity_samples(bank, it)
  record = {
    "step": it,
    "label": f"iter-{it}",
    "models": {"policy": str(onnx)},
    "onnx": str(onnx),
    "onnx_sha256": info["onnx_sha256"],
    "native": native,
    "native_sha256": sha,
    "prior": state["prior_info"],
    "parity": X.parity(graph, onnx, samples),
    "latency": X.latency(onnx, samples[0]),
    "closed_loop": X.closed_loop_check(out, policy, pushes),
  }
  (out.parent / f"iter_{it:07d}.json").write_text(json.dumps(record, indent=2))
  gc.collect()  # release ONNX Runtime sessions held in reference cycles
  return record


def export_metrics(record: dict) -> dict:
  out = record["parity"] | record["latency"]
  for r in record["closed_loop"]["rollouts"]:
    mode = "damping" if r["controller"] == X.CONTROLLERS[1] else "policy"
    key = f"export/closed_loop_{r['push']}_{mode}"
    out[f"{key}_peak_floor_force_kn"] = r["peak_floor_force_n"] / 1000
    out[f"{key}_final_torso_height_m"] = r["final_torso_height_m"]
    out["export/closed_loop_token_mismatch_steps"] = out.get(
      "export/closed_loop_token_mismatch_steps", 0.0
    ) + float(r["token_mismatch_steps"])
    out["export/closed_loop_max_action_diff"] = max(
      out.get("export/closed_loop_max_action_diff", 0.0), r["max_abs_action_diff"]
    )
  return out


class GetDownRun(ColaRun):
  stage = "rlft-get-down"
  title = "super-gpc get-down CoLA RLFT (AMP)"
  run_prefix = "get-down"
  export_fn = staticmethod(export_checkpoint)
  eval_keys = (
    "reward/task",
    "reward/amp_style",
    "peak_floor_force_kn",
    "peak_pelvis_down_speed_mps",
    "settled_on_back_rate",
    "settled_pelvis_height_m",
    "policy/kl_base_sum",
    "training_state_preserved",
  )
  train_keys = (
    "reward/task",
    "reward/amp_style",
    "reward/pelvis_height",
    "task/on_back",
    "task/peak_floor_force_n",
    "disc/accuracy",
    "policy/kl_base_sum",
    "actor/updates",
    "time/iteration_s",
  )

  def __init__(
    self,
    cfg: GetDownCfg,
    prior: Path,
    motions: Path,
    starts: Path,
    run_dir: Path,
    device: str,
    name: str,
  ):
    self.motions, self.starts_path = Path(motions), Path(starts).resolve()
    super().__init__(cfg, prior, run_dir, device, name)
    bound = self.binding["provenance"]
    if bound["starts"]["sha256"] != E.sha256(self.starts_path):
      raise ValueError("this run directory was created with a different joystick-start bank")
    if bound["motions_manifest_sha256"] != E.sha256(self.motions / "manifest.json"):
      raise ValueError("this run directory was created with a different motion archive")

  def make_trainer(self) -> GetDownTrainer:
    return GetDownTrainer(self.cfg, self.prior_path, self.motions, self.starts_path, self.device)

  def source_bundle(self, out: Path, bundle_id: str, provenance: dict, metadata: dict) -> dict:
    t = self.trainer
    graph = X.GetDownPolicyGraph(t.policy)
    return X.build_bundle(out, graph, bundle_id, provenance, metadata, t.bank, t.conditions)

  def export_metrics(self, record: dict) -> dict:
    return export_metrics(record)

  def extra_provenance(self) -> dict:
    return {
      "starts": {
        "path": str(self.starts_path),
        "sha256": E.sha256(self.starts_path),
        "meta": self.trainer.bank["meta"],
      },
      "motions_manifest_sha256": E.sha256(self.motions / "manifest.json"),
      "amp_reference": self.trainer.reference_info,
    }
