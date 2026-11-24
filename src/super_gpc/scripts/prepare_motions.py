"""Download, verify and convert pinned motion corpora to 50 Hz references.

Each ``--datasets`` key names a pinned private Hugging Face dataset revision in
``DATASETS``. Every one stores packed 30 Hz ProtoMotions libraries for the
27-joint Dollhouse G1 (``dps`` in MJCF tree order, ``grs`` XYZW, ``gts[:, 0]``
the pelvis origin; see docs/spec.md). The default is the
50-hour corpus alone. Several keys build their union into one prepared
directory: clip names must be unique across datasets, categories are prefixed
by the dataset key. Every clip is a training clip (``VAL_PERCENT`` 0). The
source-group hash rule is kept so a nonzero percentage would still keep a
take and its mirror in one split.

With one dataset ``--source`` is its download directory; with several each
dataset downloads into ``--source/<key>``. Only the manifest and the packed
chunks are downloaded, and both are checked against the dataset's SHA256SUMS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch

from super_gpc.asset_zoo.robots.dollhouse_g1 import dollhouse_g1_constants as G1
from super_gpc.tasks.tracking.mdp.motion_library import (
  POLICY_HZ,
  SOURCE_HZ,
  resample_shard,
  split_of,
)

DATASETS = {
  "bones-50h": (
    "dollhouserobotics/bones-seed-50hrs-dollhouse-g1",
    "7dae6da0c589375dd9e5d8d1a1c4bdc132333d8c",
  ),
  "bones-15h-basic": (
    "dollhouserobotics/dollhouse-g1-bones-15hrs-basic",
    "940f7173f590eac9c71338ef507f66d1474a6c5b",
  ),
  "bones-advanced": (
    "dollhouserobotics/dollhouse-g1-bones-advanced",
    "230a55803436e28c9f667ccdd75cff26428636e4",
  ),
  "bones-floor": (
    "dollhouserobotics/dollhouse-g1-bones-floor",
    "164c84a23bcb4218d815d6dd8c5b8c29d984f3a7",
  ),
  "g1-getups": (
    "dollhouserobotics/dollhouse-g1-getups",
    "0079f8caa1b103dd573110c2e4f58324538ef020",
  ),
  "g1-crawl": ("dollhouserobotics/dollhouse-g1-crawl", "f13d344d6718156b6238523fba44e0209d44d8ef"),
}
VAL_PERCENT = 0  # every stage trains on all data, no held-out split


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    while chunk := f.read(1 << 24):
      h.update(chunk)
  return h.hexdigest()


def load_dataset(key: str, source: Path, download: bool) -> tuple[dict, list[dict]]:
  """Verified source checksums and normalized clip records of one dataset."""
  repo, revision = DATASETS[key]
  if download:
    from huggingface_hub import snapshot_download

    snapshot_download(
      repo,
      repo_type="dataset",
      revision=revision,
      local_dir=source,
      allow_patterns=["SHA256SUMS", "manifest.json", "chunk_*.pt"],
    )
  sums = dict(reversed(line.split()) for line in (source / "SHA256SUMS").read_text().splitlines())
  if sha256(source / "manifest.json") != sums["manifest.json"]:
    raise RuntimeError(f"{key}/manifest.json: checksum mismatch")
  manifest = json.loads((source / "manifest.json").read_text())
  raw = manifest["training"]["motions"] if "training" in manifest else manifest["motions"]
  records = []
  for r in raw:
    name = r["filename"]
    records.append(
      dict(
        name=name,
        filename=name,
        group=r.get("pairId") or name.removesuffix("_M"),
        category=r.get("bucket") or r.get("category") or key,
        mirror=bool(r.get("isMirror", name.endswith("_M"))),
        file=r.get("outputShard") or f"chunk_{r['shard']:02d}.pt",
        outputIndex=r["outputIndex"],
        frames=r["frames"],
        # The advanced set's CSV-converted clips have no source shard entry;
        # their packed name is the filename.
        sourceName=r.get("sourceName") or f"{name}.motion",
        sourceIndex=r.get("sourceIndex", -1),
        description=r.get("description", ""),
      )
    )
  checks = {}
  for name in ["manifest.json", *sorted({r["file"] for r in records})]:
    actual = sha256(source / name)
    if actual != sums[name]:
      raise RuntimeError(f"{key}/{name}: sha256 {actual} != {sums[name]}")
    checks[name] = actual
  return dict(key=key, repository=repo, revision=revision, sourceChecksums=checks), records


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument("--source", type=Path, required=True, help="download/cache directory")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--datasets", nargs="+", default=["bones-50h"], choices=sorted(DATASETS))
  parser.add_argument("--skip-download", action="store_true")
  args = parser.parse_args()
  union = len(args.datasets) > 1

  model = mujoco.MjModel.from_xml_path(str(G1.mjcf_path()))
  tree = [model.joint(i).name for i in range(1, model.njnt)]
  tree_to_action = np.array([tree.index(n) for n in G1.ACTION_JOINT_NAMES])

  args.output.mkdir(parents=True, exist_ok=True)
  qpos_all, qvel_all, clips, sources = [], [], [], []
  offset = 0
  for key in args.datasets:
    source = args.source / key if union else args.source
    info, records = load_dataset(key, source, not args.skip_download)
    sources.append(info)
    for file in sorted({r["file"] for r in records}):
      shard_records = sorted(
        (r for r in records if r["file"] == file), key=lambda r: r["outputIndex"]
      )
      data = torch.load(source / file, mmap=True, weights_only=True)
      dt = data["motion_dt"].numpy()
      if not np.allclose(dt, 1 / SOURCE_HZ):
        raise ValueError(f"{key}/{file}: unexpected source rate")
      if tuple(data["dps"].shape[1:]) != (27,) or tuple(data["gts"].shape[1:]) != (32, 3):
        raise ValueError(f"{key}/{file}: unexpected source layout")
      names = data["motion_files"]
      for r in shard_records:
        if Path(names[r["outputIndex"]]).name != Path(r["sourceName"]).name:
          raise ValueError(f"{key}/{file}: manifest/shard name mismatch at {r['outputIndex']}")
      qp, qv, frames = resample_shard(data, tree_to_action, shard_records)
      for r, p, v, m in zip(shard_records, qp, qv, frames, strict=True):
        clips.append(
          dict(
            name=r["name"],
            group=r["group"],
            split=split_of(r["group"], VAL_PERCENT),
            offset=offset,
            frames=m,
            category=f"{key}/{r['category']}" if union else r["category"],
            mirror=r["mirror"],
            dataset=key,
            shard=file,
            sourceIndex=r["sourceIndex"],
            outputIndex=r["outputIndex"],
            description=r["description"],
          )
        )
        offset += m
        qpos_all.append(p)
        qvel_all.append(v)
      del data
  duplicates = [n for n, c in Counter(c["name"] for c in clips).items() if c > 1]
  if duplicates:
    raise RuntimeError(f"clip names repeat across datasets: {duplicates[:10]}")
  qpos = np.concatenate(qpos_all)
  qvel = np.concatenate(qvel_all)
  np.save(args.output / "qpos.npy", qpos)
  np.save(args.output / "qvel.npy", qvel)
  (args.output / "clips.json").write_text(json.dumps(clips))
  groups = {s: {c["group"] for c in clips if c["split"] == s} for s in ("train", "val")}
  if groups["train"] & groups["val"]:
    raise RuntimeError("train/validation source groups overlap")
  seconds, counts = Counter(), Counter()
  per_dataset = {s["key"]: Counter() for s in sources}
  for c in clips:
    seconds[c["split"]] += (c["frames"] - 1) / POLICY_HZ
    counts[c["split"]] += 1
    per_dataset[c["dataset"]][c["split"]] += (c["frames"] - 1) / POLICY_HZ / 3600
  for s in sources:
    s["hours"] = dict(per_dataset[s["key"]])
  out = dict(
    dataset="+".join(s["repository"] for s in sources),
    revision="+".join(s["revision"] for s in sources),
    datasets=sources,
    sourceRate=SOURCE_HZ,
    policyRate=POLICY_HZ,
    jointOrder=list(G1.ACTION_JOINT_NAMES),
    sourceJointOrder=tree,
    stateLayout="qpos: pelvis pos3, wxyz4, joints27; qvel: lin3 world, ang3 body, joints27",
    splitRule=f"sha256(pairId or filename without _M)[:8] mod 100 < {VAL_PERCENT} -> val",
    clips={k: counts[k] for k in counts},
    hours={k: seconds[k] / 3600 for k in seconds},
    sourceGroups={k: len(v) for k, v in groups.items()},
    excludedClips=[],
    qposSha256=sha256(args.output / "qpos.npy"),
    qvelSha256=sha256(args.output / "qvel.npy"),
    clipsSha256=sha256(args.output / "clips.json"),
  )
  if not union:
    out["sourceChecksums"] = sources[0]["sourceChecksums"]
  (args.output / "manifest.json").write_text(json.dumps(out, indent=2))
  print(
    json.dumps(
      {k: v for k, v in out.items() if k not in ("datasets", "sourceJointOrder")},
      indent=2,
    )
  )
  print(json.dumps([{k: s[k] for k in ("key", "revision", "hours")} for s in sources], indent=2))


if __name__ == "__main__":
  main()
