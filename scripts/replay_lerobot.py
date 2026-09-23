#!/usr/bin/env python3
"""Extract a LeRobot episode and replay it in a separate Isaac Sim process."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/replay/lerobot_rgb.png")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = args.dataset.resolve()
    if (root / "meta/simulation.json").is_file():
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        manifest = json.loads((root / "meta/simulation.json").read_text())
        dataset = LeRobotDataset(manifest["lerobot_repo_id"], root=root, video_backend="pyav")
        tabular = dataset.hf_dataset.with_format("numpy")
        mask = np.asarray(tabular["episode_index"]) == args.episode
        actions = np.asarray(tabular["sim.action.normalized"])[mask]
        seeds = np.asarray(tabular["sim.seed"])[mask]
        environment_id = manifest["environment_id"]
    else:
        info = json.loads((root / "meta/info.json").read_text())
        collection = json.loads((root / "meta/collection.json").read_text())
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"Expected LeRobot v2.1, found {info.get('codebase_version')}")
        chunk = args.episode // int(info["chunks_size"])
        parquet_path = root / info["data_path"].format(
            episode_chunk=chunk, episode_index=args.episode
        )
        if not parquet_path.is_file():
            raise IndexError(f"Episode {args.episode} is not present in {root}")
        table = pq.read_table(parquet_path, columns=["sim.action.normalized", "sim.seed"])
        actions = np.asarray(table["sim.action.normalized"].to_pylist(), dtype=np.float32)
        seeds = table["sim.seed"].to_numpy(zero_copy_only=False)
        environment_id = collection["environment_id"]
    if not len(actions):
        raise IndexError(f"Episode {args.episode} is not present in {root}")

    temporary_dir = PROJECT / "outputs/tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="lerobot-replay-", suffix=".npz", dir=temporary_dir, delete=False
    ) as temporary_file:
        episode_path = Path(temporary_file.name)
    try:
        np.savez_compressed(
            episode_path,
            actions=actions.astype(np.float32),
            seed=np.asarray(int(seeds.reshape(-1)[0]), dtype=np.int64),
            environment_id=np.asarray(environment_id),
            dataset=np.asarray(str(root)),
            episode=np.asarray(args.episode, dtype=np.int64),
        )
        command = [
            sys.executable,
            str(PROJECT / "scripts/replay_lerobot_sim.py"),
            str(episode_path),
            "--output",
            str(args.output),
            "--device",
            args.device,
        ]
        if args.headless:
            command.append("--headless")
        return subprocess.run(command, check=False).returncode
    finally:
        episode_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
