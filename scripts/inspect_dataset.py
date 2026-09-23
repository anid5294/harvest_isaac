#!/usr/bin/env python3
"""Print and validate the structure of a recorded Isaac Lab HDF5 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


COMMON_DATASETS = (
    "actions",
    "obs",
    "states/articulation/robot/joint_position",
    "sensors/camera/frames/rgb",
    "sensors/camera/frames/depth",
    "sensors/camera/calibration/intrinsic_matrix",
    "task/distance",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    with h5py.File(args.dataset, "r") as stream:
        episodes = sorted(stream["data"].keys())
        metadata = json.loads(stream["data"].attrs["env_args"])
        summary = {
            "file": str(args.dataset.resolve()),
            "metadata": metadata,
            "episodes": {},
            "valid": bool(episodes),
        }
        for episode_name in episodes:
            episode = stream[f"data/{episode_name}"]
            required = list(COMMON_DATASETS)
            missing = [name for name in required if name not in episode]
            summary["episodes"][episode_name] = {
                "steps": int(episode.attrs["num_samples"]),
                "success": bool(episode.attrs.get("success", False)),
                "action_shape": list(episode["actions"].shape),
                "observation_shape": list(episode["obs"].shape),
                "rgb_shape": list(episode["sensors/camera/frames/rgb"].shape)
                if "sensors/camera/frames/rgb" in episode else None,
                "depth_shape": list(episode["sensors/camera/frames/depth"].shape)
                if "sensors/camera/frames/depth" in episode else None,
                "rgb_compression": episode["sensors/camera/frames/rgb"].compression
                if "sensors/camera/frames/rgb" in episode else None,
                "missing": missing,
            }
            summary["valid"] = summary["valid"] and not missing
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
