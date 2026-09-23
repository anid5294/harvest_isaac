#!/usr/bin/env python3
"""Convert a neutral simulation staging file to LeRobot Dataset v3 or v2.1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from vla_isaaclab.recording.lerobot_v21 import export_staging_to_v21
from vla_isaaclab.recording.lerobot_v3 import export_staging_to_v3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging", type=Path)
    parser.add_argument(
        "--lerobot-version", choices=("3", "2.1"), default="3",
        help="Output dataset format (default: 3).",
    )
    parser.add_argument("--delete-staging", action="store_true")
    parser.add_argument(
        "--include-failed-episodes", action="store_true",
        help="Create a source-review export containing failures; never use it as the default imitation export.",
    )
    args = parser.parse_args()

    staging_path = args.staging.resolve()
    dataset_name = staging_path.stem
    exporter = export_staging_to_v3 if args.lerobot_version == "3" else export_staging_to_v21
    final_path = exporter(
        staging_path=staging_path,
        root=PROJECT / "outputs/lerobot",
        dataset_name=dataset_name,
        include_failed_episodes=args.include_failed_episodes,
    )

    if args.delete_staging:
        staging_path.unlink()
    print(json.dumps({
        "dataset": str(final_path),
        "source": str(staging_path),
        "lerobot_version": args.lerobot_version,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
