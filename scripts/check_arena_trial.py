"""Check the outcome and four inspection videos from one Arena apple trial."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


VIEWS = ("overview", "head", "left_wrist", "right_wrist")


def check(output: Path) -> int:
    result = json.loads((output / "result.json").read_text())
    # result.json contains Beijing paths; permit checking a downloaded copy.
    videos = [output / Path(path).name for path in result.get("video_files", [])]
    failures = []
    for view in VIEWS:
        matches = [path for path in videos if path.name.endswith(f"_{view}_0000.mp4")]
        if len(matches) != 1 or not matches[0].is_file():
            failures.append(f"{view}: expected one recorded MP4")
            continue
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=width,height,nb_read_frames",
                 "-of", "json", str(matches[0])],
                capture_output=True, text=True, check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            failures.append(f"{view}: ffprobe could not inspect video: {exc}")
            continue
        streams = json.loads(probe.stdout).get("streams", [])
        if len(streams) != 1:
            failures.append(f"{view}: missing video stream")
            continue
        info = streams[0]
        frames = int(info.get("nb_read_frames", 0))
        dimensions = (info.get("width"), info.get("height"))
        expected = (640, 480) if view != "overview" else dimensions
        if dimensions != expected or any(not isinstance(x, int) or x < 1 for x in dimensions) or frames < 2:
            failures.append(f"{view}: invalid dimensions or frame count; got {info}")
        else:
            print(f"{view}: {frames} frames at {dimensions[0]}x{dimensions[1]} ({matches[0].name})")

    if not result.get("verified_pick_and_place"):
        failures.append(f"pick and place not verified: {result.get('failure')}")
    if result.get("success_phase") not in ("retreat", "hold"):
        failures.append(f"unexpected success phase: {result.get('success_phase')}")
    if not result.get("sustained_lift_near_hand"):
        failures.append("no sustained lift near hand")
    if result.get("object_pose_overridden_during_rollout") is not False:
        failures.append("object-pose integrity unknown")

    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    print("PASS: trial outcome and four inspection videos")
    print("Inspection videos are not synchronized contract data. Review grasp and release visually.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    raise SystemExit(check(parser.parse_args().output))
