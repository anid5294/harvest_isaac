"""Audit one orchard report and its profile-specific MP4 set without Isaac Sim."""

import argparse
import json
import math
from pathlib import Path
import subprocess


def check(report, passive=False):
    errors = []
    if report.get("task") != "VLA-OrchardPick-G1-JointPos-v0":
        errors.append("Not an orchard-pick report")
    if report.get("action_dimension") != 43:
        errors.append("Expected 43 active joint-position channels")
    orchard = report.get("orchard", {})
    metrics = orchard.get("metrics", {})
    if not metrics or metrics.get("failure"):
        errors.append(f"Task failure or missing metrics: {metrics.get('failure')}")
    if passive:
        position = metrics.get("position", [])
        layout = orchard.get("layout", {})
        initial = layout.get("apples", [[]])[layout.get("target_index", 0)]
        if (len(position) != 3 or len(initial) != 3 or
                not all(math.isfinite(v) for v in position) or math.dist(position, initial) > 0.015):
            errors.append("Attached fruit moved more than 15 mm during passive hold")
        if metrics.get("detached") or metrics.get("speed", float("inf")) > 0.035:
            errors.append("Fruit detached or failed to settle during passive hold")
        if metrics.get("detached_indices"):
            errors.append("One or more fruit stems broke during passive hold")
        if layout.get("tree_model") == "commercial":
            for index, initial_position in enumerate(layout.get("apples", [])):
                name = "object" if index == layout.get("target_index", 0) else f"apple_{index}"
                state = report.get("rigid_objects", {}).get(name, {})
                current_position = state.get("position_m", [])
                if (len(current_position) != 3 or not state.get("finite")
                        or not state.get("stable")
                        or math.dist(current_position, initial_position) > 0.015):
                    errors.append(f"{name}: commercial fruit did not remain stably attached")
    elif not (report.get("passed") and report.get("success_count", 0) > 0
              and report.get("termination_terms", {}).get("success")
              and metrics.get("success")):
        errors.append("Named harvest success did not fire")
    return errors


def check_videos(directory, expected_frames, camera_profile=None):
    errors = []
    names = (("external", "left_wrist", "right_wrist")
             if camera_profile in ("orchard_fixed_front_top_three_view_v1",
                                   "orchard_commercial_full_tree_three_view_v1")
             else ("external", "head", "left_wrist", "right_wrist"))
    for name in names:
        path = directory / f"{name}.mp4"
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
                "-of", "json", str(path),
            ], check=True, capture_output=True, text=True)
            stream = json.loads(result.stdout)["streams"][0]
            if (stream["width"], stream["height"], stream["r_frame_rate"]) != (640, 480, "30/1"):
                errors.append(f"{name}: expected 640x480 at 30 fps")
            if int(stream["nb_read_frames"]) != expected_frames:
                errors.append(f"{name}: video frame count differs from control-step count")
        except (OSError, subprocess.CalledProcessError, KeyError, ValueError, IndexError) as exc:
            errors.append(f"{name}: video probe failed: {exc}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--passive", action="store_true")
    parser.add_argument("--videos", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    errors = check(report, args.passive)
    if args.videos:
        errors += check_videos(args.videos, report["steps"], report.get("orchard", {}).get("camera_profile"))
    print(json.dumps({"passed": not errors, "errors": errors}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
