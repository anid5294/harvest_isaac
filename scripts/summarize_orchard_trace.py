"""Summarize measured hand distance and finger contact by controller phase."""

import argparse
import json
from pathlib import Path


def summarize(rows):
    phases = {}
    for row in rows:
        phase = phases.setdefault(row["phase"], {"steps": 0, "hand_distance_min_m": float("inf"),
            "hand_distance_max_m": 0, "contact_measurements_available": False,
            "grasp_contact_steps": 0, "peak_finger_force_n": {}})
        phase["steps"] += 1
        distance = row["hand_distance"]
        phase["hand_distance_min_m"] = min(phase["hand_distance_min_m"], distance)
        phase["hand_distance_max_m"] = max(phase["hand_distance_max_m"], distance)
        phase["hand_distance_last_m"] = distance
        phase["last_failure"] = row.get("failure")
        phase["grasp_contact_steps"] += int(row.get("grasp_contact", False))
        if "finger_contact_force_n" in row:
            phase["contact_measurements_available"] = True
            for name, force in row["finger_contact_force_n"].items():
                phase["peak_finger_force_n"][name] = max(phase["peak_finger_force_n"].get(name, 0), force)
        phase["last_apple_in_palm_frame_m"] = row.get("apple_in_palm_frame_m")
    return phases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    args = parser.parse_args()
    with args.trajectory.open() as stream:
        print(json.dumps(summarize(json.loads(line) for line in stream if line.strip()), indent=2))
