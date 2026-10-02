"""Audit measured CPU orchard harvest evidence without starting Isaac Sim."""

import argparse
import json
import math
from pathlib import Path

from check_orchard_run import check as check_orchard_report, check_videos


def summarize(rows):
    """Return compact measured phase landmarks for the CLI result."""
    summary = {"samples": len(rows)}
    for label, predicate in (
        ("contact", lambda row: row.get("grasp_contact") is True),
        ("break", lambda row: row.get("detached") is True),
        ("carry", lambda row: row.get("carried") is True),
        ("release", lambda row: row.get("released") is True),
    ):
        index = next((i for i, row in enumerate(rows) if predicate(row)), None)
        summary[label] = None if index is None else {
            "sample": index,
            "step": rows[index].get("step"),
            "phase": rows[index].get("phase"),
        }
    return summary


def check_trajectory(rows):
    errors = []
    if not isinstance(rows, list) or not rows:
        return ["Missing or empty trajectory evidence"], {"samples": 0}
    if any(not isinstance(row, dict) for row in rows):
        return ["Trajectory contains a malformed non-object sample"], {"samples": len(rows)}
    summary = summarize(rows)
    breaks = [i for i, row in enumerate(rows) if row.get("detached") is True]
    if not breaks:
        return ["No observed native stem break in trajectory"], summary
    break_i = breaks[0]
    if not any(row.get("detached") is False for row in rows[:break_i + 1]):
        errors.append("Trajectory lacks an attached native-stem sample before break")
    if any(row.get("detached") is False for row in rows[break_i + 1:]):
        errors.append("Native stem detached state reverted after break")

    attached_contact = next((i for i in range(max(0, break_i - 2), break_i)
                             if rows[i].get("detached") is False
                             and rows[i].get("grasp_contact") is True), None)
    if attached_contact is None:
        errors.append("Sustained grasp contact not observed on an attached sample within 2 steps before break")

    break_position = rows[break_i].get("position")
    carry_i = None
    carry_displacement = None
    if isinstance(break_position, list) and len(break_position) >= 2:
        for index, row in enumerate(rows[break_i + 1:], start=break_i + 1):
            position = row.get("position")
            if not (row.get("detached") is True and row.get("carried") is True
                    and row.get("grasp_contact") is True
                    and isinstance(position, list) and len(position) >= 2):
                continue
            try:
                displacement = math.dist(break_position[:2], position[:2])
            except (TypeError, ValueError):
                continue
            if math.isfinite(displacement) and displacement >= 0.10:
                carry_i = index
                carry_displacement = displacement
                break
    if carry_i is None:
        errors.append("No detached, contacted carry sample shows 0.10 m horizontal travel from break")
    else:
        summary["carry"] = {"sample": carry_i, "step": rows[carry_i].get("step"),
                             "phase": rows[carry_i].get("phase"),
                             "horizontal_displacement_m": round(carry_displacement, 4)}
        release_i = next((i for i, row in enumerate(rows)
                          if row.get("released") is True), None)
        if release_i is None or release_i <= carry_i:
            errors.append("Physical release not observed after qualifying carry")

    final = rows[-1]
    if final.get("supported") is not True:
        errors.append("Final trajectory sample lacks measured support")
    if final.get("opened") is not True:
        errors.append("Final trajectory sample lacks an opened hand")
    if final.get("released") is not True:
        errors.append("Final trajectory sample lacks measured release")
    try:
        stable_steps = int(final.get("stable_steps", 0))
    except (TypeError, ValueError):
        stable_steps = 0
    if stable_steps < 30:
        errors.append("Final trajectory sample lacks 30 stable placement steps")
    if final.get("failure") is not None:
        errors.append(f"Final trajectory reports failure: {final.get('failure')}")
    if final.get("success") is not True:
        errors.append("Final trajectory sample lacks measured task success")
    return errors, summary


def check(report, trajectory=None):
    """Validate the report contract and measured trajectory evidence."""
    errors = check_orchard_report(report)
    if report.get("orchard", {}).get("physics_device") != "cpu":
        errors.append("CPU physics device required")
    trajectory_errors, summary = check_trajectory(trajectory)
    return errors + trajectory_errors, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--videos", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    trajectory_path = args.report.with_name("trajectory.jsonl")
    rows = []
    trajectory_read_error = None
    try:
        with trajectory_path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        trajectory_read_error = f"Cannot read trajectory evidence: {exc}"
    errors, summary = check(report, rows)
    if trajectory_read_error:
        errors.append(trajectory_read_error)
    if args.videos:
        errors.extend(check_videos(args.videos, report.get("steps", 0),
                                   report.get("orchard", {}).get("camera_profile")))
    print(json.dumps({"passed": not errors, "errors": errors, "evidence": summary}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
