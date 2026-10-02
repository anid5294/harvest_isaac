"""Reject incomplete free-object demos; not a stem-harvesting success checker."""
import argparse
import json
from pathlib import Path

from check_orchard_run import check_videos


def check(report):
    errors = []
    if report.get("task") != "VLA-FreeApplePickPlace-G1-JointPos-v0":
        errors.append("Wrong task")
    if report.get("action_dimension") != 43:
        errors.append("Expected 43 action channels")
    demo = report.get("free_pick_place", {})
    metrics = demo.get("metrics", {})
    if demo.get("physics_device") != "cpu":
        errors.append("CPU validation required")
    if demo.get("native_stem_present") is not False or demo.get("harvest_success_claim") is not False:
        errors.append("Free-object test must not claim stem harvesting")
    if not (report.get("passed") is True and report.get("success_count", 0) > 0
            and report.get("termination_terms", {}).get("success") is True):
        errors.append("Named successful task termination required")
    for name in ("finite", "robot_stable", "grasp_confirmed", "lifted", "carried", "released", "supported", "success"):
        if metrics.get(name) is not True:
            errors.append(f"Missing physical evidence: {name}")
    if metrics.get("failure") or report.get("policy", {}).get("failed"):
        errors.append(f"Reported failure: {metrics.get('failure')}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--videos", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    errors = check(report)
    if args.videos:
        errors += check_videos(args.videos, report["steps"], "free_pick_place_three_view_v1")
    print(json.dumps({"passed": not errors, "errors": errors, "harvest_success_claim": False}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
