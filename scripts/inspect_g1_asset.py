#!/usr/bin/env python3
"""Load the project G1 USD and verify its joints against the data contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.experience = str(PROJECT_ROOT / "configs/ycb.python.headless.kit")
args.kit_args = f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"
app = AppLauncher(args).app

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from vla_isaaclab.envs.common.g1 import (
    ACTION_JOINT_NAMES,
    CONTRACT_JOINT_NAMES,
    LEFT_END_EFFECTOR,
    RIGHT_END_EFFECTOR,
    make_g1_cfg,
)


def main() -> int:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device))
    prim_utils.create_prim("/World/Origin", "Xform")
    cfg = make_g1_cfg((0.0, 0.0, 0.74))
    cfg.prim_path = "/World/Origin/Robot"
    robot = Articulation(cfg)
    sim.reset()

    contract_ids, contract_sim_names = robot.find_joints(
        list(ACTION_JOINT_NAMES), preserve_order=True
    )
    missing_bodies = [
        name for name in ("head_link", LEFT_END_EFFECTOR, RIGHT_END_EFFECTOR)
        if name not in robot.body_names
    ]
    passed = (
        robot.num_joints == 43
        and list(contract_sim_names) == list(ACTION_JOINT_NAMES)
        and len(set(contract_ids)) == 43
        and not missing_bodies
    )
    report = {
        "passed": passed,
        "num_joints": robot.num_joints,
        "is_fixed_base": robot.is_fixed_base,
        "usd_joint_names": list(robot.joint_names),
        "contract_simulator_joint_names": list(contract_sim_names),
        "contract_canonical_joint_names": list(CONTRACT_JOINT_NAMES),
        "contract_to_usd_indices": list(contract_ids),
        "num_bodies": robot.num_bodies,
        "required_body_names": ["head_link", LEFT_END_EFFECTOR, RIGHT_END_EFFECTOR],
        "missing_body_names": missing_bodies,
    }
    print(json.dumps(report, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        app.close()
