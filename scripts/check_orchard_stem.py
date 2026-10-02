#!/usr/bin/env python3
"""Lab-only PhysX probe: hold one fruit, load its stem, observe free fall.

This applies an external force directly to the fruit. It is not a robot grasp,
harvest attempt, or successful demonstration.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchardbench-asset", type=Path, required=True,
                        help="Prepared OrchardBench tree asset directory for this seed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path,
                        help="Optional JSON output path; stdout always receives the report")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.orchardbench_asset.is_dir():
        parser.error("--orchardbench-asset must name an existing directory")
    args.headless = True
    args.enable_cameras = False
    if not args.experience:
        root = Path(os.environ["ISAACLAB_ROOT"]).resolve()
        args.experience = str(root / "apps/isaaclab.python.headless.kit")
    portable_root = PROJECT_ROOT / "outputs/runtime/kit/physics"
    if "--portable-root" not in args.kit_args:
        args.kit_args = f"{args.kit_args} --portable-root {portable_root}".strip()
    return args


ARGS = parse_args()
APP = AppLauncher(ARGS, fast_shutdown=False, multi_gpu=False).app

import gymnasium as gym
import torch

import vla_isaaclab  # noqa: F401  Register Gym task.
from vla_isaaclab.policies.standing import StandingPolicy
from isaaclab.utils.math import quat_rotate_inverse


TASK = "VLA-OrchardPick-G1-JointPos-v0"


def position(env):
    return (env.scene["object"].data.root_pos_w[0] - env.scene.env_origins[0]).tolist()


def run_probe():
    spec = gym.spec(TASK)
    module_name, cfg_name = spec.kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(importlib.import_module(module_name), cfg_name)()
    cfg.sim.device = ARGS.device
    cfg.scene.num_envs = 1
    cfg.seed = ARGS.seed
    cfg.tree_model = "orchardbench"
    cfg.external_tree_asset = str(ARGS.orchardbench_asset.resolve())
    cfg.configure_layout(ARGS.seed)
    if len(cfg.orchard_layout.apples) != 1:
        raise ValueError("Stem probe requires exactly one physical fruit")
    for camera_name in ("cam_left_high", "cam_left_wrist", "cam_right_wrist"):
        setattr(cfg.scene, camera_name, None)

    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        env.reset(seed=ARGS.seed)
        policy = StandingPolicy(env)
        layout = cfg.orchard_layout
        apple = env.scene["object"]
        initial = position(env)
        passive_positions = []
        premature_terminal = False
        with torch.inference_mode():
            for step in range(120):
                _, _, terminated, timed_out, _ = env.step(policy.compute(step))
                passive_positions.append(position(env))
                if bool(terminated[0]) or bool(timed_out[0]):
                    premature_terminal = True
                    break
            max_passive_drift = max(
                (sum((p[i] - initial[i]) ** 2 for i in range(3)) ** 0.5
                 for p in passive_positions), default=float("inf"))
            passive_speed = float(torch.linalg.vector_norm(apple.data.root_lin_vel_w[0]))
            passive_broken = bool(env.orchard_broken)
            passive_stable = (not premature_terminal and not passive_broken
                              and max_passive_drift <= 0.015
                              and passive_speed <= 0.035)

            break_step = None
            applied_force = 2.0 * layout.stem_break_force
            world_force = torch.tensor([[applied_force, 0.0, 0.0]], device=env.device)
            zero = torch.zeros((1, 1, 3), device=env.device)
            if passive_stable:
                # Isaac Lab 2.0.2 takes the wrench in the fruit's local frame.
                # Recompute it from the measured orientation each control step
                # so the applied force remains horizontal in world coordinates.
                for step in range(120):
                    local_force = quat_rotate_inverse(
                        apple.data.root_quat_w, world_force).unsqueeze(1)
                    apple.set_external_force_and_torque(local_force, zero)
                    _, _, terminated, timed_out, _ = env.step(policy.compute(120 + step))
                    if layout.target_index in env.orchard_broken:
                        break_step = step + 1
                        break
                    if bool(terminated[0]) or bool(timed_out[0]):
                        break
            apple.set_external_force_and_torque(zero, zero)
            apple.write_data_to_sim()
            height_at_break = position(env)[2]
            min_height_after = height_at_break
            if break_step is not None:
                # OrchardPickEnv preserves terminal poses and rejects another
                # env.step after termination. Advance physics directly to measure
                # gravity after force removal, without resetting or task scoring.
                for _ in range(240):
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(cfg.sim.dt)
                    min_height_after = min(min_height_after, position(env)[2])
            fall_m = height_at_break - min_height_after

        passed = passive_stable and break_step is not None and fall_m >= 0.05
        return {
            "test": "independent_external_load_of_one_native_physx_stem",
            "passed": bool(passed),
            "harvest_success_claim": False,
            "robot_grasp_claim": False,
            "seed": ARGS.seed,
            "asset": str(ARGS.orchardbench_asset.resolve()),
            "fruit_count": len(layout.apples),
            "passive_steps": len(passive_positions),
            "passive_stable": passive_stable,
            "terminal_during_passive_hold": premature_terminal,
            "max_passive_drift_m": max_passive_drift,
            "passive_final_speed_m_s": passive_speed,
            "joint_broken_during_hold": passive_broken,
            "applied_world_x_force_n": applied_force if passive_stable else 0.0,
            "break_step_after_load": break_step,
            "broken_indices": sorted(env.orchard_broken),
            "height_at_break_m": height_at_break,
            "minimum_height_after_force_clear_m": min_height_after,
            "fall_after_force_clear_m": fall_m,
        }
    finally:
        env.close()


def main():
    try:
        report = run_probe()
    except Exception as exc:
        traceback.print_exc()
        report = {"test": "independent_external_load_of_one_native_physx_stem",
                  "passed": False, "harvest_success_claim": False,
                  "robot_grasp_claim": False, "error": str(exc)}
    print(json.dumps(report, indent=2), flush=True)
    if ARGS.report:
        ARGS.report.parent.mkdir(parents=True, exist_ok=True)
        ARGS.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
