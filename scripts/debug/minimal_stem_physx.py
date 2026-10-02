#!/usr/bin/env python3
"""One-fruit, one-joint Isaac Lab 2.0.2 PhysX diagnostic (no task environment).

Run one variant per process. A force is applied at the fruit center of mass,
as in check_orchard_stem.py; its offset from the surface joint creates torque.
The JSON `break_observed_substep` is the first Python observation of the PhysX
JOINT_BREAK event, not a claim about the exact instant inside the solver.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "assets/orchard/orchardbench_seed42/isaac/manifest.json"
DEFAULT_FORCE_N = 34.682148892074274
BREAK_TORQUE_NM = 0.35
EFFECTIVE_INFINITY = 1.0e30  # finite float32, vastly above this experiment's loads
PHYSICS_DT = 1.0 / 120.0
STEM_RADIUS_M = 0.003


def thresholds(mode: str, finite_force: float) -> tuple[float, float]:
    """The two independent native USD break thresholds."""
    if mode not in ("breakable", "unbreakable", "force-only", "torque-only"):
        raise ValueError(f"Unknown break mode: {mode}")
    return (
        finite_force if mode in ("breakable", "force-only") else EFFECTIVE_INFINITY,
        BREAK_TORQUE_NM if mode in ("breakable", "torque-only") else EFFECTIVE_INFINITY,
    )


def json_safe(value):
    """Encode failed numeric states as null while retaining all_values_finite=false."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def load_seed42_layout(path: Path = MANIFEST) -> dict:
    """Read the checked-in selected fruit, without importing the task or its robot."""
    manifest = json.loads(path.read_text())
    layout = manifest["layout"]
    if manifest["seed"] != 42 or layout["seed"] != 42 or len(layout["apples"]) != 1:
        raise ValueError("Expected the checked-in seed-42 single-fruit asset")
    center = tuple(layout["apples"][0])
    anchor = tuple(layout["stem_anchors"][0])
    radius = float(layout["radius"])
    mass = float(layout["mass"])
    span = math.dist(center, anchor)
    if not (math.isclose(radius, 0.03) and math.isclose(mass, 0.16)
            and math.isclose(anchor[0], center[0], abs_tol=1e-7)
            and math.isclose(anchor[1], center[1], abs_tol=1e-7)
            and anchor[2] > center[2] + radius):
        raise ValueError("Selected fruit geometry differs from this vertical-stem reproducer")
    return {"center": center, "anchor": anchor, "radius": radius, "mass": mass,
            "span": span, "stem_length": span - radius,
            "break_force": float(layout["stem_break_force"])}


def parse_args(argv: list[str] | None = None):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-lifecycle", choices=("precreate", "late-create"), default="late-create")
    parser.add_argument("--break-mode", choices=("breakable", "unbreakable", "force-only", "torque-only"),
                        default="breakable")
    parser.add_argument("--collisions", choices=("on", "off"), default="on")
    parser.add_argument("--force-n", type=float, default=DEFAULT_FORCE_N)
    parser.add_argument("--passive-steps", type=int, default=120)
    parser.add_argument("--loaded-steps", type=int, default=120)
    parser.add_argument("--post-break-steps", type=int, default=240)
    parser.add_argument("--report", type=Path)
    AppLauncher.add_app_launcher_args(parser)  # supplies --device cpu|cuda:0
    args = parser.parse_args(argv)
    if args.device not in ("cpu", "cuda:0"):
        parser.error("--device must be cpu or cuda:0")
    if not math.isfinite(args.force_n) or args.force_n < 0:
        parser.error("--force-n must be finite and nonnegative")
    for name in ("passive_steps", "loaded_steps", "post_break_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.headless = True
    args.enable_cameras = False
    if not args.experience:
        args.experience = str(Path(os.environ["ISAACLAB_ROOT"]).resolve()
                              / "apps/isaaclab.python.headless.kit")
    portable_root = PROJECT_ROOT / "outputs/runtime/kit/physics"
    if "--portable-root" not in args.kit_args:
        args.kit_args = f"{args.kit_args} --portable-root {portable_root}".strip()
    return args


def run(args):
    # Imports which initialize Omniverse must happen after AppLauncher starts.
    import omni.physx
    import omni.usd
    import torch
    from omni.physx.bindings._physx import SimulationEvent
    from pxr import Gf, PhysicsSchemaTools, UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.utils.math import quat_rotate_inverse

    layout = load_seed42_layout()
    force_limit, torque_limit = thresholds(args.break_mode, layout["break_force"])
    center, anchor = layout["center"], layout["anchor"]
    stem_bottom_z = center[2] + layout["radius"]
    stem_center = (center[0], center[1], (stem_bottom_z + anchor[2]) / 2)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    stage = omni.usd.get_context().get_stage()
    # Both bodies have rigid-body dynamics in every variant. Collision-off only
    # disables collision geometry; it does not remove either body or the joint.
    collision_enabled = args.collisions == "on"
    stem_path, fruit_path, joint_path = "/World/Stem_0", "/World/Apple_00", "/World/FruitJoint_0"
    stem_cfg = sim_utils.CylinderCfg(
        radius=STEM_RADIUS_M, height=layout["stem_length"],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision_enabled),
    )
    stem_cfg.func(stem_path, stem_cfg, translation=stem_center)
    fruit_cfg = RigidObjectCfg(
        prim_path=fruit_path,
        spawn=sim_utils.SphereCfg(
            radius=layout["radius"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=layout["mass"]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision_enabled),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.4, dynamic_friction=1.1, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=center),
    )
    fruit = RigidObject(fruit_cfg)

    def create_joint():
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([stem_path])
        joint.CreateBody1Rel().SetTargets([fruit_path])
        # Local frames exactly match stem_geometry() for this vertical stem.
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, -layout["stem_length"] / 2))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, layout["radius"]))
        joint.CreateLocalRot0Attr(Gf.Quatf(1))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateBreakForceAttr(force_limit)
        joint.CreateBreakTorqueAttr(torque_limit)
        joint.CreateExcludeFromArticulationAttr(True)

    events: list[dict] = []
    current_substep = 0

    def on_event(event):
        if event.type == int(SimulationEvent.JOINT_BREAK):
            path = str(PhysicsSchemaTools.decodeSdfPath(*event.payload["jointPath"]))
            if path == joint_path:
                events.append({"observed_substep": current_substep, "path": path})

    subscription = (omni.physx.get_physx_interface().get_simulation_event_stream_v2()
                    .create_subscription_to_pop(on_event))
    if args.joint_lifecycle == "precreate":
        create_joint()
    sim.reset()  # initializes the fruit view and starts physics
    fruit.write_root_state_to_sim(fruit.data.default_root_state.clone())
    fruit.update(PHYSICS_DT)
    if args.joint_lifecycle == "late-create":
        create_joint()  # runtime topology mutation, analogous to the task reset

    def snapshot(phase: str, phase_step: int) -> dict:
        state = fruit.data.root_state_w[0]
        values = state.tolist()
        return {"phase": phase, "phase_step": phase_step, "substep": current_substep,
                "position_m": values[:3], "linear_velocity_m_s": values[7:10],
                "finite": all(math.isfinite(v) for v in values),
                "break_event_seen": bool(events)}

    zero = torch.zeros((1, 1, 3), dtype=torch.float32, device=args.device)
    world_force = torch.tensor([[args.force_n, 0, 0]], dtype=torch.float32, device=args.device)
    initial = snapshot("initial", 0)
    all_finite = initial["finite"]
    first_break_sample = None

    def emit(sample):
        print(f"{sample['phase']:7s} {sample['phase_step']:3d} φ={sample['substep']:4d} "
              f"z={sample['position_m'][2]:+.5f} vx={sample['linear_velocity_m_s'][0]:+.5f} "
              f"vz={sample['linear_velocity_m_s'][2]:+.5f} break={int(sample['break_event_seen'])} "
              f"finite={int(sample['finite'])}", flush=True)

    def one_step(phase: str, phase_step: int, apply_load: bool) -> dict:
        nonlocal current_substep, all_finite, first_break_sample
        current_substep += 1
        # Isaac Lab 2.0.2 stores a local-frame wrench, then flushes it from
        # write_data_to_sim immediately before each individual physics step.
        local_force = (quat_rotate_inverse(fruit.data.root_quat_w, world_force).unsqueeze(1)
                       if apply_load else zero)
        fruit.set_external_force_and_torque(local_force, zero)
        fruit.write_data_to_sim()
        sim.step(render=False)
        fruit.update(PHYSICS_DT)
        sample = snapshot(phase, phase_step)
        if events and first_break_sample is None:
            first_break_sample = sample
        all_finite &= sample["finite"]
        return sample

    passive = []
    for i in range(1, args.passive_steps + 1):
        sample = one_step("passive", i, False)
        passive.append(sample)
        if i > args.passive_steps - 2:
            emit(sample)
        if not all_finite or events:
            break
    drift = max((math.dist(s["position_m"], initial["position_m"])
                 for s in passive if s["finite"]), default=float("inf") if passive else 0.0)
    passive_final_speed = math.dist(passive[-1]["linear_velocity_m_s"], (0, 0, 0)) if passive else 0.0
    passive_stable = (len(passive) == args.passive_steps and not events and all_finite
                      and drift <= 0.015 and passive_final_speed <= 0.035)

    break_sample = None
    loaded_count = 0
    loaded_samples = []
    if passive_stable:
        print(f"load_begin next_substep={current_substep + 1} force_n={args.force_n}", flush=True)
        for i in range(1, args.loaded_steps + 1):
            sample = one_step("loaded", i, True)
            loaded_samples.append(sample)
            emit(sample)
            loaded_count = i
            if events:
                break_sample = sample
                # Clear at the first Python observation, before another step.
                fruit.set_external_force_and_torque(zero, zero)
                fruit.write_data_to_sim()
                break
            if not all_finite:
                break
    fruit.set_external_force_and_torque(zero, zero)
    fruit.write_data_to_sim()
    height_at_clear = (break_sample or (loaded_samples[-1] if loaded_samples else
                                      passive[-1] if passive else initial))["position_m"][2]
    min_height = height_at_clear
    post_count = 0
    if all_finite and passive_stable:
        for i in range(1, args.post_break_steps + 1):
            sample = one_step("post", i, False)
            post_count = i
            min_height = min(min_height, sample["position_m"][2])
            if i <= 3 or i == args.post_break_steps:
                emit(sample)
            if not all_finite:
                break

    fall = height_at_clear - min_height
    break_observed = bool(events)
    loaded_drift = max((math.dist(s["position_m"], initial["position_m"])
                        for s in loaded_samples if s["finite"]),
                       default=float("inf") if loaded_samples else 0.0)
    loaded_final_speed = (math.dist(loaded_samples[-1]["linear_velocity_m_s"], (0, 0, 0))
                          if loaded_samples else 0.0)
    held_under_load = (loaded_count == args.loaded_steps and not break_observed
                       and loaded_drift <= 0.015 and loaded_final_speed <= 0.035)
    # A no-break low-load trial is an observation, not proof that a force path
    # works. The default breakable baseline, however, requires break and fall.
    expected_break = (args.break_mode in ("breakable", "force-only", "torque-only")
                      and args.force_n >= DEFAULT_FORCE_N)
    unbreakable_ok = args.break_mode != "unbreakable" or not break_observed
    passed = (all_finite and passive_stable and unbreakable_ok and
              ((break_observed and fall >= 0.05) if expected_break else
               (fall >= 0.05 if break_observed else held_under_load)))
    outcome = ("nonfinite" if not all_finite else "passive_unstable" if not passive_stable
               else "broke_and_fell" if break_observed and fall >= 0.05
               else "break_without_measured_fall" if break_observed else "held_no_break")
    return {
        "test": "minimal_native_physx_stem_joint", "device": args.device,
        "joint_lifecycle": args.joint_lifecycle, "break_mode": args.break_mode,
        "collisions": args.collisions, "requested_force_n": args.force_n,
        "effective_breakForce_n": force_limit, "effective_breakTorque_nm": torque_limit,
        "physics_dt_s": PHYSICS_DT, "initial_fruit_pose": initial["position_m"] +
        fruit.data.default_root_state[0, 3:7].tolist(),
        "passive_steps_completed": len(passive), "loaded_steps_completed": loaded_count,
        "post_break_steps_completed": post_count, "passive_drift_m": drift,
        "passive_final_speed_m_s": passive_final_speed, "passive_stable": passive_stable,
        "loaded_drift_m": loaded_drift, "loaded_final_speed_m_s": loaded_final_speed,
        "held_under_load": held_under_load,
        "break_detected": break_observed,
        "break_observed_substep": events[0]["observed_substep"] if events else None,
        "position_at_break_m": first_break_sample["position_m"] if first_break_sample else None,
        "velocity_at_break_m_s": first_break_sample["linear_velocity_m_s"] if first_break_sample else None,
        "nominal_center_force_torque_nm": layout["radius"] * args.force_n,
        "expected_break_for_numeric_gate": expected_break,
        "height_at_force_clear_m": height_at_clear,
        "min_fruit_height_afterward_m": min_height, "fall_distance_m": fall,
        "all_values_finite": all_finite, "outcome": outcome, "passed": passed,
        "numeric_checks_passed": passed, "process_success": passed,
        "backend_validation": "requires_clean_PhysX_log",
        "harvest_success_claim": False, "robot_grasp_claim": False,
        "break_detection_limitation": "JOINT_BREAK callback is timestamped at first Python observation; "
        "callback delivery may lag the solver substep.",
        "physics_error_limitation": "PhysX/Carb errors are not captured in JSON; inspect the process log "
        "even when numeric state remains finite.",
    }


def main():
    args = parse_args()
    from isaaclab.app import AppLauncher
    app = AppLauncher(args, fast_shutdown=False, multi_gpu=False).app
    exit_code = 1
    try:
        try:
            report = run(args)
        except Exception as exc:
            traceback.print_exc()
            report = {"test": "minimal_native_physx_stem_joint", "device": args.device,
                      "joint_lifecycle": args.joint_lifecycle, "break_mode": args.break_mode,
                      "collisions": args.collisions, "requested_force_n": args.force_n,
                      "process_success": False, "passed": False, "error": str(exc),
                      "harvest_success_claim": False, "robot_grasp_claim": False}
        report = json_safe(report)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        exit_code = 0 if report["passed"] else 1
    finally:
        try:
            app.close()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                exit_code = 1
        except BaseException:
            traceback.print_exc()
            exit_code = 1
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)


if __name__ == "__main__":
    main()
