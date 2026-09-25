"""G1 tabletop-apple grasp trial on Isaac Lab Arena 0.3 / Isaac Sim 6.1.

The apple is always a dynamic rigid body. Only G1 action commands move it.
A video is evidence of the rollout; the JSON separately reports whether the
named task success termination fired. This is a calibration baseline, not a
trained policy or a claim of successful transfer to hardware.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import (
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

# G1 WBC/PINK action: left/right hand, left pose, right pose, navigation,
# base height, torso RPY. Wrist poses are relative to the pelvis (xyzw).
IDLE = (
    0.0, 0.0, 0.201, 0.145, 0.101, 0.010, -0.008, -0.011, 1.0,
    0.201, -0.145, 0.101, -0.010, -0.008, -0.011, 1.0,
    0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0,
)
APPLE_NAME = "apple_01_objaverse_robolab"
PLATE_NAME = "clay_plates_hot3d_robolab"
TABLE_CENTER = (0.25, 0.20, 0.70)  # 0.04-m thick top; upper surface z=0.72 m
ROBOT_START = (-0.3, 0.0, 0.78)
APPLE_START = (-0.05, 0.10, 0.78)
PLATE_START = (0.0, 0.40, 0.74)


def phase_at_step(step: int, warmup: int, phase_steps: int) -> tuple[str, float]:
    """Return commanded phase and its normalized time; no simulator dependency."""
    if step < warmup:
        return "settle", step / max(1, warmup)
    names = ("pregrasp", "approach", "close", "lift", "transfer", "lower", "release", "retreat")
    segment, within = divmod(step - warmup, phase_steps)
    if segment >= len(names):
        return "hold", 1.0
    return names[segment], (within + 1) / phase_steps


def wrist_target(phase: str, apple: tuple[float, float, float], plate: tuple[float, float, float],
                 offset_x: float, offset_y: float, offset_z: float) -> tuple[float, float, float]:
    """Plan a wrist path in the pelvis frame from measured fruit and plate poses."""
    grasp = (apple[0] + offset_x, apple[1] + offset_y, apple[2] + offset_z)
    pregrasp = (grasp[0] - 0.08, grasp[1], grasp[2] + 0.09)
    above_plate = (plate[0] + offset_x, plate[1] + offset_y, plate[2] + 0.20)
    targets = {
        "settle": IDLE[2:5],
        "pregrasp": pregrasp,
        "approach": grasp,
        "close": grasp,
        "lift": (grasp[0], grasp[1], grasp[2] + 0.20),
        "transfer": above_plate,
        "lower": (above_plate[0], above_plate[1], above_plate[2] - 0.10),
        "release": (above_plate[0], above_plate[1], above_plate[2] - 0.10),
        "retreat": above_plate,
        "hold": IDLE[2:5],
    }
    return targets[phase]


def _as_torch(array):
    import torch
    import warp as wp
    return array if isinstance(array, torch.Tensor) else wp.to_torch(array)


def _world_points(env):
    """Read live positions; convert targets to the PINK pelvis frame."""
    from isaaclab.utils.math import matrix_from_quat

    scene = env.unwrapped.scene
    robot = scene["robot"]
    body_poses = _as_torch(robot.data.body_link_pose_w)
    pelvis = body_poses[0, robot.data.body_names.index("pelvis"), :7]
    rotation = matrix_from_quat(pelvis[3:7].unsqueeze(0))[0]
    apple_world = _as_torch(scene[APPLE_NAME].data.root_pos_w)[0]
    plate_world = _as_torch(scene[PLATE_NAME].data.root_pos_w)[0]
    left_wrist = body_poses[0, robot.data.body_names.index("left_wrist_yaw_link"), :3]
    local_apple = rotation.T @ (apple_world - pelvis[:3])
    local_plate = rotation.T @ (plate_world - pelvis[:3])
    local_wrist = rotation.T @ (left_wrist - pelvis[:3])
    return (
        tuple(float(x) for x in local_apple),
        tuple(float(x) for x in local_plate),
        float(apple_world[2]),
        float(plate_world[2]),
        float((apple_world - left_wrist).norm()),
        tuple(float(x) for x in local_wrist),
        tuple(float(x) for x in pelvis[:3]),
    )


def _build_env(args, output: Path, video_prefix: str, total_steps: int):
    import isaaclab.sim as sim_utils
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.g1.g1 import G1WBCPinkEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import G1PickAndPlaceMimicEnvCfg, PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    ground = registry.get_asset_by_name("ground_plane")()
    light = registry.get_asset_by_name("light")()
    table = registry.get_asset_by_name("procedural_table")()
    table.set_initial_pose(Pose(position_xyz=TABLE_CENTER, rotation_xyzw=(0, 0, 0, 1)))
    table.object_cfg.spawn = table.object_cfg.spawn.copy()
    table.object_cfg.spawn.visible = True
    table.object_cfg.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.96, 0.96, 0.96)
    )
    apple = registry.get_asset_by_name(APPLE_NAME)()
    plate = registry.get_asset_by_name(PLATE_NAME)()
    apple.set_initial_pose(Pose(position_xyz=APPLE_START, rotation_xyzw=(0, 0, 0, 1)))
    plate.set_initial_pose(Pose(position_xyz=PLATE_START, rotation_xyzw=(0, 0, 0, 1)))
    robot = G1WBCPinkEmbodiment(enable_cameras=True)
    # Arena's G1 root is at pelvis height. Start above the z=0 ground plane.
    robot.set_initial_pose(Pose(position_xyz=ROBOT_START, rotation_xyzw=(0, 0, 0, 1)))
    robot.set_finger_contact_friction(
        material_path="/World/Materials/g1_apple_fingers",
        static_friction=2.0,
        dynamic_friction=1.5,
        prim_name_markers=("finger", "thumb"),
    )

    def mimic_cfg(arm_mode):
        return G1PickAndPlaceMimicEnvCfg(
            pick_up_object_name=apple.name,
            destination_location_name=plate.name,
            arm_mode=arm_mode,
        )

    task = PickAndPlaceTask(
        apple, plate, table, episode_length_s=20.0,
        task_description="Pick up the apple from the table and place it on the plate.",
        force_threshold=0.1, velocity_threshold=0.03,
        placement_consecutive_steps=90,
        mimic_env_cfg_factory=mimic_cfg,
    )
    def configure(env_cfg):
        env_cfg = set_control_rate_50hz(env_cfg)
        env_cfg.sim.visualizer_cfgs = [
            KitVisualizerCfg(headless=True, eye=(-1.4, -1.2, 1.6), lookat=(0.05, 0.2, 0.4))
        ]
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:kit", output_dir=str(output),
                output_filename_prefix=f"{video_prefix}_overview",
                video_length=total_steps, fps=30, frame_stride=2,
            ),
            VideoRecorderCfg(
                source="sensor:robot_head_cam", output_dir=str(output),
                output_filename_prefix=f"{video_prefix}_head",
                video_length=total_steps, fps=30, frame_stride=2,
            ),
        ]
        return env_cfg

    arena_env = IsaacLabArenaEnvironment(
        name="g1_table_apple_grasp_trial", embodiment=robot,
        scene=Scene(assets=[ground, light, table, apple, plate]), task=task,
        env_cfg_callback=configure,
    )
    return ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args)).make_registered()


def run(args) -> int:
    import torch
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    total = args.warmup + 8 * args.phase_steps + args.hold_steps
    video_prefix = "g1_apple_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    env = _build_env(args, output, video_prefix, total)
    if env.unwrapped.num_envs != 1 or env.unwrapped.single_action_space.shape != (23,):
        raise RuntimeError("This baseline requires one G1 WBC/PINK environment with 23 actions")
    steps = []
    rest_height = None
    max_lift = 0.0
    near_hand_lift_steps = 0
    success = False
    success_step = None
    success_phase = None
    failure = None
    no_lift = False
    try:
        env.reset()
        previous = IDLE[2:5]
        grasp_reference = None
        for step in range(total):
            phase, alpha = phase_at_step(step, args.warmup, args.phase_steps)
            apple, plate, apple_height, plate_height, wrist_distance, wrist_actual, pelvis_world = _world_points(env)
            tracking_error = math.dist(wrist_actual, previous)
            if step == args.warmup:
                rest_height = apple_height
                grasp_reference = apple
                if apple_height < 0.55 or plate_height < 0.55:
                    failure = "fruit_or_plate_not_supported_on_table"
                    no_lift = True
                elif pelvis_world[2] < 0.45 or math.dist(pelvis_world[:2], ROBOT_START[:2]) > 0.25:
                    failure = "robot_unstable_during_settle"
                    no_lift = True
            if not no_lift and phase == "close" and alpha <= 1 / args.phase_steps and tracking_error > 0.08:
                failure = "wrist_did_not_reach_apple"
                no_lift = True
            if rest_height is not None:
                max_lift = max(max_lift, apple_height - rest_height)
                if (phase in ("lift", "transfer") and
                        apple_height - rest_height >= args.min_lift and wrist_distance < 0.20):
                    near_hand_lift_steps += 1
            if not no_lift and phase == "transfer" and alpha <= 1 / args.phase_steps and near_hand_lift_steps < 5:
                failure = "no_sustained_apple_lift_near_hand"
                no_lift = True
            if no_lift:
                phase = "hold"
            planned_apple = apple if phase in ("settle", "pregrasp", "approach") else grasp_reference
            target = wrist_target(phase, planned_apple, plate,
                                  args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z)
            # A bounded Cartesian command ramp avoids sudden wrist target jumps.
            if phase == "settle":
                command = IDLE[2:5]
            else:
                blend = min(args.max_wrist_step / max(math.dist(previous, target), 1e-9), 1.0)
                command = tuple(a + blend * (b - a) for a, b in zip(previous, target))
            previous = command
            action = torch.tensor(IDLE, device=env.unwrapped.device).unsqueeze(0)
            action[0, 2:5] = torch.tensor(command, device=env.unwrapped.device)
            action[0, 0] = float(phase in ("close", "lift", "transfer", "lower"))
            _, _, terminated, truncated, _ = env.step(action)
            task_success = bool(env.unwrapped.termination_manager.get_term("success")[0])
            if task_success:
                success = True
                success_step = step
                success_phase = phase
            steps.append({"step": step, "phase": phase, "apple_z_m": apple_height,
                          "plate_z_m": plate_height,
                          "apple_to_left_wrist_m": wrist_distance,
                          "apple_pelvis_m": list(apple),
                          "pelvis_world_m": list(pelvis_world),
                          "left_wrist_actual_pelvis_m": list(wrist_actual),
                          "wrist_command_error_m": tracking_error,
                          "left_wrist_target_pelvis_m": list(command), "success": task_success})
            if bool(terminated[0]) or bool(truncated[0]):
                if not success:
                    failure = failure or "episode_terminated_without_success"
                break
    finally:
        env.close()  # flushes native Isaac Lab recorders

    success_with_lift = success and near_hand_lift_steps >= 5
    verified_pick_and_place = success_with_lift and success_phase in ("retreat", "hold")
    if not verified_pick_and_place and failure is None:
        failure = "no_verified_post_release_placement"
    result = {
        "arena_version": "0.3.0", "simulator": "Isaac Sim 6.1",
        "seed": args.seed, "steps": len(steps), "apple": APPLE_NAME,
        "table_top_z_m": TABLE_CENTER[2] + 0.02,
        "apple_z_after_settle_m": rest_height,
        "plate_z_after_settle_m": steps[args.warmup]["plate_z_m"] if len(steps) > args.warmup else None,
        "pelvis_world_after_settle_m": steps[args.warmup]["pelvis_world_m"] if len(steps) > args.warmup else None,
        "max_lift_m": round(max_lift, 4),
        "pelvis_xy_drift_m": round(math.dist(steps[0]["pelvis_world_m"][:2], steps[-1]["pelvis_world_m"][:2]), 4) if steps else None,
        "sustained_lift_near_hand": near_hand_lift_steps >= 5,
        "near_hand_lift_steps": near_hand_lift_steps,
        "success_termination": success, "success_step": success_step,
        "success_phase": success_phase, "failure": failure,
        "success_with_lift_candidate": success_with_lift,
        "verified_pick_and_place": verified_pick_and_place,
        "video_files": [str(p) for p in output.glob(f"{video_prefix}*.mp4")],
        "min_hand_distance_m": round(min(x["apple_to_left_wrist_m"] for x in steps), 4) if steps else None,
        "approach_end_wrist_error_m": next(
            (round(x["wrist_command_error_m"], 4) for x in reversed(steps) if x["phase"] == "approach"), None
        ),
        "phase_min_hand_distance_m": {
            phase: round(min(x["apple_to_left_wrist_m"] for x in steps if x["phase"] == phase), 4)
            for phase in dict.fromkeys(x["phase"] for x in steps)
        },
        "action_type": "G1 WBC/PINK hand and wrist commands",
        "object_pose_overridden_during_rollout": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "trajectory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in steps))
    print(json.dumps(result, indent=2))
    return 0 if verified_pick_and_place else 2


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    if "--headless" not in parser._option_string_actions:
        parser.add_argument("--headless", action="store_true", help="Run with offscreen rendering")
    parser.add_argument("--output", default="outputs/arena_apple_pick")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--phase-steps", type=int, default=45)
    parser.add_argument("--hold-steps", type=int, default=60)
    parser.add_argument("--grasp-offset-x", type=float, default=-0.10)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--grasp-offset-z", type=float, default=0.015)
    parser.add_argument("--max-wrist-step", type=float, default=0.008)
    parser.add_argument("--min-lift", type=float, default=0.08)
    args = parser.parse_args()
    if args.num_envs != 1 or args.phase_steps < 1 or args.warmup < 1:
        parser.error("Use --num_envs 1, positive --phase-steps and positive --warmup")
    args.headless = True
    args.enable_cameras = True
    with SimulationAppContext(args):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
