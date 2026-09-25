"""G1 tabletop-apple grasp trial on Isaac Lab Arena 0.3 / Isaac Sim 6.1.

The apple is always a dynamic rigid body. Only G1 action commands move it.
A video is evidence of the rollout; the JSON separately reports whether the
named task success termination fired. This is a calibration baseline, not a
trained policy or a claim of successful transfer to hardware.
"""

from __future__ import annotations

import json
import math
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
APPLE_START = (0.15, 0.15, 0.05)
PLATE_START = (0.15, 0.40, 0.02)


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
    pelvis = _as_torch(robot.data.body_link_state_w)[0, robot.data.body_names.index("pelvis"), :7]
    rotation = matrix_from_quat(pelvis[3:7].unsqueeze(0))[0]
    apple_world = _as_torch(scene[APPLE_NAME].data.root_pos_w)[0]
    plate_world = _as_torch(scene[PLATE_NAME].data.root_pos_w)[0]
    left_wrist = _as_torch(robot.data.body_link_state_w)[
        0, robot.data.body_names.index("left_wrist_yaw_link"), :3
    ]
    local_apple = rotation.T @ (apple_world - pelvis[:3])
    local_plate = rotation.T @ (plate_world - pelvis[:3])
    return (
        tuple(float(x) for x in local_apple),
        tuple(float(x) for x in local_plate),
        float(apple_world[2]),
        float((apple_world - left_wrist).norm()),
    )


def _build_env(args):
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.g1.g1 import G1WBCPinkEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import G1PickAndPlaceMimicEnvCfg, PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    table = registry.get_asset_by_name("table")()
    apple = registry.get_asset_by_name(APPLE_NAME)()
    plate = registry.get_asset_by_name(PLATE_NAME)()
    apple.set_initial_pose(Pose(position_xyz=APPLE_START, rotation_xyzw=(0, 0, 0, 1)))
    plate.set_initial_pose(Pose(position_xyz=PLATE_START, rotation_xyzw=(0, 0, 0, 1)))
    robot = G1WBCPinkEmbodiment(enable_cameras=False)
    robot.set_initial_pose(Pose(position_xyz=(-0.4, 0, 0), rotation_xyzw=(0, 0, 0, 1)))
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
        force_threshold=0.1, velocity_threshold=0.1,
        mimic_env_cfg_factory=mimic_cfg,
    )
    arena_env = IsaacLabArenaEnvironment(
        name="g1_table_apple_grasp_trial", embodiment=robot,
        scene=Scene(assets=[table, apple, plate]), task=task,
        env_cfg_callback=set_control_rate_50hz,
    )
    return ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args)).make_registered(
        render_mode="rgb_array"
    )


def run(args) -> int:
    import torch
    from gymnasium.wrappers import RecordVideo

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    total = args.warmup + 8 * args.phase_steps + args.hold_steps
    env = _build_env(args)
    if env.unwrapped.num_envs != 1 or env.unwrapped.single_action_space.shape != (23,):
        raise RuntimeError("This baseline requires one G1 WBC/PINK environment with 23 actions")
    env = RecordVideo(env, video_folder=str(output), name_prefix="g1_apple_pick",
                      step_trigger=lambda step: step == 0, video_length=total,
                      disable_logger=True)
    steps = []
    rest_height = None
    max_lift = 0.0
    near_hand_lift_steps = 0
    success = False
    failure = None
    try:
        env.reset()
        previous = IDLE[2:5]
        grasp_reference = None
        for step in range(total):
            phase, alpha = phase_at_step(step, args.warmup, args.phase_steps)
            apple, plate, apple_height, wrist_distance = _world_points(env)
            if step == args.warmup:
                rest_height = apple_height
                grasp_reference = apple
            if rest_height is not None:
                max_lift = max(max_lift, apple_height - rest_height)
                if (phase in ("lift", "transfer") and
                        apple_height - rest_height >= args.min_lift and wrist_distance < 0.20):
                    near_hand_lift_steps += 1
            if phase == "transfer" and alpha <= 1 / args.phase_steps and near_hand_lift_steps < 5:
                failure = "no_sustained_apple_lift_near_hand"
                break
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
            steps.append({"step": step, "phase": phase, "apple_z_m": apple_height,
                          "apple_to_left_wrist_m": wrist_distance,
                          "left_wrist_target_pelvis_m": list(command), "success": task_success})
            if bool(terminated[0]) or bool(truncated[0]):
                if not success:
                    failure = "episode_terminated_without_success"
                break
    finally:
        env.close()  # flushes MP4 even when the episode terminates early

    result = {
        "arena_version": "0.3.0", "simulator": "Isaac Sim 6.1",
        "seed": args.seed, "steps": len(steps), "apple": APPLE_NAME,
        "max_lift_m": round(max_lift, 4),
        "sustained_lift_near_hand": near_hand_lift_steps >= 5,
        "near_hand_lift_steps": near_hand_lift_steps,
        "success_termination": success, "failure": failure,
        "success_with_lift_candidate": success and near_hand_lift_steps >= 5,
        "video_files": [str(p) for p in output.glob("*.mp4")],
        "action_type": "G1 WBC/PINK hand and wrist commands",
        "object_pose_overridden_during_rollout": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "trajectory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in steps))
    print(json.dumps(result, indent=2))
    return 0 if result["success_with_lift_candidate"] else 2


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--output", default="outputs/arena_apple_pick")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--phase-steps", type=int, default=45)
    parser.add_argument("--hold-steps", type=int, default=20)
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
