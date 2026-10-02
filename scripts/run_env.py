#!/usr/bin/env python3
"""Run a registered VLA Isaac Lab environment."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def write_exit_status(code):
    if os.environ.get("VLA_RUN_STATUS_FILE"):
        Path(os.environ["VLA_RUN_STATUS_FILE"]).write_text(json.dumps({"exit_code": code}))


# Registration is pure Python; listing tasks must not initialize a GPU renderer.
if "--list-tasks" in sys.argv:
    import gymnasium as gym
    import vla_isaaclab
    print(json.dumps(sorted(key for key in gym.registry if key.startswith("VLA-")), indent=2))
    write_exit_status(0)
    raise SystemExit(0)

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="VLA-ScenePreview-YCB-G1-v0", help="Registered Gym environment ID.")
    parser.add_argument("--policy", choices=("auto", "standing", "sugar-box", "orchard"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-dir", type=Path, help="Save this run's validation.json in a separate directory.")
    parser.add_argument("--steps", type=int, default=300, help="Control steps; 0 keeps a GUI run open.")
    parser.add_argument("--preview-video", type=Path)
    parser.add_argument("--camera-videos", type=Path, help="New directory for profile-specific H.264 camera inspection videos.")
    parser.add_argument("--orchard-tree-asset", type=Path,
                        help="Prepared visual tree directory (tree.usda and manifest.json); active orchard only.")
    parser.add_argument("--orchardbench-asset", type=Path,
                        help="Compiled canonical tree directory; requires --orchard-tree-model orchardbench.")
    parser.add_argument("--orchard-tree-model", choices=("commercial", "legacy", "orchardbench"), default="orchardbench",
                        help="Pinned OrchardBench tree (default), rejected commercial prototype, or legacy calibration fixture.")
    parser.add_argument("--record-format", choices=("none", "hdf5", "lerobot"), default="none")
    parser.add_argument(
        "--lerobot-version", choices=("3", "2.1"), default="3",
        help="LeRobot output format used with --record-format lerobot (default: 3).",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--dataset-name", default="vla_demo")
    parser.add_argument("--task-prompt")
    parser.add_argument(
        "--include-failed-episodes", action="store_true",
        help="Export failed episodes for contract/source review instead of imitation training.",
    )
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--physics-only", action="store_true")
    parser.add_argument("--render-profile", choices=("sensor", "stock"), default="sensor")
    parser.add_argument("--render-smoke", action="store_true", help="Render a lit cube before loading any robot/task.")
    parser.add_argument("--grasp-offset", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Orchard palm offset from apple center in world axes (meters).")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = not args.physics_only
    if args.render_smoke and args.physics_only:
        parser.error("--render-smoke requires rendering")
    return args


ARGS = parse_args()
isaaclab_root = Path(os.environ["ISAACLAB_ROOT"]).resolve()
if not ARGS.experience:
    experience_name = (
        "isaaclab.python.headless.kit"
        if ARGS.physics_only
        else (
            "isaaclab.python.headless.rendering.kit"
            if ARGS.headless
            else "isaaclab.python.rendering.kit"
        )
    )
    ARGS.experience = str(isaaclab_root / "apps" / experience_name)
    if ARGS.headless and not ARGS.physics_only and ARGS.render_profile == "sensor":
        ARGS.experience = str(PROJECT_ROOT / "configs/orchard.sensor.kit")

runtime_profile = "physics" if ARGS.physics_only else f"rendering-{ARGS.render_profile}"
portable_root = PROJECT_ROOT / "outputs/runtime/kit" / runtime_profile
if "--portable-root" not in ARGS.kit_args:
    ARGS.kit_args = f"{ARGS.kit_args} --portable-root {portable_root}".strip()
APP = AppLauncher(ARGS, fast_shutdown=False, multi_gpu=False).app

import gymnasium as gym
import numpy as np
import torch

import vla_isaaclab  # noqa: F401  Register environments.
from isaaclab.managers import DatasetExportMode

from vla_isaaclab.envs.common import LOWER_BODY_JOINT_NAMES, SUPPORT_HEIGHT
from vla_isaaclab.envs.common.managers import ACTION_TERM_NAME
from vla_isaaclab.policies import StandingPolicy, YCBSugarBoxScriptedPolicy


ENV_PREFIX = "VLA-"


def registered_tasks():
    return sorted(env_id for env_id in gym.registry if env_id.startswith(ENV_PREFIX))


def make_policy(env):
    selected = ARGS.policy
    if selected == "auto":
        selected = {"VLA-YCBSugarBox-G1-JointPos-v0": "sugar-box",
                    "VLA-OrchardPick-G1-JointPos-v0": "orchard"}.get(ARGS.task, "standing")
        if selected == "orchard" and env.cfg.orchard_layout.tree_model in ("commercial", "orchardbench"):
            # The old grasp script is not a verified controller for this tree.
            # This milestone defaults to a stable scene for an external controller.
            selected = "standing"
            print("[INFO] Commercial tree: auto selects standing; no trained harvesting policy is bundled.")
    if selected == "orchard":
        from vla_isaaclab.policies.orchard_pick import OrchardScriptedPolicy
        return OrchardScriptedPolicy(env)
    return YCBSugarBoxScriptedPolicy(env) if selected == "sugar-box" else StandingPolicy(env)


def task_succeeded(env) -> bool:
    if "success" not in env.termination_manager.active_terms:
        return False
    return bool(env.termination_manager.get_term("success")[0].item())


def project_revision() -> dict:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=False,
        capture_output=True, text=True,
    )
    # rsync-based lab tests may intentionally omit .git. Pin the actual Python
    # source bytes as well, including uncommitted/new files in normal checkouts.
    digest = hashlib.sha256()
    for folder in ("scripts", "src"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*.py")):
            digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    if revision.returncode:
        return {"git_commit": None, "working_tree_dirty": None, "python_source_sha256": digest.hexdigest()}
    commit = revision.stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    )
    return {"git_commit": commit, "working_tree_dirty": dirty, "python_source_sha256": digest.hexdigest()}


def primary_camera(env):
    # Preview videos should show the full task from the external scene camera.
    # Robot-mounted cameras remain available to the dataset recording adapter.
    for name in ("cam_side", "camera", "cam_left_high"):
        if name in env.scene.sensors:
            return env.scene.sensors[name]
    raise RuntimeError("No scene or head camera is configured")


def validate(env, policy, steps, success_count, dataset_path=None):
    robot = env.scene["robot"]
    lower_ids, _ = robot.find_joints(list(LOWER_BODY_JOINT_NAMES), preserve_order=True)
    lower_error = torch.max(
        torch.abs(robot.data.joint_pos[:, lower_ids] - robot.data.default_joint_pos[:, lower_ids])
    ).item()
    lower_errors = torch.abs(
        robot.data.joint_pos[0, lower_ids] - robot.data.default_joint_pos[0, lower_ids]
    )
    objects = {}
    objects_valid = True
    for name, obj in env.scene.rigid_objects.items():
        speed = torch.linalg.vector_norm(obj.data.root_vel_w[0]).item()
        finite = bool(torch.isfinite(obj.data.root_state_w[0]).all().item())
        above_support = obj.data.root_pos_w[0, 2].item() > SUPPORT_HEIGHT - 0.03
        stable = speed < 0.08
        objects[name] = {
            "position_m": obj.data.root_pos_w[0].detach().cpu().tolist(),
            "speed_norm": speed,
            "finite": finite,
            "above_support": above_support,
            "stable": stable,
        }
        objects_valid = objects_valid and finite and above_support and stable

    articulations = {}
    for name, articulation in env.scene.articulations.items():
        if name == "robot":
            continue
        finite = bool(torch.isfinite(articulation.data.joint_pos[0]).all().item())
        articulations[name] = {
            "joint_names": list(articulation.joint_names),
            "joint_positions_rad": articulation.data.joint_pos[0].detach().cpu().tolist(),
            "joint_velocities_rad_s": articulation.data.joint_vel[0].detach().cpu().tolist(),
            "finite": finite,
        }
        objects_valid = objects_valid and finite

    term = env.action_manager.get_term(ACTION_TERM_NAME)
    report = {
        "passed": lower_error < 0.05 and objects_valid,
        "task": ARGS.task,
        "steps": steps,
        "rates_hz": {"physics": 120, "control": 30, "camera": 30},
        "action_dimension": env.action_manager.total_action_dim,
        "controlled_joints": list(term._joint_names),
        "max_lower_body_default_pose_error_rad": lower_error,
        "lower_body_default_pose_error_rad": {
            name: error
            for name, error in zip(LOWER_BODY_JOINT_NAMES, lower_errors.detach().cpu().tolist())
        },
        "lower_body_position_rad": {
            name: value
            for name, value in zip(
                LOWER_BODY_JOINT_NAMES,
                robot.data.joint_pos[0, lower_ids].detach().cpu().tolist(),
            )
        },
        "lower_body_target_rad": {
            name: value
            for name, value in zip(
                LOWER_BODY_JOINT_NAMES,
                robot.data.joint_pos_target[0, lower_ids].detach().cpu().tolist(),
            )
        },
        "rigid_objects": objects,
        "articulations": articulations,
        "success_count": success_count,
        "termination_terms": {
            name: bool(env.termination_manager.get_term(name)[0].item())
            for name in env.termination_manager.active_terms
        },
        "preview_video": str(ARGS.preview_video.resolve()) if ARGS.preview_video else None,
        "recording_format": ARGS.record_format,
        "lerobot_version": ARGS.lerobot_version if ARGS.record_format == "lerobot" else None,
        "dataset": str(dataset_path) if dataset_path else None,
    }
    if hasattr(policy, "diagnostics"):
        report["policy"] = policy.diagnostics()
    if hasattr(env.cfg, "orchard_layout"):
        report["orchard"] = {
            "layout": env.cfg.orchard_layout.metadata(),
            "metrics": getattr(env, "orchard_last_metrics", {}),
            "fixed_base": True,
            "camera_videos": str(ARGS.camera_videos) if ARGS.camera_videos else None,
            "camera_profile": env.cfg.camera_profile,
            "tree_asset": getattr(env.cfg, "tree_asset_metadata", None),
            "tree_model": env.cfg.orchard_layout.tree_model,
            "tree_structure": getattr(env.cfg, "commercial_tree_summary", None),
        }
    if "success" in env.termination_manager.active_terms:
        report["passed"] = bool(
            report["passed"] and success_count > 0 and not report.get("policy", {}).get("failed", False)
        )
    return report


def main() -> int:
    if ARGS.render_smoke:
        from render_smoke import render_smoke
        return render_smoke(ARGS.report_dir or PROJECT_ROOT / "outputs/render_smoke", ARGS.device)
    if ARGS.list_tasks:
        tasks = registered_tasks()
        if not tasks:
            raise RuntimeError("No VLA Gym environments were registered")
        print(json.dumps(tasks, indent=2), flush=True)
        return 0
    if ARGS.task not in registered_tasks():
        raise ValueError(f"Unknown VLA environment {ARGS.task!r}. Available: {registered_tasks()}")
    if ARGS.orchard_tree_asset and ARGS.task != "VLA-OrchardPick-G1-JointPos-v0":
        raise ValueError("--orchard-tree-asset requires the active orchard-pick task")
    if (ARGS.preview_video or ARGS.camera_videos) and ARGS.physics_only:
        raise ValueError("Video output requires rendering; remove --physics-only")
    if ARGS.record_format != "none" and ARGS.physics_only:
        raise ValueError("recording requires cameras; remove --physics-only")
    if ARGS.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if ARGS.record_format != "lerobot" and ARGS.episodes != 1:
        raise ValueError("multiple episodes currently require --record-format lerobot")

    # Avoid importing the optional training task stack (isaaclab_rl/gym) for a
    # locally registered environment. Resolve its documented config entry point.
    import importlib
    module_name, config_name = gym.spec(ARGS.task).kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(importlib.import_module(module_name), config_name)()
    cfg.sim.device = ARGS.device
    cfg.scene.num_envs = 1
    cfg.seed = ARGS.seed
    if ARGS.task == "VLA-OrchardPick-G1-JointPos-v0":
        if ARGS.episodes != 1:
            raise ValueError("Run one orchard seed per process; stem resets and batch collection need lab validation")
        cfg.tree_model = ARGS.orchard_tree_model
        if ARGS.orchardbench_asset:
            if cfg.tree_model != "orchardbench":
                raise ValueError("--orchardbench-asset requires --orchard-tree-model orchardbench")
            cfg.external_tree_asset = str(ARGS.orchardbench_asset.resolve())
        cfg.configure_layout(ARGS.seed)
        if ARGS.orchard_tree_asset:
            if cfg.tree_model != "legacy":
                raise ValueError("--orchard-tree-asset is only for --orchard-tree-model legacy; commercial tree builds its own matched geometry")
            from isaaclab.assets import AssetBaseCfg
            import isaaclab.sim as sim_utils
            from vla_isaaclab.envs.orchard_pick.tree_asset import validate_tree_asset
            tree_path, tree_manifest = validate_tree_asset(ARGS.orchard_tree_asset, cfg.orchard_layout)
            cfg.scene.foliage = None
            cfg.scene.tree_visual = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/TreeVisual",
                spawn=sim_utils.UsdFileCfg(usd_path=str(tree_path)),
            )
            cfg.tree_asset_metadata = tree_manifest
        if ARGS.grasp_offset is not None:
            if not all(np.isfinite(ARGS.grasp_offset)):
                raise ValueError("Grasp offset must be finite")
            cfg.grasp_offset = tuple(ARGS.grasp_offset)
    finite_steps = ARGS.steps if ARGS.steps > 0 else 1800
    episode_steps = finite_steps if ARGS.record_format == "hdf5" else finite_steps + 1
    cfg.episode_length_s = episode_steps * cfg.decimation * cfg.sim.dt
    if ARGS.physics_only:
        for camera_name in ("camera", "cam_side", "cam_left_high", "cam_left_wrist", "cam_right_wrist"):
            if hasattr(cfg.scene, camera_name):
                setattr(cfg.scene, camera_name, None)
    dataset_path = None
    if ARGS.record_format == "hdf5":
        from vla_isaaclab.recording.recorder_cfg import VLARecorderCfg

        dataset_path = PROJECT_ROOT / "outputs/datasets" / f"{Path(ARGS.dataset_name).stem}.hdf5"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.recorders = VLARecorderCfg()
        cfg.recorders.dataset_export_dir_path = str(dataset_path.parent)
        cfg.recorders.dataset_filename = dataset_path.stem
        cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

    env = gym.make(ARGS.task, cfg=cfg).unwrapped
    video_container = None
    video_stream = None
    camera_videos = None
    try:
        env.reset(seed=ARGS.seed)
        if ARGS.camera_videos:
            from preview_views import PreviewViews
            camera_videos = PreviewViews(env, ARGS.camera_videos)
        if ARGS.record_format == "hdf5":
            env.recorder_manager._dataset_file_handler.add_env_args(
                {
                    "environment_id": ARGS.task,
                    "task_instruction": cfg.task_instruction,
                    "rates_hz": {"physics": 120, "control": 30, "camera": 30},
                }
            )
        policy = make_policy(env)
        if ARGS.preview_video:
            import av

            path = ARGS.preview_video.resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            video_container = av.open(str(path), mode="w")
            video_stream = video_container.add_stream("libx264", rate=30)
            video_stream.width = 640
            video_stream.height = 480
            video_stream.pix_fmt = "yuv420p"

        steps = 0
        success_count = 0
        if ARGS.record_format == "lerobot":
            from vla_isaaclab.recording.frame import EnvironmentFrameAdapter
            from vla_isaaclab.recording.staging import StagingHDF5Writer

            adapter = EnvironmentFrameAdapter(env)
            task_prompt = ARGS.task_prompt or cfg.task_instruction
            is_sugar = ARGS.task == "VLA-YCBSugarBox-G1-JointPos-v0"
            is_orchard = ARGS.task == "VLA-OrchardPick-G1-JointPos-v0"
            collection = {
                "contract_version": "1.0",
                "dataset_profile": "g1_29body_dex3_43d_v1",
                "robot_type": "Unitree_G1",
                "robot_configuration": "G1 29 body joints + left/right Dex3-1; fixed simulation base excluded from vectors",
                "robot_identifier": "unitree-official-g1-29dof-dex3-base-fix-usd",
                "hand_identifiers": {"left": "Dex3-1-sim", "right": "Dex3-1-sim"},
                "real_or_sim": "sim",
                "fps": 30,
                "joint_units": "rad",
                "joint_coordinate_mode": "PR",
                "sdk_mode_pr": 0,
                "joint_reference_and_signs": (
                    "Explicit simulator-to-contract mapping in envs/common/g1.py; no mirroring, "
                    "absolute-value conversion, normalization, or A/B actuator coordinates"
                ),
                "actuator_to_joint_conversion": "none; simulator articulation exposes PR joint coordinates",
                "collection_date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                "collector": os.environ.get("USER", "unknown"),
                "software": {
                    "recorder": f"vla_isaaclab LeRobot v{ARGS.lerobot_version} exporter",
                    "isaac_sim": "4.5.0.0",
                    "isaac_lab": "v2.0.2",
                    **project_revision(),
                },
                "calibration_identifier": "Isaac Sim USD articulation and pinhole camera configuration",
                "cameras": adapter.camera_metadata,
                "video_encoding": "AV1/yuv420p, RGB uint8 640x480 at 30 fps",
                "clock_synchronization": {
                    "clock": "Isaac simulation clock (integer nanoseconds)",
                    "reference_grid": "t[n] = n / 30 seconds",
                    "association": (
                        "Synchronous simulation snapshot; body, hands, cameras, and the issued "
                        "processed joint target share the frame reference timestamp"
                    ),
                    "tolerance_ms": 15,
                    "resampling": "none",
                    "raw_timestamp_fields": "source.timestamp.*_ns in source HDF5 and exported Parquet",
                },
                "controllers": {
                    "all_joints": "Isaac Lab JointPositionToLimitsActionCfg position targets",
                    "legs_and_waist": "standing hold targets",
                    "arms_and_hands": "scripted bounded-DLS policy" if is_sugar else "standing hold targets",
                    "stored_action": "processed absolute target after action-limit mapping",
                    "configuration_reference": [
                        "src/vla_isaaclab/envs/common/g1.py",
                        "src/vla_isaaclab/envs/common/managers.py",
                        "src/vla_isaaclab/policies/ycb_sugar_box.py" if is_sugar else "src/vla_isaaclab/policies/standing.py",
                    ],
                },
                "held_and_moving_groups": (
                    {"held": ["legs", "waist"], "moving": ["arms", "hands"]}
                    if is_sugar else {"held": ["legs", "waist", "arms", "hands"], "moving": []}
                ),
                "task_success_criteria": (
                    "Named success termination: <=15 mm XY/height error, low object speed, "
                    "palm separation >20 cm, held for 15 steps"
                    if is_sugar else "No task success termination; preview episode is labeled success=false"
                ),
                "contract_exceptions": [
                    "observation.images.cam_right_wrist intentionally omitted per user direction on 2026-09-22"
                ],
            }
            if is_orchard:
                collection.update({
                    "orchard_layout": cfg.orchard_layout.metadata(),
                    "camera_profile": cfg.camera_profile,
                    "camera_semantics": "cam_left_high is fixed world front/top, not the legacy head-mounted view",
                    "tree_asset": getattr(cfg, "tree_asset_metadata", None),
                    "contract_exceptions": [],
                    "task_success_criteria": (
                        "Native stem break after pull; sustained lift; physical basket floor contact; "
                        "open hand >18 cm away; whole fruit inside basket, speed <0.035 m/s for 30 steps; "
                        "no non-target detachment or task failure"
                    ),
                    "held_and_moving_groups": {
                        "held": ["legs", "waist roll/pitch", "right arm", "right hand"],
                        "moving": ["waist yaw", "left arm", "left hand"],
                    },
                })
                collection["controllers"]["arms_and_hands"] = "scripted orchard phases with existing bounded-DLS IK"
                collection["controllers"]["legs_and_waist"] = "legs and waist roll/pitch held; waist yaw participates in IK"
                collection["controllers"]["configuration_reference"] = [
                    "src/vla_isaaclab/policies/orchard_pick.py",
                    "src/vla_isaaclab/policies/ycb_sugar_box.py",
                    "src/vla_isaaclab/envs/orchard_pick/env_cfg.py",
                ]
            writer = StagingHDF5Writer(
                root=PROJECT_ROOT / "outputs/lerobot_staging",
                dataset_name=Path(ARGS.dataset_name).stem,
                features=adapter.features,
                metadata={
                    "collection": collection,
                    "environment_id": ARGS.task,
                    "rates_hz": {"physics": 120, "control": 30, "camera": 30},
                    "joint_names": adapter.joint_names,
                    "simulator_joint_names": adapter.simulator_joint_names,
                    "camera_keys": adapter.camera_feature_keys,
                    "environment_state_names": adapter.environment_state_names,
                    "task_prompt": task_prompt,
                    "units": {"joint_position": "rad", "joint_velocity": "rad/s", "position": "m"},
                    "isaac_sim_version": "4.5.0.0",
                    "isaac_lab_version": "v2.0.2",
                    "lerobot_version": ARGS.lerobot_version,
                },
            )
            try:
                with torch.inference_mode():
                    for episode_index in range(ARGS.episodes):
                        if episode_index:
                            env.reset(seed=ARGS.seed + episode_index)
                            policy = make_policy(env)
                        episode_succeeded = False
                        for episode_step in range(finite_steps):
                            reference_time_ns = round(episode_step * 1_000_000_000 / 30)
                            snapshot = adapter.capture(reference_time_ns)
                            action = policy.compute(episode_step)
                            _, reward, terminated, timed_out, _ = env.step(action)
                            if camera_videos is not None:
                                camera_videos.write()
                            step_succeeded = task_succeeded(env) and not getattr(policy, "failed", False)
                            writer.add_frame(
                                adapter.complete_frame(
                                    snapshot,
                                    reward,
                                    terminated,
                                    timed_out,
                                    episode_step + 1 == finite_steps,
                                    task_prompt,
                                    ARGS.seed + episode_index,
                                    step_succeeded,
                                )
                            )
                            success_count += int(step_succeeded)
                            episode_succeeded = episode_succeeded or step_succeeded
                            steps += 1
                            if getattr(policy, "failed", False) or bool(terminated[0]) or bool(timed_out[0]):
                                break
                        writer.save_episode(success=episode_succeeded)
                staging_path = writer.finalize()
                conversion_command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/convert_staging_to_lerobot.py"),
                    str(staging_path),
                    "--lerobot-version",
                    ARGS.lerobot_version,
                ]
                if ARGS.include_failed_episodes:
                    conversion_command.append("--include-failed-episodes")
                conversion = subprocess.run(conversion_command, check=False)
                if conversion.returncode:
                    raise RuntimeError(f"LeRobot conversion failed with exit code {conversion.returncode}")
                dataset_path = PROJECT_ROOT / "outputs/lerobot" / Path(ARGS.dataset_name).stem
            except BaseException:
                writer.abort()
                raise
        else:
            with torch.inference_mode():
                while APP.is_running() and (ARGS.steps == 0 or steps < ARGS.steps):
                    action = policy.compute(steps)
                    _, _, terminated, timed_out, _ = env.step(action)
                    if camera_videos is not None:
                        camera_videos.write()
                    if video_stream is not None:
                        import av

                        rgb = primary_camera(env).data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                        frame = av.VideoFrame.from_ndarray(rgb.astype(np.uint8), format="rgb24")
                        for packet in video_stream.encode(frame):
                            video_container.mux(packet)
                    success_count += int(task_succeeded(env))
                    steps += 1
                    if getattr(policy, "failed", False):
                        break
                    if bool(terminated[0].item()) or bool(timed_out[0].item()):
                        break
                    if ARGS.steps == 0 and ARGS.headless and steps >= finite_steps:
                        break

        report = validate(env, policy, steps, success_count, dataset_path)
        output_dir = ARGS.report_dir or PROJECT_ROOT / "outputs/environments" / ARGS.task
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        if hasattr(env, "orchard_trace"):
            (output_dir / "trajectory.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in env.orchard_trace)
            )
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["passed"] else 2
    finally:
        if camera_videos is not None:
            camera_videos.close()
        if video_stream is not None:
            for packet in video_stream.encode():
                video_container.mux(packet)
        if video_container is not None:
            video_container.close()
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        write_exit_status(exit_code)
        try:
            APP.close()
        except SystemExit:
            pass
    raise SystemExit(exit_code)
