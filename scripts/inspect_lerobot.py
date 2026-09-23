#!/usr/bin/env python3
"""Validate a G1 + Dex3 LeRobot v3 or v2.1 dataset against contract 1.0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


JOINT_NAMES = [
    "kLeftHipPitch", "kLeftHipRoll", "kLeftHipYaw", "kLeftKnee", "kLeftAnklePitch",
    "kLeftAnkleRoll", "kRightHipPitch", "kRightHipRoll", "kRightHipYaw", "kRightKnee",
    "kRightAnklePitch", "kRightAnkleRoll", "kWaistYaw", "kWaistRoll", "kWaistPitch",
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw", "kRightShoulderPitch",
    "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow", "kRightWristRoll",
    "kRightWristPitch", "kRightWristYaw", "kLeftHandThumb0", "kLeftHandThumb1",
    "kLeftHandThumb2", "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0",
    "kLeftHandIndex1", "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
]
REQUIRED_CAMERAS = {
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
}
REQUIRED_COLUMNS = {
    "observation.state", "action", "timestamp", "frame_index", "episode_index", "index",
    "task_index", "source.timestamp.reference_ns", "source.timestamp.body_feedback_ns",
    "source.timestamp.hand_feedback_ns", "source.timestamp.command_ns",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_v3(root: Path, allow_missing_right_wrist: bool) -> int:
    manifest = json.loads((root / "meta/simulation.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    features = info.get("features", {})
    errors = []
    warnings = []
    for key in ("observation.state", "action", "sim.action.normalized"):
        feature = features.get(key, {})
        if feature.get("dtype") != "float32" or feature.get("shape") != [43]:
            errors.append(f"{key} metadata is not float32[43]")
        if feature.get("names") != JOINT_NAMES:
            errors.append(f"{key} joint names/order do not match the contract")

    camera_keys = sorted(key for key, value in features.items() if value.get("dtype") == "video")
    missing_cameras = sorted(REQUIRED_CAMERAS - set(camera_keys))
    approved_missing = (
        missing_cameras == ["observation.images.cam_right_wrist"] and allow_missing_right_wrist
    )
    if missing_cameras and not approved_missing:
        errors.append(f"missing required cameras: {missing_cameras}")
    elif approved_missing:
        warnings.append("cam_right_wrist omitted under the explicit temporary user exception")

    collection = manifest.get("collection", {})
    for key, expected in {
        "contract_version": "1.0",
        "dataset_profile": "g1_29body_dex3_43d_v1",
        "robot_type": "Unitree_G1",
        "fps": 30,
        "joint_units": "rad",
        "joint_coordinate_mode": "PR",
        "real_or_sim": "sim",
    }.items():
        if collection.get(key) != expected:
            errors.append(f"collection.{key}: expected {expected!r}, found {collection.get(key)!r}")

    parquet_files = sorted(root.glob("data/**/*.parquet"))
    if not parquet_files:
        errors.append("dataset contains no Parquet files")
    total_frames = 0
    for parquet_path in parquet_files:
        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "action", "sim.action.normalized"],
        )
        total_frames += table.num_rows
        for key in ("observation.state", "action", "sim.action.normalized"):
            field_type = table.schema.field(key).type
            if not (
                pa.types.is_fixed_size_list(field_type)
                and field_type.list_size == 43
                and pa.types.is_float32(field_type.value_type)
            ):
                errors.append(f"{parquet_path.name} {key} is not fixed-size-list<float32>[43]")
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if values.shape != (table.num_rows, 43) or not np.isfinite(values).all():
                errors.append(f"{parquet_path.name} {key} has invalid shape or non-finite values")

    if manifest.get("format") != "LeRobotDataset-v3.0":
        errors.append(f"unexpected manifest format: {manifest.get('format')!r}")
    if info.get("codebase_version") != "v3.0":
        errors.append(f"unexpected info codebase_version: {info.get('codebase_version')!r}")
    if manifest.get("recording_complete") is not True:
        errors.append("recording is not marked complete")
    if total_frames != manifest.get("frames"):
        errors.append("manifest frame count does not match Parquet rows")

    full_contract_valid = not errors and not missing_cameras
    valid_with_approved_exception = not errors and (not missing_cameras or approved_missing)
    report = {
        "valid_with_approved_exception": valid_with_approved_exception,
        "full_contract_valid": full_contract_valid,
        "approved_exception_active": approved_missing,
        "root": str(root),
        "format": info.get("codebase_version"),
        "episodes": manifest.get("episodes"),
        "frames": total_frames,
        "action_shape": [total_frames, 43] if not errors or total_frames else None,
        "state_shape": [total_frames, 43] if not errors or total_frames else None,
        "normalized_action_shape": [total_frames, 43] if not errors or total_frames else None,
        "joint_names_match_contract": features.get("action", {}).get("names") == JOINT_NAMES,
        "camera_keys": camera_keys,
        "missing_required_cameras": missing_cameras,
        "warnings": warnings,
        "errors": errors,
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if valid_with_approved_exception else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--lerobot-version", choices=("auto", "3", "2.1"), default="auto",
        help="Dataset format; auto detects it from metadata (default: auto).",
    )
    parser.add_argument(
        "--allow-missing-right-wrist", action="store_true",
        help="Accept the explicitly user-approved temporary right-wrist-camera exception.",
    )
    args = parser.parse_args()
    root = args.dataset.resolve()
    detected_version = "3" if (root / "meta/simulation.json").is_file() else "2.1"
    selected_version = detected_version if args.lerobot_version == "auto" else args.lerobot_version
    if selected_version != detected_version:
        raise ValueError(
            f"Requested LeRobot {selected_version}, but metadata identifies {detected_version}: {root}"
        )
    if selected_version == "3":
        return validate_v3(root, args.allow_missing_right_wrist)
    errors = []
    warnings = []

    info = json.loads((root / "meta/info.json").read_text())
    collection = json.loads((root / "meta/collection.json").read_text())
    tasks = read_jsonl(root / "meta/tasks.jsonl")
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    episode_stats = read_jsonl(root / "meta/episodes_stats.jsonl")
    features = info.get("features", {})
    camera_keys = sorted(key for key, value in features.items() if value.get("dtype") == "video")

    expected_info = {
        "codebase_version": "v2.1", "robot_type": "Unitree_G1", "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    for key, expected in expected_info.items():
        if info.get(key) != expected:
            errors.append(f"info.{key}: expected {expected!r}, found {info.get(key)!r}")
    for key in ("observation.state", "action"):
        feature = features.get(key, {})
        if feature.get("dtype") != "float32" or feature.get("shape") != [43]:
            errors.append(f"{key} metadata is not float32[43]")
        if feature.get("names") != JOINT_NAMES:
            errors.append(f"{key} joint names/order do not match the contract")

    missing_cameras = sorted(REQUIRED_CAMERAS - set(camera_keys))
    approved_missing = missing_cameras == ["observation.images.cam_right_wrist"] and args.allow_missing_right_wrist
    if missing_cameras and not approved_missing:
        errors.append(f"missing required cameras: {missing_cameras}")
    elif approved_missing:
        warnings.append("cam_right_wrist omitted under the explicit temporary user exception")
    for key in camera_keys:
        feature = features[key]
        if feature.get("shape") != [3, 480, 640] or feature.get("names") != ["channels", "height", "width"]:
            errors.append(f"{key} does not declare contract CHW video metadata")
        video_info = feature.get("info", {})
        if (
            video_info.get("video.codec") != "av1"
            or video_info.get("video.pix_fmt") != "yuv420p"
            or video_info.get("video.fps") != 30
        ):
            errors.append(f"{key} info.json encoding metadata is not AV1/yuv420p at 30 fps")

    if info.get("total_videos") != len(camera_keys) * info.get("total_episodes", 0):
        errors.append("total_videos does not match camera/episode count")
    if info.get("total_chunks") != (info.get("total_episodes", 0) + 999) // 1000:
        errors.append("total_chunks is inconsistent")
    if info.get("splits") != {"train": f"0:{info.get('total_episodes', 0)}"}:
        errors.append("training split is inconsistent")

    for key, expected in {
        "contract_version": "1.0", "dataset_profile": "g1_29body_dex3_43d_v1",
        "robot_type": "Unitree_G1", "fps": 30, "joint_units": "rad",
        "joint_coordinate_mode": "PR", "real_or_sim": "sim",
    }.items():
        if collection.get(key) != expected:
            errors.append(f"collection.{key}: expected {expected!r}, found {collection.get(key)!r}")
    for key in (
        "robot_identifier", "hand_identifiers", "software", "collection_date", "collector",
        "calibration_identifier", "cameras", "clock_synchronization", "controllers",
        "task_success_criteria", "source_recording",
    ):
        if not collection.get(key):
            errors.append(f"collection metadata missing {key}")
    source_recording = Path(collection.get("source_recording", ""))
    if not source_recording.is_file():
        errors.append("retained source recording is missing")
    camera_metadata = {row.get("feature_key"): row for row in collection.get("cameras", [])}
    if set(camera_metadata) != set(camera_keys):
        errors.append("collection camera set does not match info.json")
    for key, row in camera_metadata.items():
        calibration = row.get("calibration", {})
        if not all(
            name in calibration
            for name in ("model", "focal_length_mm", "horizontal_aperture_mm", "clipping_range_m")
        ):
            errors.append(f"{key} collection calibration is incomplete")

    if len(tasks) != info.get("total_tasks"):
        errors.append("task count does not match info.json")
    if len(episodes) != info.get("total_episodes") or len(episode_stats) != len(episodes):
        errors.append("episode metadata/stat count does not match info.json")
    contains_failed = any(row.get("success") is False for row in episodes)
    if collection.get("contains_failed_episodes") is not contains_failed:
        errors.append("collection failed-episode annotation is inconsistent")
    if contains_failed and collection.get("default_imitation_training_eligible") is not False:
        errors.append("an export containing failed episodes must be marked training-ineligible")
    task_by_index = {row["task_index"]: row["task"] for row in tasks}
    if sorted(task_by_index) != list(range(len(task_by_index))):
        errors.append("task indices are not contiguous")

    expected_global_index = 0
    total_frames = 0
    video_reports = []
    max_sync_error_ms = 0.0
    for expected_episode_index, episode in enumerate(episodes):
        episode_index = episode.get("episode_index")
        length = episode.get("length")
        if episode_index != expected_episode_index:
            errors.append("episode indices are not contiguous")
            continue
        if not isinstance(episode.get("success"), bool):
            errors.append(f"episode {episode_index} lacks boolean success")
        if len(episode.get("tasks", [])) != 1 or episode["tasks"][0] not in task_by_index.values():
            errors.append(f"episode {episode_index} task does not resolve")
        chunk = episode_index // 1000
        parquet_path = root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        if not parquet_path.is_file():
            errors.append(f"missing {parquet_path.relative_to(root)}")
            continue
        table = pq.read_table(parquet_path)
        if table.num_rows != length:
            errors.append(f"episode {episode_index} parquet length mismatch")
        missing_columns = REQUIRED_COLUMNS - set(table.column_names)
        missing_columns |= {
            f"source.timestamp.{row['sensor_name']}_acquisition_ns"
            for row in camera_metadata.values()
        } - set(table.column_names)
        if missing_columns:
            errors.append(f"episode {episode_index} missing columns {sorted(missing_columns)}")
            continue
        for key in ("observation.state", "action"):
            field_type = table.schema.field(key).type
            if not (
                pa.types.is_fixed_size_list(field_type) and field_type.list_size == 43
                and pa.types.is_float32(field_type.value_type)
            ):
                errors.append(f"episode {episode_index} {key} is not fixed-size-list<float32>[43]")
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if values.shape != (length, 43) or not np.isfinite(values).all():
                errors.append(f"episode {episode_index} {key} has invalid shape or non-finite values")
        for field in table.schema:
            if field.name in camera_keys:
                errors.append(f"video feature {field.name} must not be embedded in Parquet")
                continue
            if pa.types.is_floating(field.type):
                values = table[field.name].to_numpy(zero_copy_only=False)
                if not np.isfinite(values).all():
                    errors.append(f"episode {episode_index} {field.name} contains non-finite values")
        timestamp = table["timestamp"].to_numpy(zero_copy_only=False)
        frame_index = table["frame_index"].to_numpy(zero_copy_only=False)
        global_index = table["index"].to_numpy(zero_copy_only=False)
        task_index = table["task_index"].to_numpy(zero_copy_only=False)
        episode_indices = table["episode_index"].to_numpy(zero_copy_only=False)
        expected_timestamps = np.arange(length, dtype=np.float32) / np.float32(30)
        if timestamp.dtype != np.float32 or not np.array_equal(timestamp, expected_timestamps):
            errors.append(f"episode {episode_index} timestamp is not exact float32 n/30")
        if not np.array_equal(frame_index, np.arange(length, dtype=np.int64)):
            errors.append(f"episode {episode_index} frame_index is not contiguous")
        if not np.array_equal(global_index, np.arange(expected_global_index, expected_global_index + length)):
            errors.append(f"episode {episode_index} global index is not contiguous")
        if len(set(task_index.tolist())) != 1 or int(task_index[0]) not in task_by_index:
            errors.append(f"episode {episode_index} task_index is not constant/resolvable")
        if not np.array_equal(episode_indices, np.full(length, episode_index, dtype=np.int64)):
            errors.append(f"episode {episode_index} episode_index column is inconsistent")

        reference = table["source.timestamp.reference_ns"].to_numpy(zero_copy_only=False)
        expected_reference = np.rint(np.arange(length) * 1_000_000_000 / 30).astype(np.int64)
        if not np.array_equal(reference, expected_reference):
            errors.append(f"episode {episode_index} source reference timestamps do not match 30 Hz grid")
        timestamp_columns = [
            name for name in table.column_names
            if name.startswith("source.timestamp.") and name.endswith("_ns")
        ]
        for name in timestamp_columns:
            source_time = table[name].to_numpy(zero_copy_only=False)
            error_ms = float(np.max(np.abs(source_time - reference)) / 1_000_000)
            max_sync_error_ms = max(max_sync_error_ms, error_ms)
            if error_ms > 15.0:
                errors.append(f"episode {episode_index} {name} exceeds 15 ms ({error_ms:.3f} ms)")

        for camera_key in camera_keys:
            video_path = root / f"videos/chunk-{chunk:03d}/{camera_key}/episode_{episode_index:06d}.mp4"
            if not video_path.is_file():
                errors.append(f"missing video {video_path.relative_to(root)}")
                continue
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                decoded = 0
                shape = None
                for frame in container.decode(stream):
                    rgb = frame.to_ndarray(format="rgb24")
                    decoded += 1
                    shape = list(rgb.shape)
                video_report = {
                    "episode_index": episode_index, "key": camera_key, "frames": decoded,
                    "shape_hwc": shape, "codec": "av1" if stream.codec_context.codec_tag == "av01" else stream.codec_context.name,
                    "pixel_format": stream.pix_fmt, "fps": float(stream.average_rate),
                }
                video_reports.append(video_report)
                if decoded != length or shape != [480, 640, 3]:
                    errors.append(f"{camera_key} episode {episode_index} frame count/shape mismatch")
                if (
                    video_report["codec"] != "av1" or video_report["pixel_format"] != "yuv420p"
                    or video_report["fps"] != 30.0
                ):
                    errors.append(f"{camera_key} episode {episode_index} encoding mismatch: {video_report}")
        total_frames += length
        expected_global_index += length

    if total_frames != info.get("total_frames"):
        errors.append("total frame count does not match info.json")
    full_contract_valid = not errors and not missing_cameras
    valid_with_approved_exception = not errors and (not missing_cameras or approved_missing)
    report = {
        "valid_with_approved_exception": valid_with_approved_exception,
        "full_contract_valid": full_contract_valid,
        "approved_exception_active": approved_missing,
        "root": str(root),
        "format": info.get("codebase_version"),
        "episodes": len(episodes),
        "frames": total_frames,
        "success": [row.get("success") for row in episodes],
        "camera_keys": camera_keys,
        "missing_required_cameras": missing_cameras,
        "max_sync_error_ms": max_sync_error_ms,
        "video_reports": video_reports,
        "warnings": warnings,
        "errors": errors,
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if valid_with_approved_exception else 2


if __name__ == "__main__":
    raise SystemExit(main())
