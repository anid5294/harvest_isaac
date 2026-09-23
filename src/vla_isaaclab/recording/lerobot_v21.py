"""LeRobot v2.1 exporter for the G1 + Dex3 data contract."""

from __future__ import annotations

import json
import os
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FPS = 30
CHUNKS_SIZE = 1000
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_jsonable(row), ensure_ascii=False) + "\n" for row in rows)
    )


def _feature_stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    keepdims = values.ndim == 1
    return {
        "min": np.min(values, axis=0, keepdims=keepdims),
        "max": np.max(values, axis=0, keepdims=keepdims),
        "mean": np.mean(values, axis=0, keepdims=keepdims),
        "std": np.std(values, axis=0, keepdims=keepdims),
        "count": np.asarray([len(values)], dtype=np.int64),
    }


def _image_stats(dataset: h5py.Dataset) -> dict:
    sample_count = min(len(dataset), 100)
    indices = np.unique(np.rint(np.linspace(0, len(dataset) - 1, sample_count)).astype(int))
    samples = []
    for index in indices:
        image = np.asarray(dataset[index], dtype=np.float32)
        # Match the pinned v2.1 implementation's inexpensive image-stat sampling.
        factor = max(image.shape[0] // 150, image.shape[1] // 150, 1)
        samples.append(np.transpose(image[::factor, ::factor, :3], (2, 0, 1)) / 255.0)
    images = np.stack(samples)
    return {
        "min": np.min(images, axis=(0, 2, 3), keepdims=True).squeeze(0),
        "max": np.max(images, axis=(0, 2, 3), keepdims=True).squeeze(0),
        "mean": np.mean(images, axis=(0, 2, 3), keepdims=True).squeeze(0),
        "std": np.std(images, axis=(0, 2, 3), keepdims=True).squeeze(0),
        "count": np.asarray([len(images)], dtype=np.int64),
    }


def _arrow_array(values: np.ndarray, feature: dict) -> pa.Array:
    dtype = feature["dtype"]
    shape = tuple(feature["shape"])
    values = np.asarray(values)
    arrow_type = {
        "float32": pa.float32(),
        "float64": pa.float64(),
        "int64": pa.int64(),
        "bool": pa.bool_(),
    }[dtype]
    if shape == (1,):
        return pa.array(values.reshape(-1), type=arrow_type)
    if len(shape) != 1:
        raise ValueError(f"Only scalar and 1-D numeric features are supported, found {feature}")
    if values.shape != (len(values), shape[0]):
        raise ValueError(f"Feature has shape {values.shape}, expected (N, {shape[0]})")
    flat = pa.array(values.reshape(-1), type=arrow_type)
    return pa.FixedSizeListArray.from_arrays(flat, shape[0])


def _encode_video(dataset: h5py.Dataset, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = dataset.shape[1:]
    if (height, width, channels) != (480, 640, 3):
        raise ValueError(f"Expected RGB uint8[480,640,3], found {dataset.shape[1:]}")
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libsvtav1", rate=FPS, options={"preset": "10", "crf": "35"})
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for image in dataset:
            image = np.asarray(image)
            if image.dtype != np.uint8:
                raise ValueError(f"Expected uint8 camera data, found {image.dtype}")
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        return {
            "video.height": stream.height,
            "video.width": stream.width,
            "video.codec": "av1" if stream.codec_context.codec_tag == "av01" else stream.codec_context.name,
            "video.pix_fmt": stream.pix_fmt,
            "video.is_depth_map": False,
            "video.fps": int(stream.average_rate),
            "video.channels": 3,
            "has_audio": False,
        }


def export_staging_to_v21(
    staging_path: Path,
    root: Path,
    dataset_name: str,
    include_failed_episodes: bool = False,
) -> Path:
    """Convert one complete staging file into an atomic LeRobot v2.1 dataset."""
    final_root = root / dataset_name
    work_root = root / f".{dataset_name}.incomplete-{os.getpid()}"
    if final_root.exists() or work_root.exists():
        raise FileExistsError(f"Dataset path already exists: {final_root}")
    work_root.mkdir(parents=True)

    try:
        with h5py.File(staging_path, "r") as stream:
            if not bool(stream.attrs.get("complete", False)):
                raise RuntimeError(f"Staging recording is incomplete: {staging_path}")
            features = json.loads(stream.attrs["features"])
            metadata = json.loads(stream.attrs["metadata"])
            camera_keys = [key for key, feature in features.items() if feature["dtype"] == "video"]
            numeric_features = {
                key: {**feature, "shape": list(feature["shape"])}
                for key, feature in features.items()
                if feature["dtype"] != "video"
            }
            output_features = {**numeric_features, **DEFAULT_FEATURES}
            for key in camera_keys:
                output_features[key] = {
                    "dtype": "video",
                    "shape": [3, 480, 640],
                    "names": ["channels", "height", "width"],
                }

            episode_rows = []
            episode_stats_rows = []
            task_rows = []
            task_indices = {}
            global_index = 0
            video_info = {}

            skipped_failed_episodes = 0
            for episode_name in sorted(stream["episodes"]):
                episode = stream[f"episodes/{episode_name}"]
                length = int(episode.attrs["length"])
                task = str(episode.attrs["task"])
                success = bool(episode.attrs.get("success", False))
                if not success and not include_failed_episodes:
                    skipped_failed_episodes += 1
                    continue
                episode_index = len(episode_rows)
                if task not in task_indices:
                    task_indices[task] = len(task_indices)
                    task_rows.append({"task_index": task_indices[task], "task": task})
                task_index = task_indices[task]

                columns = {}
                stats = {}
                for key, feature in numeric_features.items():
                    values = np.asarray(episode[key])
                    if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                        raise ValueError(f"Non-finite values in episode {episode_index} feature {key}")
                    columns[key] = _arrow_array(values, feature)
                    stats[key] = _feature_stats(values.reshape(length, -1) if values.ndim > 1 else values)

                timestamps = np.arange(length, dtype=np.float32) / np.float32(FPS)
                default_values = {
                    "timestamp": timestamps,
                    "frame_index": np.arange(length, dtype=np.int64),
                    "episode_index": np.full(length, episode_index, dtype=np.int64),
                    "index": np.arange(global_index, global_index + length, dtype=np.int64),
                    "task_index": np.full(length, task_index, dtype=np.int64),
                }
                for key, values in default_values.items():
                    columns[key] = _arrow_array(values, DEFAULT_FEATURES[key])
                    stats[key] = _feature_stats(values)

                chunk = episode_index // CHUNKS_SIZE
                parquet_path = work_root / (
                    f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
                )
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.table(columns), parquet_path, compression="zstd")

                for key in camera_keys:
                    video_path = work_root / (
                        f"videos/chunk-{chunk:03d}/{key}/episode_{episode_index:06d}.mp4"
                    )
                    current_info = _encode_video(episode[key], video_path)
                    if key in video_info and current_info != video_info[key]:
                        raise ValueError(f"Inconsistent video encoding for {key}")
                    video_info[key] = current_info
                    stats[key] = _image_stats(episode[key])

                episode_rows.append(
                    {
                        "episode_index": episode_index,
                        "tasks": [task],
                        "length": length,
                        "success": success,
                    }
                )
                episode_stats_rows.append({"episode_index": episode_index, "stats": stats})
                global_index += length

            if not episode_rows:
                raise RuntimeError(
                    "No successful episodes are available for the default imitation export. "
                    "The source HDF5 is retained; use --include-failed-episodes only for audit datasets."
                )

            for key in camera_keys:
                output_features[key]["info"] = video_info[key]
            info = {
                "codebase_version": "v2.1",
                "robot_type": "Unitree_G1",
                "total_episodes": len(episode_rows),
                "total_frames": global_index,
                "total_tasks": len(task_rows),
                "total_videos": len(episode_rows) * len(camera_keys),
                "total_chunks": (len(episode_rows) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
                "chunks_size": CHUNKS_SIZE,
                "fps": FPS,
                "splits": {"train": f"0:{len(episode_rows)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": output_features,
            }
            collection = dict(metadata["collection"])
            collection["environment_id"] = metadata["environment_id"]
            collection["task_prompt"] = metadata["task_prompt"]
            collection["rates_hz"] = metadata["rates_hz"]
            collection["joint_names"] = metadata["joint_names"]
            collection["simulator_joint_names"] = metadata["simulator_joint_names"]
            collection["source_recording"] = str(staging_path.resolve())
            collection["episodes"] = len(episode_rows)
            collection["frames"] = global_index
            collection["camera_keys"] = camera_keys
            collection["contains_failed_episodes"] = any(
                not row["success"] for row in episode_rows
            )
            collection["default_imitation_training_eligible"] = all(
                row["success"] for row in episode_rows
            )
            collection["export_purpose"] = (
                "successful-only imitation training export"
                if collection["default_imitation_training_eligible"]
                else "contract-validation/source-review export; exclude failed episodes from imitation training"
            )
            collection["failed_source_episodes_excluded"] = skipped_failed_episodes

            _write_json(work_root / "meta/info.json", info)
            _write_jsonl(work_root / "meta/tasks.jsonl", task_rows)
            _write_jsonl(work_root / "meta/episodes.jsonl", episode_rows)
            _write_jsonl(work_root / "meta/episodes_stats.jsonl", episode_stats_rows)
            _write_json(work_root / "meta/collection.json", collection)
        work_root.rename(final_root)
    except BaseException:
        # Keep the incomplete directory for diagnosis; never publish a partial dataset.
        raise
    return final_root
