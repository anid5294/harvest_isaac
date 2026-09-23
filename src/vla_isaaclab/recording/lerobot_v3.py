"""Atomic LeRobot Dataset v3 writer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotV3Writer:
    """Write episodes with the official LeRobot API and publish on finalize."""

    def __init__(
        self,
        root: Path,
        dataset_name: str,
        features: dict,
        metadata: dict,
        fps: int = 30,
        image_writer_threads: int = 4,
    ):
        self.final_root = root / dataset_name
        self.work_root = root / f".{dataset_name}.incomplete-{os.getpid()}"
        if self.final_root.exists():
            raise FileExistsError(f"Dataset already exists: {self.final_root}")
        if self.work_root.exists():
            raise FileExistsError(f"Temporary dataset already exists: {self.work_root}")
        root.mkdir(parents=True, exist_ok=True)

        self.repo_id = f"local/{dataset_name}"
        self.metadata = metadata
        self.episode_count = 0
        self.frame_count = 0
        self.episode_success = []
        self._pending_frames = 0
        self._closed = False
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=fps,
            features=features,
            root=self.work_root,
            robot_type="unitree_g1",
            use_videos=True,
            image_writer_threads=image_writer_threads,
            video_backend="pyav",
            vcodec="libsvtav1",
        )
        self._write_manifest(complete=False)

    def _write_manifest(self, complete: bool) -> None:
        payload = {
            "format": "LeRobotDataset-v3.0",
            "lerobot_repo_id": self.repo_id,
            "recording_complete": complete,
            "episodes": self.episode_count,
            "frames": self.frame_count,
            **self.metadata,
        }
        path = self.work_root / "meta" / "simulation.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def add_frame(self, frame: dict) -> None:
        if self._closed:
            raise RuntimeError("Cannot add a frame after finalization")
        self.dataset.add_frame(frame)
        self._pending_frames += 1

    def save_episode(self, success: bool) -> None:
        if self._closed:
            raise RuntimeError("Cannot save an episode after finalization")
        self.dataset.save_episode(parallel_encoding=False)
        self.episode_count += 1
        self.frame_count += self._pending_frames
        self.episode_success.append(bool(success))
        self._pending_frames = 0
        self._write_manifest(complete=False)

    def discard_episode(self) -> None:
        if not self._closed:
            self.dataset.clear_episode_buffer(delete_images=True)
            self._pending_frames = 0

    def finalize(self) -> Path:
        if self._closed:
            return self.final_root
        self.dataset.stop_image_writer()
        self.dataset.finalize()
        self._write_manifest(complete=True)
        self.work_root.rename(self.final_root)
        self._closed = True
        return self.final_root

    def abort(self) -> None:
        """Close writers while leaving the incomplete directory for diagnosis."""
        if self._closed:
            return
        try:
            self.dataset.clear_episode_buffer(delete_images=True)
            self.dataset.stop_image_writer()
            self.dataset.finalize()
            self._write_manifest(complete=False)
        finally:
            self._closed = True


def export_staging_to_v3(
    staging_path: Path,
    root: Path,
    dataset_name: str,
    include_failed_episodes: bool = False,
) -> Path:
    """Convert one complete staging file using the official LeRobot v3 API."""
    with h5py.File(staging_path, "r") as stream:
        if not bool(stream.attrs.get("complete", False)):
            raise RuntimeError(f"Staging recording is incomplete: {staging_path}")
        features = json.loads(stream.attrs["features"])
        for feature in features.values():
            feature["shape"] = tuple(feature["shape"])
        metadata = json.loads(stream.attrs["metadata"])
        metadata["source_recording"] = str(staging_path.resolve())
        metadata["lerobot_version"] = "3"
        episode_names = sorted(stream["episodes"])
        eligible_episode_names = [
            name
            for name in episode_names
            if include_failed_episodes
            or bool(stream[f"episodes/{name}"].attrs.get("success", False))
        ]
        if not eligible_episode_names:
            raise RuntimeError(
                "No successful episodes are available for the default imitation export. "
                "The source HDF5 is retained; use --include-failed-episodes only for audit datasets."
            )
        skipped_failed = len(episode_names) - len(eligible_episode_names)

        writer = LeRobotV3Writer(
            root=root,
            dataset_name=dataset_name,
            features=features,
            metadata=metadata,
        )
        try:
            for episode_name in eligible_episode_names:
                episode = stream[f"episodes/{episode_name}"]
                success = bool(episode.attrs.get("success", False))
                task = str(episode.attrs["task"])
                for frame_index in range(int(episode.attrs["length"])):
                    frame = {key: episode[key][frame_index] for key in episode}
                    frame["task"] = task
                    writer.add_frame(frame)
                writer.save_episode(success=success)
            writer.metadata["contains_failed_episodes"] = not all(writer.episode_success)
            writer.metadata["default_imitation_training_eligible"] = all(writer.episode_success)
            writer.metadata["failed_source_episodes_excluded"] = skipped_failed
            writer.metadata["episode_success"] = writer.episode_success
            collection = writer.metadata.setdefault("collection", {})
            collection["source_recording"] = writer.metadata["source_recording"]
            collection["contains_failed_episodes"] = writer.metadata["contains_failed_episodes"]
            collection["default_imitation_training_eligible"] = writer.metadata[
                "default_imitation_training_eligible"
            ]
            collection["failed_source_episodes_excluded"] = skipped_failed
            return writer.finalize()
        except BaseException:
            writer.abort()
            raise
