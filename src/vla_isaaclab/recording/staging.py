"""Neutral HDF5 staging format between Isaac Sim and LeRobot."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np


class StagingHDF5Writer:
    """Buffer one episode at a time and publish a complete staging file atomically."""

    def __init__(self, root: Path, dataset_name: str, features: dict, metadata: dict):
        root.mkdir(parents=True, exist_ok=True)
        self.final_path = root / f"{dataset_name}.hdf5"
        self.work_path = root / f".{dataset_name}.incomplete-{os.getpid()}.hdf5"
        if self.final_path.exists():
            raise FileExistsError(f"Staging dataset already exists: {self.final_path}")
        self.file = h5py.File(self.work_path, "w")
        self.file.attrs["format"] = "sim-platform-staging-v1"
        self.file.attrs["complete"] = False
        self.file.attrs["features"] = json.dumps(features)
        self.file.attrs["metadata"] = json.dumps(metadata)
        self.frames = []
        self.episode_count = 0
        self.frame_count = 0
        self.closed = False

    def add_frame(self, frame: dict) -> None:
        task = frame.pop("task")
        self.frames.append(({key: np.asarray(value) for key, value in frame.items()}, task))

    def save_episode(self, success: bool) -> None:
        if not self.frames:
            raise RuntimeError("Cannot save an empty episode")
        group = self.file.create_group(f"episodes/episode_{self.episode_count:06d}")
        group.attrs["task"] = self.frames[0][1]
        group.attrs["success"] = bool(success)
        for key in self.frames[0][0]:
            values = np.stack([frame[key] for frame, _ in self.frames])
            options = {"compression": "gzip", "compression_opts": 1, "shuffle": True} if values.ndim > 1 else {}
            group.create_dataset(key, data=values, **options)
        group.attrs["length"] = len(self.frames)
        self.frame_count += len(self.frames)
        self.episode_count += 1
        self.frames.clear()
        self.file.flush()

    def finalize(self) -> Path:
        if self.closed:
            return self.final_path
        if self.frames:
            raise RuntimeError("An unsaved episode remains in the staging writer")
        self.file.attrs["episodes"] = self.episode_count
        self.file.attrs["frames"] = self.frame_count
        self.file.attrs.modify("complete", True)
        self.file.close()
        self.work_path.rename(self.final_path)
        self.closed = True
        return self.final_path

    def abort(self) -> None:
        if not self.closed:
            self.file.close()
            self.closed = True
