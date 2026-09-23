"""Scenario-independent frame capture for dataset writers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vla_isaaclab.envs.common import (
    ACTION_JOINT_NAMES,
    CONTRACT_JOINT_NAMES,
    LEFT_END_EFFECTOR,
    RIGHT_END_EFFECTOR,
)
from vla_isaaclab.envs.common.managers import ACTION_TERM_NAME


POSE_NAMES = ("position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z")


@dataclass(frozen=True)
class FrameSnapshot:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    environment_state: np.ndarray
    images: dict[str, np.ndarray]
    control_phase: np.ndarray
    source_timestamps_ns: dict[str, np.ndarray]


class EnvironmentFrameAdapter:
    """Capture one environment using stable, robot-facing data semantics."""

    def __init__(self, env):
        if env.num_envs != 1:
            raise ValueError("LeRobot recording currently supports exactly one environment")
        self.env = env
        self.robot = env.scene["robot"]
        self.action_term = env.action_manager.get_term(ACTION_TERM_NAME)
        self.simulator_joint_names = list(ACTION_JOINT_NAMES)
        self.joint_names = list(CONTRACT_JOINT_NAMES)
        self.joint_ids, found = self.robot.find_joints(
            self.simulator_joint_names, preserve_order=True
        )
        if list(found) != self.simulator_joint_names:
            raise RuntimeError(
                f"Contract joint mismatch: expected {self.simulator_joint_names}, found {found}"
            )
        action_term_names = list(self.action_term._joint_names)
        if len(action_term_names) != 43 or set(action_term_names) != set(self.simulator_joint_names):
            raise RuntimeError(
                "The action term must control exactly the 43 G1 + Dex3 contract joints; "
                f"found {action_term_names}"
            )
        self.action_contract_indices = [
            action_term_names.index(name) for name in self.simulator_joint_names
        ]

        self.body_entries = []
        for role, body_name in (
            ("left_end_effector", LEFT_END_EFFECTOR),
            ("right_end_effector", RIGHT_END_EFFECTOR),
        ):
            body_ids, _ = self.robot.find_bodies([body_name], preserve_order=True)
            self.body_entries.append((role, body_ids[0]))

        self.rigid_object_names = sorted(env.scene.rigid_objects)
        self.environment_state_names = []
        for role, _ in self.body_entries:
            self.environment_state_names.extend(f"{role}.{axis}" for axis in POSE_NAMES)
        for name in self.rigid_object_names:
            self.environment_state_names.extend(f"{name}.{axis}" for axis in POSE_NAMES)
            self.environment_state_names.extend(
                f"{name}.{axis}"
                for axis in (
                    "linear_velocity.x", "linear_velocity.y", "linear_velocity.z",
                    "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
                )
            )

        camera_features = (
            ("cam_side", "observation.images.cam_side"),
            ("camera", "observation.images.cam_side"),
            ("cam_left_high", "observation.images.cam_left_high"),
            ("cam_left_wrist", "observation.images.cam_left_wrist"),
        )
        self.cameras = [
            (sensor_name, feature_name, env.scene.sensors[sensor_name])
            for sensor_name, feature_name in camera_features
            if sensor_name in env.scene.sensors
        ]
        # A scene may use either the legacy sensor name `camera` or `cam_side`,
        # but never export both under the same feature key.
        deduplicated = {}
        for entry in self.cameras:
            deduplicated.setdefault(entry[1], entry)
        self.cameras = list(deduplicated.values())
        if not self.cameras and "camera" in env.scene.sensors:
            self.cameras = [("camera", "observation.images.front", env.scene.sensors["camera"])]
        if not self.cameras:
            raise RuntimeError("LeRobot recording requires at least one RGB camera")
        self.image_shapes = {
            feature_name: tuple(int(value) for value in camera.data.output["rgb"][0, ..., :3].shape)
            for _, feature_name, camera in self.cameras
        }

    @property
    def camera_metadata(self) -> list[dict]:
        parent_links = {
            "cam_left_high": "head_link",
            "cam_left_wrist": "left_hand_palm_link",
            "cam_side": "world",
            "camera": "world",
        }
        result = []
        for sensor_name, feature_name, camera in self.cameras:
            offset = camera.cfg.offset
            result.append(
                {
                    "sensor_name": sensor_name,
                    "feature_key": feature_name,
                    "identifier": f"isaac-sim:{sensor_name}",
                    "parent_link": parent_links[sensor_name],
                    "position_xyz_m": list(offset.pos),
                    "orientation_wxyz": list(offset.rot),
                    "orientation_convention": offset.convention,
                    "resolution_hw": [camera.cfg.height, camera.cfg.width],
                    "color_order": "RGB",
                    "image_orientation": "native; no flip or rotation",
                    "encoding": "AV1/yuv420p",
                    "calibration": {
                        "model": "Isaac Sim pinhole",
                        "focal_length_mm": camera.cfg.spawn.focal_length,
                        "horizontal_aperture_mm": camera.cfg.spawn.horizontal_aperture,
                        "focus_distance_m": camera.cfg.spawn.focus_distance,
                        "clipping_range_m": list(camera.cfg.spawn.clipping_range),
                    },
                }
            )
        return result

    @property
    def camera_feature_keys(self) -> list[str]:
        return [feature_name for _, feature_name, _ in self.cameras]

    @property
    def features(self) -> dict:
        state_shape = (len(self.joint_names),)
        features = {
            "observation.state": {"dtype": "float32", "shape": state_shape, "names": self.joint_names},
            "observation.velocity": {"dtype": "float32", "shape": state_shape, "names": self.joint_names},
            "action": {"dtype": "float32", "shape": state_shape, "names": self.joint_names},
            "sim.action.normalized": {"dtype": "float32", "shape": state_shape, "names": self.joint_names},
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (len(self.environment_state_names),),
                "names": self.environment_state_names,
            },
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
            "next.success": {"dtype": "bool", "shape": (1,), "names": None},
            "sim.seed": {"dtype": "int64", "shape": (1,), "names": None},
            "control.phase": {"dtype": "int64", "shape": (1,), "names": ["phase_index"]},
            "source.timestamp.reference_ns": {"dtype": "int64", "shape": (1,), "names": None},
            "source.timestamp.body_feedback_ns": {"dtype": "int64", "shape": (1,), "names": None},
            "source.timestamp.hand_feedback_ns": {"dtype": "int64", "shape": (1,), "names": None},
            "source.timestamp.command_ns": {"dtype": "int64", "shape": (1,), "names": None},
        }
        for sensor_name, _, _ in self.cameras:
            features[f"source.timestamp.{sensor_name}_acquisition_ns"] = {
                "dtype": "int64", "shape": (1,), "names": None
            }
        for feature_name, shape in self.image_shapes.items():
            features[feature_name] = {
                "dtype": "video",
                "shape": (3, shape[0], shape[1]),
                "names": ["channels", "height", "width"],
            }
        return features

    @staticmethod
    def _numpy(tensor, dtype=np.float32):
        return tensor.detach().cpu().numpy().astype(dtype, copy=True)

    def capture(self, reference_time_ns: int) -> FrameSnapshot:
        origin = self.env.scene.env_origins[0]
        state_parts = []
        for _, body_id in self.body_entries:
            position = self.robot.data.body_pos_w[0, body_id] - origin
            orientation = self.robot.data.body_quat_w[0, body_id]
            state_parts.append(self._numpy(torch.cat((position, orientation))))
        for name in self.rigid_object_names:
            rigid_object = self.env.scene[name]
            position = rigid_object.data.root_pos_w[0] - origin
            orientation = rigid_object.data.root_quat_w[0]
            state_parts.append(
                self._numpy(
                    torch.cat(
                        (
                            position,
                            orientation,
                            rigid_object.data.root_lin_vel_w[0],
                            rigid_object.data.root_ang_vel_w[0],
                        )
                    )
                )
            )

        return FrameSnapshot(
            joint_position=self._numpy(self.robot.data.joint_pos[0, self.joint_ids]),
            joint_velocity=self._numpy(self.robot.data.joint_vel[0, self.joint_ids]),
            environment_state=np.concatenate(state_parts).astype(np.float32, copy=False),
            images={
                feature_name: self._numpy(camera.data.output["rgb"][0, ..., :3], dtype=np.uint8)
                for _, feature_name, camera in self.cameras
            },
            control_phase=np.asarray(
                [int(getattr(self.env, "policy_phase", torch.zeros(1, device=self.env.device))[0].item())],
                dtype=np.int64,
            ),
            source_timestamps_ns={
                "source.timestamp.reference_ns": np.asarray([reference_time_ns], dtype=np.int64),
                "source.timestamp.body_feedback_ns": np.asarray([reference_time_ns], dtype=np.int64),
                "source.timestamp.hand_feedback_ns": np.asarray([reference_time_ns], dtype=np.int64),
                **{
                    f"source.timestamp.{sensor_name}_acquisition_ns": np.asarray(
                        [reference_time_ns], dtype=np.int64
                    )
                    for sensor_name, _, _ in self.cameras
                },
            },
        )

    def complete_frame(self, snapshot, reward, terminated, timed_out, horizon, task, seed, success=None) -> dict:
        success = bool(terminated[0].item()) if success is None else bool(success)
        done = success or bool(timed_out[0].item()) or horizon
        frame = {
            "observation.state": snapshot.joint_position,
            "observation.velocity": snapshot.joint_velocity,
            "action": self._numpy(
                self.action_term.processed_actions[0, self.action_contract_indices]
            ),
            "sim.action.normalized": self._numpy(
                self.action_term.raw_actions[0, self.action_contract_indices]
            ),
            "observation.environment_state": snapshot.environment_state,
            "next.reward": np.asarray([reward[0].item()], dtype=np.float32),
            "next.done": np.asarray([done], dtype=np.bool_),
            "next.success": np.asarray([success], dtype=np.bool_),
            "sim.seed": np.asarray([seed], dtype=np.int64),
            "control.phase": snapshot.control_phase,
            "source.timestamp.command_ns": snapshot.source_timestamps_ns[
                "source.timestamp.reference_ns"
            ].copy(),
            "task": task,
        }
        frame.update(snapshot.source_timestamps_ns)
        frame.update(snapshot.images)
        return frame
