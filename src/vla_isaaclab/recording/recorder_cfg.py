"""RecorderManager terms shared by registered environments."""

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from .hdf5 import CompressedHDF5DatasetFileHandler


class CameraCalibrationRecorder(RecorderTerm):
    def record_post_reset(self, env_ids):
        cameras = {
            name: {
                "intrinsic_matrix": camera.data.intrinsic_matrices[env_ids],
                "position_world": camera.data.pos_w[env_ids],
                "orientation_world": camera.data.quat_w_world[env_ids],
            }
            for name, camera in self._env.scene.sensors.items()
            if "rgb" in camera.data.output
        }
        return "sensors/cameras/calibration", cameras


class CameraFrameRecorder(RecorderTerm):
    def record_post_step(self):
        cameras = {
            name: {"rgb": camera.data.output["rgb"][..., :3]}
            for name, camera in self._env.scene.sensors.items()
            if "rgb" in camera.data.output
        }
        return "sensors/cameras/frames", cameras


class TaskMetricRecorder(RecorderTerm):
    def record_post_step(self):
        if "object" in self._env.scene.rigid_objects and "target_pose" in self._env.command_manager.active_terms:
            obj = self._env.scene["object"]
            target = self._env.command_manager.get_command("target_pose")[:, :3] + self._env.scene.env_origins
            distance = torch.linalg.vector_norm(obj.data.root_pos_w - target, dim=-1, keepdim=True)
        elif "object" in self._env.scene.rigid_objects and "goal" in self._env.scene.rigid_objects:
            obj = self._env.scene["object"]
            goal = self._env.scene["goal"]
            distance = torch.linalg.vector_norm(obj.data.root_pos_w - goal.data.root_pos_w, dim=-1, keepdim=True)
        else:
            distance = torch.zeros((self._env.num_envs, 1), device=self._env.device)
        return "task", {"distance": distance}


@configclass
class CameraCalibrationRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CameraCalibrationRecorder


@configclass
class CameraFrameRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CameraFrameRecorder


@configclass
class TaskMetricRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = TaskMetricRecorder


@configclass
class VLARecorderCfg(ActionStateRecorderManagerCfg):
    dataset_file_handler_class_type: type = CompressedHDF5DatasetFileHandler
    camera_calibration = CameraCalibrationRecorderCfg()
    camera_frames = CameraFrameRecorderCfg()
    task_metrics = TaskMetricRecorderCfg()
