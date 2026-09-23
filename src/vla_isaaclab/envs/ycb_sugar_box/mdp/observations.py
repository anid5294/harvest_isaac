"""Sugar-box observation terms."""

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def left_ee_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    return torch.cat(
        (robot.data.body_pos_w[:, body_id] - env.scene.env_origins, robot.data.body_quat_w[:, body_id]),
        dim=-1,
    )


def object_state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    obj: RigidObject = env.scene[asset_cfg.name]
    return torch.cat(
        (
            obj.data.root_pos_w - env.scene.env_origins,
            obj.data.root_quat_w,
            obj.data.root_lin_vel_w,
            obj.data.root_ang_vel_w,
        ),
        dim=-1,
    )
