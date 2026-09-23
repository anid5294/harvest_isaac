"""Sugar-box reward terms."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .terminations import task_metrics


def placement_reward(env: ManagerBasedRLEnv, palm_body_name: str, command_name: str) -> torch.Tensor:
    metrics = task_metrics(env, palm_body_name, command_name)
    return 1.0 - torch.tanh(25.0 * metrics["xy_error"] + 10.0 * metrics["height_error"])
