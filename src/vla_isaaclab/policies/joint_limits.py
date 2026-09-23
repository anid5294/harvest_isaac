"""Helpers matching Isaac Lab's JointPositionToLimitsAction mapping."""

import torch

from vla_isaaclab.envs.common.managers import ACTION_TERM_NAME


def resolved_action_joints(env):
    """Return the built-in action term's resolved names and articulation indices."""
    term = env.action_manager.get_term(ACTION_TERM_NAME)
    names = list(term._joint_names)
    joint_ids, found = env.scene[term.cfg.asset_name].find_joints(names, preserve_order=True)
    if list(found) != names:
        raise RuntimeError(f"Resolved action joint mismatch: term={names}, articulation={found}")
    return term, names, joint_ids


def joint_targets_to_normalized(robot, joint_ids, targets: torch.Tensor) -> torch.Tensor:
    """Invert Isaac Lab's linear [-1, 1] to soft-limit transformation."""
    limits = robot.data.soft_joint_pos_limits[:, joint_ids]
    lower, upper = limits[..., 0], limits[..., 1]
    span = (upper - lower).clamp_min(1.0e-6)
    return torch.clamp(2.0 * (targets - lower) / span - 1.0, -1.0, 1.0)
