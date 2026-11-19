"""Small boundary between Isaac Lab 2.0.2 and 3.0 asset APIs."""

from __future__ import annotations

import importlib.util

import torch


LAB3 = importlib.util.find_spec("isaaclab_physx") is not None


def as_torch(value):
    """Lab 3 asset buffers wrap tensors; Lab 2 buffers are tensors."""
    return value.torch if hasattr(value, "torch") else value


def default_root_pose(data):
    """Return the default root pose; Lab 2 stores it in the 13-D root state."""
    if LAB3:
        return as_torch(data.default_root_pose)
    return as_torch(data.default_root_state)[..., :7]


def asset_call(asset, name: str, **kwargs):
    """Call a Lab 3 indexed writer or its Lab 2 counterpart."""
    method = f"{name}_index" if LAB3 else name
    return getattr(asset, method)(**kwargs)


def backend_quat_xyzw(quat):
    """Convert a canonical xyzw tuple/tensor to the installed Lab's order."""
    if LAB3:
        return quat
    if isinstance(quat, torch.Tensor):
        return quat[..., [3, 0, 1, 2]]
    return (quat[3], quat[0], quat[1], quat[2])


def canonical_quat(quat: torch.Tensor) -> torch.Tensor:
    """Return an asset quaternion as xyzw for the public state API."""
    return quat if LAB3 else quat[..., [1, 2, 3, 0]]
