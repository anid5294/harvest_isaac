"""Deterministic goal command used by the sugar-box task."""

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class FixedPoseCommand(CommandTerm):
    """Expose one fixed environment-local pose through CommandManager."""

    cfg: "FixedPoseCommandCfg"

    def __init__(self, cfg: "FixedPoseCommandCfg", env):
        super().__init__(cfg, env)
        self._command = torch.tensor(cfg.pose, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        self._command[env_ids] = torch.tensor(self.cfg.pose, dtype=torch.float32, device=self.device)

    def _update_command(self):
        pass


@configclass
class FixedPoseCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = FixedPoseCommand
    pose: tuple[float, float, float, float, float, float, float] = MISSING
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False
