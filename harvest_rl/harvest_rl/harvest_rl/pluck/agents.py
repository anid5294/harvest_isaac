"""PPO settings for the grip-pluck-hold task: same as the baseline, own log folder."""

from isaaclab.utils import configclass

from ..baseline.agents import PickPlacePPORunnerCfg


@configclass
class PluckPPORunnerCfg(PickPlacePPORunnerCfg):
    experiment_name = "harvest_pluck"
