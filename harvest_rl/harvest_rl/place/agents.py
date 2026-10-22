"""PPO settings for the pluck-and-place task: same as the baseline, own log folder."""

from isaaclab.utils import configclass

from ..baseline.agents import PickPlacePPORunnerCfg


@configclass
class PlacePPORunnerCfg(PickPlacePPORunnerCfg):
    experiment_name = "harvest_place"
