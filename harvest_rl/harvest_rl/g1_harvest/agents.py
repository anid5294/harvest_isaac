"""rsl_rl PPO settings referenced by the Gym registrations (so Isaac Lab's train/play scripts
can run these tasks later). Nothing in this package starts training."""

from isaaclab.utils import configclass

from ..baseline.agents import PickPlacePPORunnerCfg


@configclass
class G1HarvestPPORunnerCfg(PickPlacePPORunnerCfg):
    experiment_name = "g1_harvest"
