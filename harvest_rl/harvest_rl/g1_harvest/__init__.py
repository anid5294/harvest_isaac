"""Unitree G1 apple-harvesting benchmark: randomized trees, apples and stems, rough orchard
ground, a torso-mounted basket and joint-level control (environment only; no training here).

Importing registers one Gym task per stage of the training sequence (import after the
simulator is launched, like any Isaac Lab task package). Each carries an rsl_rl PPO config entry
point so Isaac Lab's ``train.py``/``play.py`` can use it; this package itself never trains.

=================================  =====================================================
Gym ID                             Config (``harvest_rl.g1_harvest.env_cfg``)
=================================  =====================================================
``Harvest-G1-S1-Validate-v0``      ``G1HarvestValidateEnvCfg``
``Harvest-G1-S2-ReachGrasp-v0``    ``G1HarvestReachGraspEnvCfg``
``Harvest-G1-S3-Pluck-v0``         ``G1HarvestPluckEnvCfg``
``Harvest-G1-S4-Place-v0``         ``G1HarvestPlaceEnvCfg``
``Harvest-G1-S5-Full-v0``          ``G1HarvestFullEnvCfg``
``Harvest-G1-S6-HeldOut-v0``       ``G1HarvestHeldOutEnvCfg``
=================================  =====================================================
"""

import gymnasium as gym

TASKS = {
    "Harvest-G1-S1-Validate-v0": "G1HarvestValidateEnvCfg",
    "Harvest-G1-S2-ReachGrasp-v0": "G1HarvestReachGraspEnvCfg",
    "Harvest-G1-S3-Pluck-v0": "G1HarvestPluckEnvCfg",
    "Harvest-G1-S4-Place-v0": "G1HarvestPlaceEnvCfg",
    "Harvest-G1-S5-Full-v0": "G1HarvestFullEnvCfg",
    "Harvest-G1-S6-HeldOut-v0": "G1HarvestHeldOutEnvCfg",
}


def register():
    for task_id, cfg_name in TASKS.items():
        if task_id not in gym.registry:
            gym.register(
                id=task_id,
                entry_point=f"{__name__}.env:G1HarvestEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{__name__}.env_cfg:{cfg_name}",
                    "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1HarvestPPORunnerCfg",
                    "default_agent": "rsl_rl",
                },
            )


register()
