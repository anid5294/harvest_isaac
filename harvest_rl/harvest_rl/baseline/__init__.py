"""Baseline PPO pick-and-place task (``Harvest-Orchard-PickPlace-v0``).

Train with Isaac Lab's rsl_rl trainer; ``--external_callback`` registers the task:

    isaaclab.bat -p scripts\\reinforcement_learning\\train.py --rl_library rsl_rl ^
        --external_callback harvest_rl.baseline.register --task Harvest-Orchard-PickPlace-v0
"""

import gymnasium as gym

TASK_ID = "Harvest-Orchard-PickPlace-v0"


def register():
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point=f"{__name__}.pick_place_env:PickPlaceEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:PickPlaceEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents:PickPlacePPORunnerCfg",
                "default_agent": "rsl_rl",
            },
        )


register()
