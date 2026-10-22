"""Grip-pluck-hold task (``Harvest-Orchard-Pluck-v0``), no basket placing.

    isaaclab.bat -p scripts\\reinforcement_learning\\train.py --rl_library rsl_rl ^
        --external_callback harvest_rl.pluck.register --task Harvest-Orchard-Pluck-v0
"""

import gymnasium as gym

TASK_ID = "Harvest-Orchard-Pluck-v0"


def register():
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point=f"{__name__}.env:PluckEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:PluckEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents:PluckPPORunnerCfg",
                "default_agent": "rsl_rl",
            },
        )


register()
