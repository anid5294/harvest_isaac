"""Pluck-and-place task (``Harvest-Orchard-Place-v0``), meant to be fine-tuned from a pluck model.

    isaaclab.bat -p scripts\\reinforcement_learning\\train.py --rl_library rsl_rl ^
        --external_callback harvest_rl.place.register --task Harvest-Orchard-Place-v0 ^
        --checkpoint <pluck model.pt>
"""

import gymnasium as gym

TASK_ID = "Harvest-Orchard-Place-v0"


def register():
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point=f"{__name__}.env:PlaceEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:PlaceEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents:PlacePPORunnerCfg",
                "default_agent": "rsl_rl",
            },
        )


register()
