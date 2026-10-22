"""Randomized pick-and-place: apple (and basket) at a random spot each episode.

* ``Harvest-Orchard-Randomized-v0``: training task (random apple + basket, release shaping,
  random arm start pose, random stem strength). Train with Isaac Lab's rsl_rl trainer and
  ``--external_callback harvest_rl.random_apple.register``.
* ``scripts/eval_policy.py``: evaluate any checkpoint on random or fixed apple positions.
"""

import gymnasium as gym

TASK_ID = "Harvest-Orchard-Randomized-v0"


def register():
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point=f"{__name__}.env:RandomAppleEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:RandomizedTrainEnvCfg",
                "rsl_rl_cfg_entry_point": "harvest_rl.baseline.agents:PickPlacePPORunnerCfg",
                "default_agent": "rsl_rl",
            },
        )


register()
