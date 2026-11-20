"""Orchard apple-harvesting environment for Isaac Lab 3.0.

Importing this package registers two Gym tasks (import it only after the simulator has been
launched with ``AppLauncher``, like any Isaac Lab task package):

* ``Harvest-Orchard-v0``: replica of harvest_isaac's OrchardBench pick task (1 env, cameras,
  43-D action, 129-D observation). Needs ``--enable_cameras`` (or a window).
* ``Harvest-Orchard-RL-v0``: same task for RL (many envs, no cameras, state observation).
* ``Harvest-Orchard-Rough-v0`` / ``Harvest-Orchard-RL-Rough-v0``: the same two tasks on mild
  rough orchard ground (``isaaclab.terrains``; see :mod:`harvest_rl.terrain`).
"""

import gymnasium as gym

TASK_ID = "Harvest-Orchard-v0"
RL_TASK_ID = "Harvest-Orchard-RL-v0"
ROUGH_TASK_ID = "Harvest-Orchard-Rough-v0"
ROUGH_RL_TASK_ID = "Harvest-Orchard-RL-Rough-v0"

for _task, _cfg in (
    (TASK_ID, "OrchardEnvCfg"),
    (RL_TASK_ID, "OrchardRLEnvCfg"),
    (ROUGH_TASK_ID, "OrchardRoughEnvCfg"),
    (ROUGH_RL_TASK_ID, "OrchardRoughRLEnvCfg"),
):
    if _task not in gym.registry:
        gym.register(
            id=_task,
            entry_point=f"{__name__}.orchard_env:OrchardEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{__name__}.orchard_env_cfg:{_cfg}"},
        )
