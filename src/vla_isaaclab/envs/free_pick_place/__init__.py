"""Single free-apple physical pick-and-place task."""

import gymnasium as gym

ENV_ID = "VLA-FreeApplePickPlace-G1-JointPos-v0"
if ENV_ID not in gym.registry:
    gym.register(
        id=ENV_ID,
        entry_point=f"{__name__}.env:FreePickPlaceEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:FreePickPlaceEnvCfg"},
        disable_env_checker=True,
    )
