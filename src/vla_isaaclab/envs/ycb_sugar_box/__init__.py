"""Registered YCB sugar-box manipulation environment."""

import gymnasium as gym


ENV_ID = "VLA-YCBSugarBox-G1-JointPos-v0"

if ENV_ID not in gym.registry:
    gym.register(
        id=ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:YCBSugarBoxEnvCfg"},
        disable_env_checker=True,
    )


__all__ = ["ENV_ID"]
