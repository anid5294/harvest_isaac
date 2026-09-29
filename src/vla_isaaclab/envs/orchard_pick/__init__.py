"""Contact-based fixed-pelvis G1 orchard task."""

import gymnasium as gym

ENV_ID = "VLA-OrchardPick-G1-JointPos-v0"
if ENV_ID not in gym.registry:
    gym.register(
        id=ENV_ID,
        entry_point=f"{__name__}.env:OrchardPickEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:OrchardPickEnvCfg"},
        disable_env_checker=True,
    )
