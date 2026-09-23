"""Registered scene-preview environments."""

import gymnasium as gym


_ENVIRONMENTS = {
    "VLA-ScenePreview-YCB-G1-v0": "YCBScenePreviewEnvCfg",
    "VLA-ScenePreview-Dinnerware-G1-v0": "DinnerwareScenePreviewEnvCfg",
    "VLA-ScenePreview-Microwave-G1-v0": "MicrowaveScenePreviewEnvCfg",
}

for env_id, cfg_name in _ENVIRONMENTS.items():
    if env_id not in gym.registry:
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{cfg_name}"},
            disable_env_checker=True,
        )


__all__ = ["_ENVIRONMENTS"]
