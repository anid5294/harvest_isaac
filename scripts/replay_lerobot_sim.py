#!/usr/bin/env python3
"""Isaac Sim half of LeRobot replay; consumes a neutral NumPy episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("episode_data", type=Path)
parser.add_argument("--output", type=Path, default=PROJECT / "outputs/replay/lerobot_rgb.png")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
ARGS.enable_cameras = True
ARGS.experience = str(
    PROJECT / "configs" / ("ycb.python.headless.rendering.kit" if ARGS.headless else "ycb.python.rendering.kit")
)
ARGS.kit_args = f"--portable-root {PROJECT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import numpy as np
import torch
from PIL import Image

import gymnasium as gym
import vla_isaaclab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from vla_isaaclab.envs.common import ACTION_JOINT_NAMES
from vla_isaaclab.envs.common.managers import ACTION_TERM_NAME


def main() -> int:
    with np.load(ARGS.episode_data, allow_pickle=False) as episode_data:
        actions = episode_data["actions"]
        seed = int(episode_data["seed"])
        environment_id = str(episode_data["environment_id"])
        dataset_root = str(episode_data["dataset"])
        episode_index = int(episode_data["episode"])

    cfg = parse_env_cfg(environment_id, device=ARGS.device, num_envs=1)
    cfg.seed = seed
    cfg.episode_length_s = (len(actions) + 1) * cfg.decimation * cfg.sim.dt
    env = gym.make(environment_id, cfg=cfg).unwrapped
    try:
        env.reset(seed=seed)
        action_term_names = list(env.action_manager.get_term(ACTION_TERM_NAME)._joint_names)
        contract_to_term = [ACTION_JOINT_NAMES.index(name) for name in action_term_names]
        actions = actions[:, contract_to_term]
        # Row n stores the observation before action n is advanced through the
        # simulator. Apply rows [0, N-2] to reproduce observation row N-1.
        replay_actions = actions[:-1]
        with torch.inference_mode():
            for action in replay_actions:
                env.step(torch.as_tensor(action, device=env.device).unsqueeze(0))

        ARGS.output.parent.mkdir(parents=True, exist_ok=True)
        camera_name = "cam_left_high" if "cam_left_high" in env.scene.sensors else "camera"
        rgb = env.scene.sensors[camera_name].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(ARGS.output)
        report = {
            "dataset": dataset_root,
            "episode": episode_index,
            "environment_id": environment_id,
            "seed": seed,
            "actions_available": len(actions),
            "actions_applied": len(replay_actions),
            "observation_frame_reproduced": len(actions) - 1,
            "output": str(ARGS.output.resolve()),
        }
        ARGS.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
