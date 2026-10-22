"""Evaluate a grip-pluck-hold checkpoint: how often it grasps, plucks with a grip, and holds.

Runs the policy deterministically without a window on random apple positions (the training
distribution) until enough episodes finish, then prints the pluck statistics.

    isaaclab.bat -p C:\\isaac\\harvest_rl\\scripts\\eval_pluck.py --checkpoint <model.pt>
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a grip-pluck-hold policy.")
parser.add_argument("--checkpoint", required=True, help="Path to an rsl_rl model_*.pt file.")
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--fixed", action="store_true", help="Apple at its original spot instead of random.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from importlib import metadata  # noqa: E402

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from harvest_rl.pluck.agents import PluckPPORunnerCfg  # noqa: E402
from harvest_rl.pluck.env import PluckEnv  # noqa: E402
from harvest_rl.pluck.env_cfg import PluckEnvCfg  # noqa: E402


def main():
    cfg = PluckEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.fixed:
        cfg.apple_offset_xy = cfg.apple_offset_z = 0.0
    env = PluckEnv(cfg)
    agent_cfg = handle_deprecated_rsl_rl_cfg(PluckPPORunnerCfg(), metadata.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    obs = wrapped.get_observations()
    step = 0
    while env.pluck_counts["episodes"] < args.episodes and simulation_app.is_running():
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
            if hasattr(policy, "reset"):
                policy.reset(dones)
        step += 1
        if step % 150 == 0:
            print(f"  ... {env.pluck_counts['episodes']} / {args.episodes} episodes finished")

    c = env.pluck_counts
    n, p = max(1, c["episodes"]), max(1, c["plucked"])
    print(f"\n=== {args.checkpoint}")
    print(f"=== {'fixed' if args.fixed else 'random'} apple, {c['episodes']} episodes, deterministic policy")
    print(f"  grasped (all 3 fingers on the apple)   {100 * c['grasped'] / n:6.1f} %  ({c['grasped']})")
    print(f"  PLUCKED WITH GRIP                      {100 * c['plucked'] / n:6.1f} %  ({c['plucked']})")
    print(f"    of those, still holding at the end   {100 * c['held_to_end'] / p:6.1f} %  ({c['held_to_end']})")
    print(f"    of those, dropped later              {100 * c['dropped_after_pluck'] / p:6.1f} %  ({c['dropped_after_pluck']})")
    print(f"    average hold time after plucking     {c['hold_seconds_when_plucked'] / p:6.1f} s")
    print(f"  knocked off without a grip             {100 * c['knocked_off_without_grip'] / n:6.1f} %"
          f"  ({c['knocked_off_without_grip']})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
