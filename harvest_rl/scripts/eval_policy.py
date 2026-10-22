"""Evaluate a trained pick-and-place policy on random (or fixed) apple positions.

Runs the policy deterministically (no exploration noise) without a window until enough
episodes have finished, then prints how often each harvest stage was reached.

    isaaclab.bat -p C:\\isaac\\harvest_rl\\scripts\\eval_policy.py --checkpoint <model.pt>
    isaaclab.bat -p C:\\isaac\\harvest_rl\\scripts\\eval_policy.py --checkpoint <model.pt> --fixed
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a pick-and-place policy.")
parser.add_argument("--checkpoint", required=True, help="Path to an rsl_rl model_*.pt file.")
parser.add_argument("--episodes", type=int, default=200, help="Episodes to evaluate.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--fixed", action="store_true", help="Keep the apple at its original spot (baseline).")
parser.add_argument("--offset_xy", type=float, default=0.08, help="Max random apple shift in x/y [m].")
parser.add_argument("--offset_z", type=float, default=0.05, help="Max random apple shift in z [m].")
parser.add_argument("--basket_xy", type=float, default=0.06, help="Max random basket shift in x/y [m].")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from importlib import metadata  # noqa: E402

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from harvest_rl.baseline.agents import PickPlacePPORunnerCfg  # noqa: E402
from harvest_rl.random_apple.env import RandomAppleEnv  # noqa: E402
from harvest_rl.random_apple.env_cfg import RandomAppleEnvCfg  # noqa: E402


def main():
    cfg = RandomAppleEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.fixed:
        cfg.apple_offset_xy = cfg.apple_offset_z = cfg.basket_offset_xy = 0.0
    else:
        cfg.apple_offset_xy, cfg.apple_offset_z = args.offset_xy, args.offset_z
        cfg.basket_offset_xy = args.basket_xy
    env = RandomAppleEnv(cfg)
    agent_cfg = handle_deprecated_rsl_rl_cfg(PickPlacePPORunnerCfg(), metadata.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    obs = wrapped.get_observations()
    step = 0
    while env.counts["episodes"] < args.episodes and simulation_app.is_running():
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
            if hasattr(policy, "reset"):
                policy.reset(dones)
        step += 1
        if step % 150 == 0:
            print(f"  ... {env.counts['episodes']} / {args.episodes} episodes finished")

    c = env.counts
    n = max(1, c["episodes"])
    mode = "fixed apple and basket" if args.fixed else (
        f"random apple (+-{args.offset_xy} m xy, +-{args.offset_z} m z), basket +-{args.basket_xy} m xy")
    print(f"\n=== {args.checkpoint}\n=== {mode}, {c['episodes']} episodes, deterministic policy")
    for key in RandomAppleEnv.OUTCOMES[1:]:
        print(f"  {key:34s} {100.0 * c[key] / n:6.1f} %   ({c[key]})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
