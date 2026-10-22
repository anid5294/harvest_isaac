"""Open the orchard environment and watch it, with no policy.

The robot holds its home pose (or wiggles randomly with --random) and the key state values
are printed once per second. --pull demonstrates the stem: a test force ramps up on the
apple until it snaps off (a physics demo, not a policy).

    isaaclab.bat -p C:\\isaac\\harvest_rl\\scripts\\explore.py --viz kit
    isaaclab.bat -p C:\\isaac\\harvest_rl\\scripts\\explore.py --viz kit --num_envs 4 --pull
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Watch the orchard environment without a policy.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--random", action="store_true", help="Small random joint motions around the home pose.")
parser.add_argument("--pull", action="store_true", help="Ramp a test force on the apple until the stem snaps.")
parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps (0 = until window closed).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

from harvest_rl.orchard_env import OrchardEnv  # noqa: E402
from harvest_rl.orchard_env_cfg import OrchardEnvCfg  # noqa: E402


def main():
    cfg = OrchardEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.terminate_on_drop = not args.pull  # let the snapped apple lie there during the demo
    env = OrchardEnv(cfg)
    env.reset()
    print(f"\nAction: {env.num_controlled_joints} joints ({cfg.action_mode} mode): {env.ctrl_joint_names}")
    print(f"Observation size: {env.single_observation_space['policy'].shape[0]}")
    print(f"State keys: {list(env.get_state().keys())}\n")

    hz = round(1.0 / env.step_dt)
    step = 0
    while simulation_app.is_running() and (args.steps == 0 or step < args.steps):
        action = env.home_action()
        if args.random:
            action = action + 0.05 * torch.randn_like(action)
        if args.pull:
            # Downward test force growing 2 N per second; restarts (with a fresh apple) every 15 s.
            t = step % (15 * hz)
            if t == 0 and step > 0:
                env._reset_idx(None)
            env.apple_external_force[:, 2] = torch.where(env.apple_attached, -2.0 * t / hz, 0.0)
        env.step(action)
        step += 1
        if step % hz == 0:
            s = env.get_state()
            print(
                f"t={step / hz:5.1f}s  attached={s['apple_attached'][0].item()!s:5}  "
                f"stem={s['stem_tension'][0].item():5.2f}/{env.stem_break_force[0].item():.1f} N  "
                f"apple z={s['apple_pos'][0, 2].item():.3f} m  "
                f"palm->apple={torch.linalg.norm(s['body_pos'][0, 0] - s['apple_pos'][0]).item():.3f} m  "
                f"finger force={s['finger_contact_force'][0].sum().item():.2f} N  "
                f"in basket={s['apple_in_basket'][0].item()}"
            )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
