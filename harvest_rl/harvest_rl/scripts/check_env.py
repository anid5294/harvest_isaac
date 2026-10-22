"""Headless self-test of the orchard environment's physics and task bookkeeping.

Checks that the apple hangs still on its stem, that a pull above the break force snaps the
stem while a weaker pull does not, that harvest_isaac's failure/success rules fire as they
should, that contacts are sensed, and that reset re-attaches the apple. No policy involved;
test forces are applied directly to the apple.

    isaaclab.bat -p C:\\isaac\\harvest_rl\\scripts\\check_env.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

from harvest_rl.orchard_env import OrchardEnv  # noqa: E402
from harvest_rl.orchard_env_cfg import FRUIT, OrchardRLEnvCfg  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def failure_name(env, i):
    return env.FAILURE_NAMES[int(env.get_state()["failure"][i])]


def place_apple(env, i, pos_local):
    """Test helper: put env i's (already detached) apple at a position, at rest."""
    e = torch.tensor([i], device=env.device)
    pose = torch.zeros(1, 7, device=env.device)
    pose[0, :3] = pos_local + env.scene.env_origins[i]
    pose[0, 6] = 1.0
    env._apple.write_root_pose_to_sim_index(root_pose=pose, env_ids=e)
    env._apple.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=e)


def main():
    cfg = OrchardRLEnvCfg()
    cfg.scene.num_envs = args.num_envs
    # Keep finished episodes around so their outcome can be inspected.
    cfg.terminate_on_success = False
    cfg.terminate_on_failure = False
    env = OrchardEnv(cfg)
    env.reset()
    hold = env.home_action()
    print(f"action dim {env.single_action_space.shape}, obs dim {env.single_observation_space['policy'].shape}")
    print(f"controlled joints ({env.num_controlled_joints}): {env.ctrl_joint_names}")
    print(f"tracked bodies: {env.tracked_body_names}; contact bodies: {len(env.contact_body_names)}")

    # 1. Apple hangs still and the robot holds its home pose.
    for _ in range(60):
        env.step(hold)
    s = env.get_state()
    center = torch.tensor(FRUIT.center, device=env.device)
    drift = torch.linalg.norm(s["apple_pos"] - center, dim=-1).max().item()
    check("apple hangs at its tree.json position", drift < 0.005, f"(max drift {drift * 1000:.2f} mm)")
    check("apple attached in all envs", bool(s["apple_attached"].all()))
    tension = s["stem_tension"].mean().item()
    check("stem carries the apple's weight", abs(tension - FRUIT.mass * 9.81) < 0.3,
          f"(tension {tension:.2f} N, weight {FRUIT.mass * 9.81:.2f} N)")
    track = (s["ctrl_joint_pos"] - s["ctrl_joint_target"]).abs().max().item()
    check("robot holds home pose", track < 0.05, f"(max joint error {track:.4f} rad)")
    check("no failure while idle", bool((s["failure"] == 0).all()))
    check("observations finite", bool(torch.isfinite(env._get_observations()["policy"]).all()))

    # 2. Pull env 0 well above the break force, env 1 below it.
    weak = 0.5 * FRUIT.stem_break_force
    strong = 2.0 * FRUIT.stem_break_force
    env.apple_external_force[0, 2] = -strong
    env.apple_external_force[1, 2] = -weak
    for _ in range(10):
        env.step(hold)
        if not env.apple_attached[0]:
            env.apple_external_force[0] = 0.0  # stop pulling once it snaps
            break
    s = env.get_state()
    check(f"{strong:.0f} N pull snaps the stem", not bool(s["apple_attached"][0]))
    check("snapping without a grip is a failure (harvest_isaac rule)",
          failure_name(env, 0) == "detached_without_sustained_grasp", f"({failure_name(env, 0)})")
    for _ in range(10):
        env.step(hold)
    s = env.get_state()
    env.apple_external_force[:] = 0.0
    check(f"{weak:.0f} N pull does not", bool(s["apple_attached"][1]), f"(tension {s['stem_tension'][1].item():.1f} N)")

    for _ in range(45):
        env.step(hold)
    s = env.get_state()
    z0 = s["apple_pos"][0, 2].item()
    check("snapped apple falls", z0 < FRUIT.center[2] - 0.3, f"(z = {z0:.2f} m)")
    check("dropped apple flagged", failure_name(env, 0) == "apple_dropped", f"({failure_name(env, 0)})")
    check("other apples still attached", bool(s["apple_attached"][1:].all()))

    # 3. Basket: an apple resting in the basket is sensed, but is NOT a success unless it was
    #    carried there (harvest_isaac never credits success without the measured harvest).
    if env.num_envs > 2:
        env.apple_attached[2] = False
        place_apple(env, 2, env._basket_pos[2] + torch.tensor([0.0, 0.0, 0.06], device=env.device))
        for _ in range(60):
            env.step(hold)
        s = env.get_state()
        floor = s["basket_contact_force"][2].item()
        check("basket floor contact sensed", abs(floor - FRUIT.mass * 9.81) < 0.5,
              f"({floor:.2f} N vs weight {FRUIT.mass * 9.81:.2f} N)")
        check("apple settled + supported in basket", bool(s["apple_settled_in_basket"][2] & s["basket_supported"][2]))
        check("no success without carrying", not bool(s["success"][2]))

    # 4. Finger contact sensing: drop a loose apple 5 cm above each grasp-hand finger link in
    #    turn (fingertips first) until one registers a touch.
    if env.num_envs > 3 and env.contact_body_names:
        env.apple_attached[3] = False
        links = list(reversed(cfg.robot.grasp_finger_bodies + cfg.robot.grasp_thumb_bodies))
        peak, hit = 0.0, None
        for link in links:
            link_id = env._robot.find_bodies([link])[0][0]
            link_pos = env._robot.data.body_pos_w.torch[3, link_id] - env.scene.env_origins[3]
            place_apple(env, 3, link_pos + torch.tensor([0.0, 0.0, 0.05], device=env.device))
            for _ in range(20):
                env.step(hold)
                peak = max(peak, env.get_state()["finger_contact_force"][3].max().item())
            if peak > 0.05:
                hit = link
                break
        check("finger contact sensors report touches", hit is not None, f"(peak {peak:.2f} N on drop over {hit})")

    # 5. Reset re-attaches and clears the outcome.
    env.reset()
    for _ in range(10):
        env.step(hold)
    s = env.get_state()
    drift = torch.linalg.norm(s["apple_pos"] - center, dim=-1).max().item()
    check("reset re-attaches every apple", bool(s["apple_attached"].all()) and drift < 0.005,
          f"(max drift {drift * 1000:.2f} mm)")
    check("reset clears failures", bool((s["failure"] == 0).all()))

    print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED: {failures}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
