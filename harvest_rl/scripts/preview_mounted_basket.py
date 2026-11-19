"""Inspect the G1's pelvis-mounted basket in OrchardBench."""

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def preview(args, app):
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext
    from isaacsim.core.utils.stage import get_current_stage

    from harvest_rl.compat import as_torch, asset_call
    from harvest_rl.orchard_env_cfg import OrchardRLEnvCfg
    from harvest_rl.orchard_mounted_basket import MountedBasketSpec, build_basket_plan, spawn_mounted_basket
    from harvest_rl.orchard_mounted_env import LEGACY_BASKET_ENTITIES

    config = json.loads(args.basket_config.read_text(encoding="utf-8"))
    spec = MountedBasketSpec(**config)
    if build_basket_plan(spec) is None:
        raise ValueError("The preview requires an enabled mounted basket.")

    cfg = OrchardRLEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    # The preview uses the tree and robot from the orchard. The physical fruit
    # and its stem need the harvest task's control loop.
    for name in (*LEGACY_BASKET_ENTITIES, "apple", "stem", "apple_contact"):
        setattr(cfg.scene, name, None)

    sim = SimulationContext(cfg.sim)
    scene = InteractiveScene(cfg.scene)
    plan = spawn_mounted_basket(get_current_stage(), spec)
    sim.reset()
    robot = scene["robot"]
    position = as_torch(robot.data.default_joint_pos)
    velocity = as_torch(robot.data.default_joint_vel)
    asset_call(robot, "write_joint_position_to_sim", position=position)
    asset_call(robot, "write_joint_velocity_to_sim", velocity=velocity)
    asset_call(robot, "set_joint_position_target", target=position)
    print(f"Mounted basket authored at {plan['body_path']} via {plan['joint_path']}", flush=True)

    step = 0
    while app.is_running() and (args.steps == 0 or step < args.steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(cfg.sim.dt)
        step += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basket-config", type=Path, default=PROJECT_ROOT / "configs/orchard_mounted_basket.json")
    parser.add_argument("--steps", type=int, default=0, help="Physics steps to run, or 0 to run until the app closes.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")

    sys.path.insert(0, str(PROJECT_ROOT / "harvest_rl"))
    launcher = AppLauncher(args)
    try:
        preview(args, launcher.app)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
