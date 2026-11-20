"""Render a grid of G1 harvesting envs (different trees, tree poses, apples) to a PNG.

    isaaclab.bat -p scripts\\snapshot_g1_harvest.py --cfg G1HarvestFullEnvCfg --num_envs 4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--cfg", default="G1HarvestFullEnvCfg")
parser.add_argument("--num_envs", type=int, default=1, help="multi-env rendering currently comes out blank on this machine; keep 1")
parser.add_argument("--show", type=int, default=1, help="panels in the grid")
parser.add_argument("--variation", type=float, default=1.0, help="tree pose variation (curriculum level) to show")
parser.add_argument("--out", default="outputs/g1_harvest_snapshot.png")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

import harvest_rl.g1_harvest.env_cfg as env_cfgs  # noqa: E402
from harvest_rl.g1_harvest.env import G1HarvestEnv  # noqa: E402
from harvest_rl.orchard_env_cfg import _look_at_xyzw  # noqa: E402


def main():
    from isaaclab.cloner.cloner_strategies import random as random_strategy  # noqa: PLC0415

    cfg = getattr(env_cfgs, args.cfg)()
    cfg.scene.num_envs = args.num_envs
    # Random prototype assignment so a small scene still shows several tree variants.
    cfg.scene.clone_cfg.clone_strategy = random_strategy
    cfg.tree_pose_variation = args.variation
    cfg.curriculum = False
    cfg.wait_for_textures = True
    cfg.num_rerenders_on_reset = 2
    # One camera per env; its world pose is set explicitly after the scene is built.
    cfg.scene.snapshot_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/SnapshotCamera", update_period=0.0, width=960, height=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=15.0, focus_distance=3.0, horizontal_aperture=20.955, clipping_range=(0.05, 30.0)),
    )
    env = G1HarvestEnv(cfg)
    env.reset()
    hold = torch.zeros(env.num_envs, env.num_ctrl, device=env.device)
    for _ in range(40):
        env.step(hold)
    s = env.get_state()
    # Show up to `--show` envs, preferring different tree variants.
    order, seen = [], set()
    for i in range(env.num_envs):
        v = int(s["tree_variant"][i])
        if v not in seen:
            seen.add(v)
            order.append(i)
    order += [i for i in range(env.num_envs) if i not in order]
    order = order[: args.show]
    cam = env.scene["snapshot_camera"]
    eye_local = torch.tensor((2.4, -3.6, 1.8), device=env.device)
    target_local = torch.tensor((0.6, -1.2, 0.95), device=env.device)
    o = env.scene.env_origins
    cam.set_world_poses_from_view(eye_local + o, target_local + o)
    for _ in range(15):
        env.sim.render()
        cam.update(dt=0.0, force_recompute=True)
    rgb = cam.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
    print("[snapshot-debug] cam pos", cam.data.pos_w[:2].tolist(), "quat", cam.data.quat_w_world[:2].tolist(),
          "origins", o[:2].tolist(), "rgb dtype/shape", cam.data.output["rgb"].dtype, tuple(cam.data.output["rgb"].shape),
          "min/max", int(rgb.min()), int(rgb.max()))
    imgs = [rgb[i] for i in order]
    cols = math.ceil(math.sqrt(len(imgs)))
    rows = math.ceil(len(imgs) / cols)
    h, w = imgs[0].shape[:2]
    grid = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for j, im in enumerate(imgs):
        grid[(j // cols) * h:(j // cols + 1) * h, (j % cols) * w:(j % cols + 1) * w] = im
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Image.fromarray(grid).save(args.out)
    for j, i in enumerate(order):
        print(f"[snapshot] panel {j} = env {i}: tree={env.tree_names[int(s['tree_variant'][i])]} pose(x,y,yaw)="
              f"{[round(v, 2) for v in s['tree_pose'][i].tolist()]} apples={int(s['apple_active'][i].sum())} "
              f"target mass={s['apple_mass'][i, 0].item():.3f}kg scale={s['apple_scale'][i, 0].item():.2f} "
              f"attached={bool(s['apple_attached'][i])} image std={imgs[j].std():.1f}")
    print(f"[snapshot] saved {args.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
