"""Validate the rough orchard ground (task 1): visible, collidable, the G1 stands on it, and the
orchard (tree, apple, stem, basket) still works and resets correctly on top of it.

Two modes, one simulator launch each:

    :: physics + orchard checks (default)
    isaaclab.bat -p scripts\\check_terrain.py --mode orchard
    :: save the fixed camera's RGB view of the rough orchard to outputs/terrain_snapshot.png
    isaaclab.bat -p scripts\\check_terrain.py --mode snapshot

"The G1 can stand on it" is checked for the robot as the harvest task uses it (pelvis fixed,
feet on the stance pad at their authored height). A free-floating G1 needs a balance
controller: with PD-only joint holding it folds over on flat ground as well, so that is not a
terrain property and is not tested here.

Every check prints PASS/FAIL; the script exits non-zero if any check fails.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--mode", choices=("orchard", "snapshot"), default="orchard")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--out", default="outputs/terrain_snapshot.png")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.mode == "snapshot":
    args.enable_cameras = True
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from harvest_rl.orchard_env import OrchardEnv  # noqa: E402
from harvest_rl.orchard_env_cfg import FRUIT, OrchardRoughRLEnvCfg  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


_MESH = []


def terrain_trimesh():
    """The terrain as a trimesh, read back from the USD prim the importer created (world frame)."""
    if not _MESH:
        import trimesh  # noqa: PLC0415
        from pxr import UsdGeom  # noqa: PLC0415

        import isaaclab.sim as sim_utils  # noqa: PLC0415

        stage = sim_utils.get_current_stage()
        prim = next(p for p in stage.Traverse() if str(p.GetPath()).startswith("/World/ground") and p.IsA(UsdGeom.Mesh))
        geom = UsdGeom.Mesh(prim)
        pts = np.array(geom.GetPointsAttr().Get(), dtype=np.float64)
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0), dtype=np.float64)  # row-vector convention
        pts = pts @ xf[:3, :3] + xf[3, :3]
        faces = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        _MESH.append(trimesh.Trimesh(vertices=pts, faces=faces, process=False))
    return _MESH[0]


def terrain_height(env, xy_world: np.ndarray) -> np.ndarray:
    """Ground height under world (x, y) points, by ray casting the imported terrain mesh."""
    mesh = terrain_trimesh()
    origins = np.column_stack([xy_world, np.full(len(xy_world), 10.0)])
    dirs = np.tile([0.0, 0.0, -1.0], (len(xy_world), 1))
    locs, ray_ids, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    z = np.full(len(xy_world), np.nan)
    z[ray_ids] = locs[:, 2]
    return z


def ground_prims_ok():
    from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415

    import isaaclab.sim as sim_utils  # noqa: PLC0415

    stage = sim_utils.get_current_stage()
    meshes = [p for p in stage.Traverse() if str(p.GetPath()).startswith("/World/ground") and p.IsA(UsdGeom.Mesh)]
    collides = [p for p in meshes if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
    visible = [p for p in meshes if UsdGeom.Imageable(p).ComputeVisibility() != UsdGeom.Tokens.invisible]
    return meshes, collides, visible


def place_apple(env, i, pos_local):
    e = torch.tensor([i], device=env.device)
    pose = torch.zeros(1, 7, device=env.device)
    pose[0, :3] = pos_local + env.scene.env_origins[i]
    pose[0, 6] = 1.0
    env._apple.write_root_pose_to_sim_index(root_pose=pose, env_ids=e)
    env._apple.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=e)


# ---------------------------------------------------------------------------------------


def run_orchard():
    cfg = OrchardRoughRLEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.terminate_on_success = False
    cfg.terminate_on_failure = False
    env = OrchardEnv(cfg)
    env.reset()
    hold = env.home_action()
    origins = env.scene.env_origins

    # 1. Terrain exists, is visible and collidable.
    meshes, collides, visible = ground_prims_ok()
    check("terrain mesh imported under /World/ground", len(meshes) > 0, f"({len(meshes)} mesh prims)")
    check("terrain collision enabled", len(collides) == len(meshes) and len(meshes) > 0)
    check("terrain visible", len(visible) == len(meshes) and len(meshes) > 0)
    zr = terrain_trimesh().vertices[:, 2]
    check("terrain is actually rough", zr.max() - zr.min() > 0.01, f"(height range {zr.min():.3f}..{zr.max():.3f} m)")

    # 2. Env origins are the cloned grid (where the static tree is), and the pads are at z = 0.
    grid = env.scene.clone_plan.positions.to(origins.device)
    check("env origins = cloned env grid", torch.allclose(origins[:, :2], grid[:, :2], atol=1e-4))
    o = origins.cpu().numpy()
    keep = cfg.rough_ground_keepouts()
    for name, (kx, ky, r) in zip(("tree base", "robot feet", "basket pedestal"), keep):
        ring = np.array([[kx + r * 0.8 * np.cos(a), ky + r * 0.8 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 12)] + [[kx, ky]])
        z = np.concatenate([terrain_height(env, o[i, :2] + ring) for i in range(env.num_envs)])
        check(f"{name} pad is flat at the authored ground height", np.nanmax(np.abs(z)) < 1e-4 and not np.isnan(z).any(),
              f"(max |z| {np.nanmax(np.abs(z)) * 1000:.2f} mm)")

    # 3. Orchard still works on it: apple hangs, robot holds home pose, no failure.
    for _ in range(60):
        env.step(hold)
    s = env.get_state()
    center = torch.tensor(FRUIT.center, device=env.device)
    drift = torch.linalg.norm(s["apple_pos"] - center, dim=-1).max().item()
    check("apple hangs at its tree.json position", drift < 0.005, f"(max drift {drift * 1000:.2f} mm)")
    track = (s["ctrl_joint_pos"] - s["ctrl_joint_target"]).abs().max().item()
    check("robot holds home pose", track < 0.05, f"(max joint error {track:.4f} rad)")
    feet = [env._robot.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    feet_z = (env._robot.data.body_pos_w.torch[:, feet, 2] - origins[:, 2:3]).flatten()
    check("feet neither buried nor floating", bool(((feet_z - 0.047).abs() < 0.01).all()),
          f"(ankle z {feet_z.min().item():.3f}..{feet_z.max().item():.3f} m, flat-ground value 0.047)")
    check("no failure while idle", bool((s["failure"] == 0).all()))

    # 4. Collisions: drop the (detached) apple onto rough ground away from the pads; it must
    #    come to rest on the surface, not fall through.
    rng = np.random.default_rng(0)
    drops = []
    for i in range(env.num_envs):
        while True:
            xy = rng.uniform(-1.8, 1.8, size=2)
            if all(np.hypot(*(xy - np.array(k[:2]))) > k[2] + 0.5 for k in keep):
                break
        drops.append(xy)
        env.apple_attached[i] = False
        place_apple(env, i, torch.tensor([xy[0], xy[1], 0.4], device=env.device, dtype=torch.float32))
    for _ in range(90):
        env.step(hold)
    # Apples roll on the slopes, so measure the ground under where each one came to rest.
    apple = (env._apple.data.root_pos_w.torch - origins).cpu().numpy()
    ground = np.array([terrain_height(env, o[i, :2] + apple[i, None, :2])[0] for i in range(env.num_envs)])
    rest = apple[:, 2] - ground
    speed = torch.linalg.norm(env._apple.data.root_lin_vel_w.torch, dim=-1).max().item()
    print(f"[INFO] dropped apples rolled {np.linalg.norm(apple[:, :2] - np.array(drops), axis=1).max():.2f} m at most; "
          f"max speed now {speed:.3f} m/s")
    lo, hi = FRUIT.half_extents[2] - 0.01, max(FRUIT.half_extents) + 0.01
    check("dropped apples rest on the rough surface", bool(((rest > lo) & (rest < hi)).all()),
          f"(centre above ground {rest.min():.3f}..{rest.max():.3f} m, expected {lo:.3f}..{hi:.3f}; "
          f"ground under them {ground.min():+.3f}..{ground.max():+.3f} m)")

    # 5. Reset restores the orchard on top of the terrain.
    env.reset()
    for _ in range(10):
        env.step(hold)
    s = env.get_state()
    drift = torch.linalg.norm(s["apple_pos"] - center, dim=-1).max().item()
    check("reset re-attaches every apple", bool(s["apple_attached"].all()) and drift < 0.005, f"(max drift {drift * 1000:.2f} mm)")
    check("reset clears failures", bool((s["failure"] == 0).all()))
    root = s["root_pos"]
    stance = torch.tensor(cfg.scene.robot.init_state.pos, device=env.device)
    check("robot back at its stance after reset", torch.allclose(root, stance.expand_as(root), atol=1e-3))
    env.close()


def run_snapshot():
    """One RGB view of a rough tile with the orchard on it (same recipe as record_episode.py)."""
    import os  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.sensors import CameraCfg  # noqa: PLC0415

    from harvest_rl.orchard_env_cfg import _look_at_xyzw  # noqa: PLC0415

    cfg = OrchardRoughRLEnvCfg()
    cfg.scene.num_envs = 1
    cfg.ground_difficulty = (1.0, 1.0)
    cfg.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
    eye, target = (2.6, -3.4, 1.9), (0.3, -0.6, 0.5)
    cfg.scene.snapshot_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/SnapshotCamera", update_period=0.0, width=1280, height=720, data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(pos=eye, rot=_look_at_xyzw(eye, target), convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, focus_distance=3.0, horizontal_aperture=20.955, clipping_range=(0.05, 30.0)),
    )
    env = OrchardEnv(cfg)
    env.reset()
    hold = env.home_action()
    for _ in range(30):
        env.step(hold)
    camera = env.scene["snapshot_camera"]
    for _ in range(5):
        env.sim.render()
        camera.update(dt=0.0, force_recompute=True)
    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Image.fromarray(rgb).save(args.out)
    check("snapshot saved", rgb.std() > 1.0, f"({args.out}, {rgb.shape[1]}x{rgb.shape[0]})")
    env.close()


if __name__ == "__main__":
    {"orchard": run_orchard, "snapshot": run_snapshot}[args.mode]()
    print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED: {failures}", flush=True)
    simulation_app.close()
    sys.exit(1 if failures else 0)
