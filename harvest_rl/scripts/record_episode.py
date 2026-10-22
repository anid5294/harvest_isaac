"""Record one episode of a trained pick-and-place policy to an MP4.

Adds a camera to the scene (default: over the robot's left shoulder, looking at the hand,
apple and basket), runs the policy deterministically for one episode with random apple and
basket positions, and writes the frames to a video. A preview PNG of the first frame is saved
next to it so the framing can be checked quickly.

    isaaclab.bat -p C:\\isaac\\harvest_rl\\scripts\\record_episode.py --checkpoint <model.pt>
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record one episode to MP4.")
parser.add_argument("--checkpoint", required=True, help="Path to a pick-and-place model_*.pt file.")
parser.add_argument("--output", default=r"C:\isaac\harvest_rl\videos\episode.mp4")
parser.add_argument("--seed", type=int, default=0, help="Seed for the random apple/basket positions.")
parser.add_argument("--fixed", action="store_true", help="Apple and basket at their original spots.")
parser.add_argument("--eye", type=float, nargs=3, default=(0.15, -2.35, 1.65), help="Camera position (tree frame, m).")
parser.add_argument("--lookat", type=float, nargs=3, default=(0.65, -1.30, 0.95), help="Point the camera looks at.")
parser.add_argument("--episodes", type=int, default=1, help="Number of videos to save (one episode each).")
parser.add_argument("--only_success", action="store_true", help="Save only successful episodes.")
parser.add_argument("--max_attempts", type=int, default=100, help="Give up after this many episodes.")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import shutil  # noqa: E402
import subprocess  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from harvest_rl.orchard_env_cfg import _look_at_xyzw  # noqa: E402
from harvest_rl.place.agents import PlacePPORunnerCfg  # noqa: E402
from harvest_rl.place.env import PlaceEnv  # noqa: E402
from harvest_rl.place.env_cfg import PlaceEnvCfg  # noqa: E402


def main():
    torch.manual_seed(args.seed)
    cfg = PlaceEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    if args.fixed:
        cfg.apple_offset_xy = cfg.apple_offset_z = cfg.basket_offset_xy = 0.0
    cfg.scene.demo_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DemoCamera",
        update_period=0.0,
        width=args.width,
        height=args.height,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(pos=tuple(args.eye), rot=_look_at_xyzw(args.eye, args.lookat), convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=1.5, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)
        ),
    )
    env = PlaceEnv(cfg)
    agent_cfg = handle_deprecated_rsl_rl_cfg(PlacePPORunnerCfg(), metadata.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)
    camera = env.scene["demo_camera"]

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH (e.g. winget install Gyan.FFmpeg)")
    fps = round(1.0 / env.step_dt)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    def settle():
        # Let the renderer catch up (textures, lighting, post-reset poses) before recording.
        for _ in range(10):
            env.sim.render()
        camera.update(dt=0.0, force_recompute=True)

    def run_episode():
        """Record until the episode ends (success/failure end it inside step(); then the time limit)."""
        nonlocal obs
        frames, outcome = [], "timeout"
        successes_before = env.place_counts["success"]
        for _ in range(int(env.max_episode_length) + 5):
            frames.append(camera.data.output["rgb"][0, ..., :3].cpu().numpy().copy())
            with torch.inference_mode():
                obs, _, dones, _ = wrapped.step(policy(obs))
            if bool(dones[0]):
                if env.place_counts["success"] > successes_before:
                    outcome = "success"
                else:
                    log = env.extras.get("log", {})
                    failed = [k.split("/", 1)[1] for k, v in log.items() if k.startswith("Failure/") and v > 0]
                    outcome = failed[0] if failed else "timeout"
                break
        return frames, outcome

    def write_video(frames, path):
        # H.264 MP4 via ffmpeg (this OpenCV build has no video encoder).
        h, w = frames[0].shape[:2]
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
            stdin=subprocess.PIPE,
        )
        for frame in frames:
            proc.stdin.write(frame.astype("uint8").tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to write the video")
        cv2.imwrite(str(path.with_suffix(".png")), cv2.cvtColor(frames[min(len(frames) - 1, fps)], cv2.COLOR_RGB2BGR))

    obs = wrapped.get_observations()
    saved, attempt, outcomes = 0, 0, []
    while saved < args.episodes and attempt < args.max_attempts:
        attempt += 1
        settle()
        frames, outcome = run_episode()
        outcomes.append(outcome)
        keep = outcome == "success" or not args.only_success
        if keep:
            saved += 1
            path = out if args.episodes == 1 else out.with_name(f"{out.stem}_{saved:02d}{out.suffix}")
            write_video(frames, path)
            print(f"[record] attempt {attempt}: {outcome}, {len(frames) / fps:.1f} s -> {path}")
        else:
            print(f"[record] attempt {attempt}: {outcome}, {len(frames) / fps:.1f} s (not saved)")
    summary = {k: outcomes.count(k) for k in sorted(set(outcomes))}
    print(f"\n[record] saved {saved} video(s) from {attempt} attempt(s); outcomes: {summary}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
