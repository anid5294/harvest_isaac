"""Lit-cube camera test: no humanoid, orchard, task managers, or contact joints."""

import json


def render_smoke(directory, device):
    import av
    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    directory.mkdir(parents=True, exist_ok=False)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/30, device=device))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=1500)
    light.func("/World/Light", light)
    cube = sim_utils.CuboidCfg(size=(0.4, 0.4, 0.4),
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.05)))
    cube.func("/World/Cube", cube, translation=(0, 0, 0.2))
    camera = Camera(CameraCfg(prim_path="/World/Camera", width=640, height=480,
                              data_types=["rgb"], update_period=0,
                              spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.05, 10))))
    sim.reset()
    import torch
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[1.5, -1.5, 1.2]], device=device),
        targets=torch.tensor([[0.0, 0.0, 0.2]], device=device))
    frames = 0
    minimum_std = float("inf")
    with av.open(str(directory / "smoke.mp4"), mode="w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
        for step in range(80):
            sim.step()
            camera.update(1/30)
            rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
            if step < 20:
                continue
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise RuntimeError(f"Invalid RGB output: {rgb.shape}, {rgb.dtype}")
            minimum_std = min(minimum_std, float(rgb.std()))
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
            frames += 1
        for packet in stream.encode():
            container.mux(packet)
    report = {"passed": frames == 60 and minimum_std > 2, "frames": frames,
              "minimum_pixel_std": minimum_std, "video": str(directory / "smoke.mp4")}
    (directory / "render_result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 2
