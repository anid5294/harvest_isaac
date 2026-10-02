"""Small H.264 inspection videos, separate from the AV1 training export."""

from pathlib import Path


def camera_views(profile=None):
    if profile in ("orchard_fixed_front_top_three_view_v1",
                   "orchardbench_single_tree_three_view_v1",
                   "free_pick_place_three_view_v1",
                   "orchard_commercial_full_tree_three_view_v1"):
        return (("cam_left_high", "external"), ("cam_left_wrist", "left_wrist"),
                ("cam_right_wrist", "right_wrist"))
    return (("cam_side", "external"), ("cam_left_high", "head"),
            ("cam_left_wrist", "left_wrist"), ("cam_right_wrist", "right_wrist"))


class PreviewViews:
    def __init__(self, env, directory):
        import av

        self.outputs = []
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        try:
            for key, filename in camera_views(getattr(env.cfg, "camera_profile", None)):
                if key not in env.scene.sensors:
                    raise ValueError(f"Camera inspection profile requires {key}")
                sensor = env.scene.sensors[key]
                container = av.open(str(directory / f"{filename}.mp4"), mode="w")
                stream = container.add_stream("libx264", rate=30)
                stream.width = sensor.cfg.width
                stream.height = sensor.cfg.height
                stream.pix_fmt = "yuv420p"
                self.outputs.append((sensor, container, stream))
        except BaseException:
            self.close()
            raise

    def write(self):
        import av

        for sensor, container, stream in self.outputs:
            rgb = sensor.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)

    def close(self):
        for _, container, stream in self.outputs:
            try:
                for packet in stream.encode():
                    container.mux(packet)
            finally:
                container.close()
        self.outputs = []
