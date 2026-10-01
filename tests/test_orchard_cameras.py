"""Simulator-free checks for the orchard three-camera contract mapping."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT / "src/vla_isaaclab/recording/frame.py"
ORCHARD_CFG = ROOT / "src/vla_isaaclab/envs/orchard_pick/env_cfg.py"
G1_CFG = ROOT / "src/vla_isaaclab/envs/common/g1.py"


def pure_frame_parts():
    """Load only camera code; Isaac Lab is unavailable in the local test env."""
    tree = ast.parse(FRAME.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in (
                "ORCHARD_THREE_VIEW_PROFILE",
                "ORCHARD_COMMERCIAL_THREE_VIEW_PROFILE",
                "ORCHARD_THREE_VIEW_PROFILES",
            )
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "camera_features_for_profile":
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "EnvironmentFrameAdapter":
            method = next(
                item for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "camera_metadata"
            )
            method.decorator_list = []
            nodes.append(method)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(FRAME), "exec"), namespace)
    return namespace


def camera():
    offset = SimpleNamespace(pos=(0.9, 3.6, 2.7), rot=(1.0, 0.0, 0.0, 0.0),
                             convention="opengl")
    spawn = SimpleNamespace(focal_length=28.0, horizontal_aperture=20.955,
                            focus_distance=2.0, clipping_range=(0.05, 10.0))
    return SimpleNamespace(cfg=SimpleNamespace(offset=offset, spawn=spawn,
                                                width=640, height=480))


class OrchardCameraTests(unittest.TestCase):
    def test_external_view_is_in_front_of_rotated_robot(self):
        orchard = ast.parse(ORCHARD_CFG.read_text(encoding="utf-8"))
        eye = ast.literal_eval(next(node.value for node in orchard.body
                                    if isinstance(node, ast.Assign)
                                    and any(isinstance(target, ast.Name) and target.id == "EYE"
                                            for target in node.targets)))
        scene = next(node for node in orchard.body
                     if isinstance(node, ast.ClassDef) and node.name == "OrchardSceneCfg")
        robot = next(node.value for node in scene.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "robot"
                             for target in node.targets))
        robot_position = ast.literal_eval(robot.args[0])
        g1 = ast.parse(G1_CFG.read_text(encoding="utf-8"))
        make_g1 = next(node for node in g1.body
                       if isinstance(node, ast.FunctionDef) and node.name == "make_g1_cfg")
        w, x, y, z = ast.literal_eval(make_g1.args.defaults[0])
        # Quaternion rotation of robot-local +X, the G1 forward direction.
        forward = (1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 2 * (x*z - w*y))
        camera_direction = tuple(eye[i] - robot_position[i] for i in range(3))
        self.assertGreater(sum(forward[i] * camera_direction[i] for i in range(3)), 0)

    def test_active_scene_defines_exactly_three_views(self):
        tree = ast.parse(ORCHARD_CFG.read_text(encoding="utf-8"))
        profile_assignment = next(
            node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "CAMERA_PROFILE"
                    for target in node.targets)
        )
        self.assertEqual(ast.literal_eval(profile_assignment.value),
                         pure_frame_parts()["ORCHARD_THREE_VIEW_PROFILE"])
        commercial_assignment = next(
            node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "COMMERCIAL_CAMERA_PROFILE"
                    for target in node.targets)
        )
        self.assertEqual(ast.literal_eval(commercial_assignment.value),
                         pure_frame_parts()["ORCHARD_COMMERCIAL_THREE_VIEW_PROFILE"])
        scene = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "OrchardSceneCfg")
        sensors = {
            target.id: value.func.id
            for assignment in scene.body if isinstance(assignment, ast.Assign)
            for target in assignment.targets if isinstance(target, ast.Name)
            for value in [assignment.value]
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and target.id.startswith("cam_")
        }
        self.assertEqual(sensors, {
            "cam_left_high": "camera_cfg",
            "cam_left_wrist": "g1_left_wrist_camera_cfg",
            "cam_right_wrist": "robot_rgb_camera_cfg",
        })

    def test_contract_keys_and_legacy_profile(self):
        functions = pure_frame_parts()
        features = functions["camera_features_for_profile"]
        profile = functions["ORCHARD_THREE_VIEW_PROFILE"]
        self.assertEqual(features(profile), (
            ("cam_left_high", "observation.images.cam_left_high"),
            ("cam_left_wrist", "observation.images.cam_left_wrist"),
            ("cam_right_wrist", "observation.images.cam_right_wrist"),
        ))
        self.assertEqual(
            features(functions["ORCHARD_COMMERCIAL_THREE_VIEW_PROFILE"]),
            features(profile),
        )
        self.assertIn(("cam_side", "observation.images.cam_right_high"),
                      features("legacy_orchard"))
        self.assertIn(("cam_side", "observation.images.cam_side"), features(None))

    def test_external_mount_is_explicit_and_legacy_head_stays_head(self):
        functions = pure_frame_parts()
        metadata = functions["camera_metadata"]
        external = camera()
        orchard = SimpleNamespace(camera_profile=functions["ORCHARD_THREE_VIEW_PROFILE"],
                                  cameras=[("cam_left_high", "observation.images.cam_left_high",
                                            external)])
        entry = metadata(orchard)[0]
        self.assertEqual(entry["parent_link"], "world")
        self.assertEqual(entry["mount_role"], "fixed_external_front_top")
        self.assertEqual(entry["camera_profile"], orchard.camera_profile)
        self.assertEqual(entry["resolution_hw"], [480, 640])
        orchard.camera_profile = functions["ORCHARD_COMMERCIAL_THREE_VIEW_PROFILE"]
        commercial_entry = metadata(orchard)[0]
        self.assertEqual(commercial_entry["parent_link"], "world")
        self.assertEqual(commercial_entry["mount_role"], "fixed_external_front_top")
        self.assertEqual(commercial_entry["camera_profile"], orchard.camera_profile)
        legacy = SimpleNamespace(camera_profile=None, cameras=orchard.cameras)
        legacy_entry = metadata(legacy)[0]
        self.assertEqual(legacy_entry["parent_link"], "head_link")
        self.assertNotIn("camera_profile", legacy_entry)

    def test_preview_and_video_checker_accept_both_three_view_profiles(self):
        def load(path):
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        preview = load(ROOT / "scripts/preview_views.py")
        checker = load(ROOT / "scripts/check_orchard_run.py")
        profiles = pure_frame_parts()
        expected = (("cam_left_high", "external"),
                    ("cam_left_wrist", "left_wrist"),
                    ("cam_right_wrist", "right_wrist"))
        for name in ("ORCHARD_THREE_VIEW_PROFILE", "ORCHARD_COMMERCIAL_THREE_VIEW_PROFILE"):
            profile = profiles[name]
            self.assertEqual(preview.camera_views(profile), expected)
            calls = []

            def probe(args, **kwargs):
                calls.append(args[-1])
                return SimpleNamespace(stdout='{"streams":[{"width":640,"height":480,'
                                              '"r_frame_rate":"30/1","nb_read_frames":"3"}]}')

            with patch.object(checker.subprocess, "run", side_effect=probe):
                self.assertEqual(checker.check_videos(Path("videos"), 3, profile), [])
            self.assertEqual([Path(path).name for path in calls],
                             ["external.mp4", "left_wrist.mp4", "right_wrist.mp4"])


if __name__ == "__main__":
    unittest.main()
