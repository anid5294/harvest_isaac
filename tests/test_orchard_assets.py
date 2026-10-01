"""Simulator-free validation of tree provenance and video profiles."""

import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assets = load("tree_asset_validation", "src/vla_isaaclab/envs/orchard_pick/tree_asset.py")
views = load("orchard_preview_views", "scripts/preview_views.py")
checker = load("orchard_video_check", "scripts/check_orchard_run.py")


class AssetTests(unittest.TestCase):
    def test_layout_and_bytes_must_match(self):
        layout = SimpleNamespace(metadata=lambda: {"seed": 42, "apples": ((0, 0, 1),)})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "tree.usda"
            tree.write_text('#usda 1.0\n')
            manifest = {"layout": layout.metadata(), "tree_file": "tree.usda",
                        "sha256": hashlib.sha256(tree.read_bytes()).hexdigest(),
                        "collision_status": "visual_only_existing_scaffold"}
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(assets.validate_tree_asset(root, layout)[0], tree.resolve())
            wrong_layout = SimpleNamespace(metadata=lambda: {"seed": 43})
            with self.assertRaisesRegex(ValueError, "layout differs"):
                assets.validate_tree_asset(root, wrong_layout)
            tree.write_text('#usda 1.0\n# modified\n')
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                assets.validate_tree_asset(root, layout)

    def test_three_view_inspection_and_legacy_four_views(self):
        profile = "orchard_fixed_front_top_three_view_v1"
        self.assertEqual([name for _, name in views.camera_views(profile)],
                         ["external", "left_wrist", "right_wrist"])
        self.assertEqual(len(views.camera_views()), 4)
        result = SimpleNamespace(stdout=json.dumps({"streams": [{
            "width": 640, "height": 480, "r_frame_rate": "30/1", "nb_read_frames": "120",
        }]}))
        with patch.object(checker.subprocess, "run", return_value=result) as probe:
            self.assertEqual(checker.check_videos(Path("unused"), 120, profile), [])
            self.assertEqual(probe.call_count, 3)
        with patch.object(checker.subprocess, "run", return_value=result) as probe:
            self.assertEqual(checker.check_videos(Path("unused"), 120), [])
            self.assertEqual(probe.call_count, 4)


if __name__ == "__main__":
    unittest.main()
