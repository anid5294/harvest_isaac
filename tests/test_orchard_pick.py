"""Seed, geometry, controller gate, and terminal-state checks without Isaac Sim."""

import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "src/vla_isaaclab/envs/orchard_pick"


def load(name):
    spec = importlib.util.spec_from_file_location(f"orchard_test_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


layout = load("layout")
Progress = load("progress").Progress


class OrchardPickTests(unittest.TestCase):
    def test_stem_frames_match_vertical_and_horizontal_attachments(self):
        for anchor in ((0.0, 0.0, 1.10), (0.10, 0.0, 1.0), (0.0, 0.0, 0.90)):
            scene = layout.OrchardLayout(42, ((0.0, 0.0, 1.0),), 0, (0, 0, 0),
                                        stem_anchors=(anchor,))
            geometry = layout.stem_geometry(scene, 0)
            self.assertAlmostEqual(math.dist(geometry["bottom"], scene.target), scene.radius)
            self.assertAlmostEqual(math.dist(geometry["bottom"], anchor), 0.1-scene.radius)
            self.assertAlmostEqual(sum(v*v for v in geometry["fruit_joint_rotation"]), 1.0)
            self.assertAlmostEqual(geometry["stem_joint_position"][2], -(0.1-scene.radius)/2)

    def test_short_stem_attachment_rejected(self):
        scene = layout.OrchardLayout(42, ((0, 0, 1),), 0, (0, 0, 0),
                                    stem_anchors=((0, 0, 1.001),))
        with self.assertRaises(ValueError):
            layout.stem_geometry(scene, 0)

    def test_seed_randomizes_target_and_basket_reproducibly(self):
        self.assertEqual(layout.make_layout(42), layout.make_layout(42))
        self.assertNotEqual(layout.make_layout(42).target, layout.make_layout(43).target)
        self.assertNotEqual(layout.make_layout(42).basket, layout.make_layout(43).basket)

    def test_seed_sweep_has_clear_supported_workspace(self):
        for seed in range(500):
            scene = layout.make_layout(seed)
            self.assertEqual(scene.target_index, 0)
            self.assertGreater(scene.stem_break_force, scene.mass * 9.81 * 3)
            self.assertLess(math.dist(scene.target, (-0.10, -0.30, 1.0)), 0.06)
            self.assertGreater(math.dist(scene.target, scene.basket), 0.25)
            for i, apple in enumerate(scene.apples):
                for other in scene.apples[i+1:]:
                    self.assertGreater(math.dist(apple, other), 2*scene.radius)

    def test_target_selection_rejects_impossible_scene(self):
        with self.assertRaises(ValueError):
            layout.choose_target([(2, 2, 3)])
        self.assertEqual(layout.choose_target([(2, 2, 3), (-0.1, -0.3, 1)]), 1)

    def test_placement_requires_entire_fruit_floor_height_and_low_speed(self):
        scene = layout.make_layout()
        x, y, z = scene.basket
        center = (x, y, z+0.01+scene.radius)
        self.assertTrue(layout.settled_in_basket(center, 0.01, scene.basket))
        for point, speed in [((x+0.14, y, center[2]), 0),
                             ((x, y, center[2]+0.05), 0),
                             (center, 0.2), ((float("nan"), y, z), 0)]:
            self.assertFalse(layout.settled_in_basket(point, speed, scene.basket))

    def test_pull_cannot_advance_without_detachment_and_carry(self):
        progress = Progress()
        progress.index = 4
        for _ in range(299):
            progress.update(reached=True, detached=False, carried=False)
        self.assertEqual(progress.phase, "pull")
        progress.update(reached=True, detached=False, carried=False)
        self.assertEqual(progress.failure, "pull_timeout")
        progress = Progress()
        progress.index = 4
        for _ in range(45):
            progress.update(reached=True, detached=True, carried=False)
        self.assertEqual(progress.phase, "pull")
        progress.update(reached=True, detached=True, carried=True)
        self.assertEqual(progress.phase, "transfer")

    def test_hold_never_manufactures_success(self):
        progress = Progress()
        progress.index = 9
        for _ in range(300):
            progress.update(reached=True, detached=True, carried=True)
        self.assertEqual(progress.phase, "hold")
        self.assertEqual(progress.failure, "hold_timeout")

    def test_environment_preserves_terminal_scene_until_explicit_reset(self):
        class FakeBase:
            def __init__(self, cfg, **kwargs):
                self.reset_count = 0
            def _reset_idx(self, ids):
                self.reset_count += 1
            def step(self, action):
                self._reset_idx([0])
                return None, None, [True], [False], {}

        fake = ModuleType("isaaclab.envs")
        fake.ManagerBasedRLEnv = FakeBase
        with patch.dict(sys.modules, {"isaaclab.envs": fake}):
            cls = load("env").OrchardPickEnv
        env = cls(SimpleNamespace(scene=SimpleNamespace(num_envs=1)))
        env._reset_idx([0])
        env.step(None)
        self.assertEqual(env.reset_count, 1)
        with self.assertRaises(RuntimeError):
            env.step(None)
        env._reset_idx([0])
        self.assertEqual(env.reset_count, 2)
        env.step(None)


if __name__ == "__main__":
    unittest.main()
