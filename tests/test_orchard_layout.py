"""Simulator-free checks for procedural orchard preview geometry."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest


spec = importlib.util.spec_from_file_location(
    "orchard_layout",
    Path(__file__).resolve().parents[1]
    / "src/vla_isaaclab/envs/orchard_preview/layout.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
cylinder_between = module.cylinder_between
make_orchard_layout = module.make_orchard_layout


class OrchardLayoutTests(unittest.TestCase):
    def test_layout_is_seeded_and_target_is_fixed(self):
        first = make_orchard_layout(42)
        repeated = make_orchard_layout(42)
        alternate = make_orchard_layout(43)
        self.assertEqual(first, repeated)
        self.assertEqual(first.apples[0], alternate.apples[0])
        self.assertNotEqual(first.apples[1:], alternate.apples[1:])

    def test_branch_transform_spans_requested_endpoints(self):
        branch = cylinder_between((0.0, 0.0, 0.0), (0.3, -0.4, 1.2), 0.02)
        self.assertAlmostEqual(branch.length, 1.3)
        quaternion_norm = math.sqrt(
            sum(value * value for value in branch.orientation_wxyz)
        )
        self.assertAlmostEqual(quaternion_norm, 1.0)
        self.assertEqual(branch.center, (0.15, -0.2, 0.6))

    def test_reachable_apple_and_supported_tray_are_separated(self):
        target = make_orchard_layout(42).apples[0].position
        tray_center = (-0.47, -0.15, 0.77)
        self.assertLess(math.dist(target, (0.0, -0.78, 1.25)), 0.75)
        self.assertGreater(math.dist(target, tray_center), 0.35)


if __name__ == "__main__":
    unittest.main()
