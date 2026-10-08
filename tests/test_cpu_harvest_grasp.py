"""Pure CPU checks for harvest hand targets and grasp geometry."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest


PATH = (Path(__file__).resolve().parents[1] / "src/vla_isaaclab/policies"
        / "cpu_harvest_grasp.py")
spec = importlib.util.spec_from_file_location("cpu_harvest_grasp", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class CPUHarvestGraspTests(unittest.TestCase):
    def test_quaternion_rotation_identity_and_quarter_turn(self):
        vector = (1.0, 2.0, -3.0)
        self.assertTupleAlmostEqual(module.rotate_vector((1.0, 0.0, 0.0, 0.0), vector), vector)

        half = math.sqrt(0.5)
        rotated = module.rotate_vector((half, 0.0, 0.0, half), (1.0, 0.0, 0.0))
        self.assertTupleAlmostEqual(rotated, (0.0, 1.0, 0.0))

    def test_scaled_quaternion_gives_same_rotation(self):
        vector = (0.1, -0.05, 0.0)
        quaternion = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
        scaled = tuple(4.0 * component for component in quaternion)
        self.assertTupleAlmostEqual(module.rotate_vector(scaled, vector),
                                    module.rotate_vector(quaternion, vector))

    def test_nonfinite_quaternions_are_rejected(self):
        for quaternion in ((math.nan, 0.0, 0.0, 0.0),
                           (math.inf, 0.0, 0.0, 0.0)):
            with self.subTest(quaternion=quaternion):
                with self.assertRaises(ValueError):
                    module.rotate_vector(quaternion, (1.0, 0.0, 0.0))

    def test_grasp_offset_reconstructs_profile_apple_center(self):
        quaternion = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
        center_in_palm = module.APPLE_CENTER_IN_PALM_M
        center_world_offset = module.rotate_vector(quaternion, center_in_palm)
        grasp_offset = module.grasp_offset_world(quaternion)
        self.assertTupleAlmostEqual(tuple(a + b for a, b in zip(center_world_offset, grasp_offset)),
                                    (0.0, 0.0, 0.0))

    def test_pregrasp_withdraws_along_local_negative_x(self):
        half = math.sqrt(0.5)
        quaternion = (half, 0.0, 0.0, half)
        offset_world = module.pregrasp_offset_world(quaternion)
        local_offset = module.rotate_vector((half, 0.0, 0.0, -half), offset_world)
        self.assertTupleAlmostEqual(local_offset, (-module.ENTRY_DISTANCE_M, 0.0, 0.0))
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in offset_world)),
                               module.ENTRY_DISTANCE_M)

    def test_signed_hand_targets_fit_usd_hard_and_soft_limits(self):
        hard_limits_deg = {
            "left_hand_thumb_0_joint": (-60.0, 60.0),
            "left_hand_thumb_1_joint": (-35.0, 60.0),
            "left_hand_thumb_2_joint": (0.0, 100.0),
            "left_hand_middle_0_joint": (-90.0, 0.0),
            "left_hand_middle_1_joint": (-100.0, 0.0),
            "left_hand_index_0_joint": (-90.0, 0.0),
            "left_hand_index_1_joint": (-100.0, 0.0),
        }
        self.assertEqual(set(module.CLOSED_HAND_RAD), set(hard_limits_deg))
        for name, target in module.CLOSED_HAND_RAD.items():
            hard_lower, hard_upper = (math.radians(value) for value in hard_limits_deg[name])
            midpoint = 0.5 * (hard_lower + hard_upper)
            soft_half_range = 0.45 * (hard_upper - hard_lower)
            soft_lower, soft_upper = midpoint - soft_half_range, midpoint + soft_half_range
            with self.subTest(joint=name):
                self.assertGreaterEqual(target, hard_lower)
                self.assertLessEqual(target, hard_upper)
                self.assertGreaterEqual(target, soft_lower)
                self.assertLessEqual(target, soft_upper)

        # The former zero-valued middle/index targets sat at the open endpoint.
        # These signed profile values must survive both limit clamps as closure.
        for name in ("left_hand_middle_0_joint", "left_hand_middle_1_joint",
                     "left_hand_index_0_joint", "left_hand_index_1_joint"):
            hard_lower, hard_upper = (math.radians(value) for value in hard_limits_deg[name])
            midpoint = 0.5 * (hard_lower + hard_upper)
            soft_half_range = 0.45 * (hard_upper - hard_lower)
            soft_lower, soft_upper = midpoint - soft_half_range, midpoint + soft_half_range
            clamped_open = min(max(0.0, hard_lower), hard_upper)
            clamped_open = min(max(clamped_open, soft_lower), soft_upper)
            clamped_closed = min(max(module.CLOSED_HAND_RAD[name], hard_lower), hard_upper)
            clamped_closed = min(max(clamped_closed, soft_lower), soft_upper)
            inherited_closed = min(max(1.2, hard_lower), hard_upper)
            inherited_closed = min(max(inherited_closed, soft_lower), soft_upper)
            with self.subTest(joint=name):
                self.assertAlmostEqual(clamped_open, soft_upper, places=7)
                self.assertAlmostEqual(inherited_closed, clamped_open, places=7)
                self.assertLess(clamped_closed, clamped_open)

    def assertTupleAlmostEqual(self, actual, expected, places=7):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)


if __name__ == "__main__":
    unittest.main()
