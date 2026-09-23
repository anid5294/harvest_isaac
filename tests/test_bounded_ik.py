"""Numerical checks independent of Kit/GPU startup."""

import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "bounded_ik", Path(__file__).resolve().parents[1]
    / "src/vla_isaaclab/policies/bounded_ik.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bounded_dls = module.bounded_dls


class BoundedIKTests(unittest.TestCase):
    def test_blocked_joint_residual_is_transferred(self):
        jacobian = torch.tensor([[[1.0, 1.0]]])
        result = bounded_dls(jacobian, torch.tensor([[0.1]]),
                             torch.tensor([[0.0, -0.2]]),
                             torch.tensor([[0.0, 0.2]]), 0.01)
        self.assertEqual(result[0, 0].item(), 0.0)
        self.assertAlmostEqual(result[0, 1].item(), 0.1, places=4)

    def test_singular_batch_stays_finite_and_bounded(self):
        torch.manual_seed(42)
        jacobian = torch.randn(8, 6, 6)
        jacobian[0] = 0.0
        jacobian[1, :, 2] = jacobian[1, :, 1]
        lower = -torch.rand(8, 6) * 0.025
        upper = torch.rand(8, 6) * 0.025
        result = bounded_dls(jacobian, torch.randn(8, 6), lower, upper, 0.04)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue((result >= lower).all() and (result <= upper).all())
        self.assertTrue((result[0] == 0).all())


if __name__ == "__main__":
    unittest.main()
