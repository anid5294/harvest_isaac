import importlib.util
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location("harvest_evidence",
    Path(__file__).resolve().parents[1]/"src/vla_isaaclab/envs/orchard_pick/interaction.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class EvidenceTests(unittest.TestCase):
    def test_physical_sequence_needs_no_script_flags(self):
        state = module.HarvestEvidence()
        state.update(detached=False, sustained_contact=True, lift_candidate=False)
        for _ in range(10):
            state.update(detached=True, sustained_contact=True, lift_candidate=True)
        self.assertTrue(state.carried)
        self.assertIsNone(state.failure)
        self.assertTrue(state.released(opened=True, contact=False, hand_distance=.25))
        self.assertFalse(state.released(opened=True, contact=True, hand_distance=.25))
        self.assertFalse(state.released(opened=False, contact=False, hand_distance=.25))

    def test_spontaneous_break_is_failure(self):
        state = module.HarvestEvidence()
        for _ in range(20):
            state.update(detached=True, sustained_contact=False, lift_candidate=True)
        self.assertEqual(state.failure, "detached_without_sustained_grasp")
        self.assertFalse(state.carried)

    def test_proximity_without_grasp_does_not_establish_carry(self):
        state = module.HarvestEvidence()
        state.update(detached=False, sustained_contact=True, lift_candidate=False)
        for _ in range(20):
            state.update(detached=True, sustained_contact=False, lift_candidate=True)
        self.assertFalse(state.carried)


if __name__ == "__main__":
    unittest.main()
