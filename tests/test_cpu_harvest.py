"""Pure CPU harvest state tests; runnable without Isaac Lab or torch."""

import importlib.util
from pathlib import Path
import sys
import unittest


PATH = (Path(__file__).resolve().parents[1] / "src/vla_isaaclab/policies"
        / "cpu_harvest_fsm.py")
spec = importlib.util.spec_from_file_location("cpu_harvest_fsm", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
HarvestFSM = module.HarvestFSM


class HarvestFSMTests(unittest.TestCase):
    def test_break_requires_preceding_measured_grasp_and_stops_pull(self):
        fsm = HarvestFSM()
        fsm.phase = "close"
        for step in range(1, 31):
            fsm.update(step=step, metrics={"grasp_contact": True}, ready=True)
        self.assertEqual(fsm.phase, "verify_grasp")
        self.assertEqual(fsm.grasp_step, 30)
        for step in range(31, 41):
            fsm.update(step=step, metrics={"grasp_contact": True}, ready=True)
        self.assertEqual(fsm.phase, "pull")
        fsm.update(step=41, metrics={"grasp_contact": True, "detached": True}, ready=False)
        self.assertEqual(fsm.phase, "lift")
        self.assertEqual(fsm.break_step, 41)
        self.assertLess(fsm.grasp_step, fsm.break_step)

    def test_premature_break_and_contact_loss_fail(self):
        premature = HarvestFSM()
        premature.update(step=1, metrics={"detached": True}, ready=False)
        self.assertEqual(premature.failure, "break_before_measured_grasp")
        early_after_grasp = HarvestFSM()
        early_after_grasp.phase = "verify_grasp"
        early_after_grasp.grasp_step = 10
        early_after_grasp.update(step=11, metrics={"detached": True,
                                                   "grasp_contact": True}, ready=True)
        self.assertEqual(early_after_grasp.failure, "break_before_pull")
        self.assertEqual(early_after_grasp.phase, "failed")
        lost = HarvestFSM()
        lost.phase = "pull"
        lost.grasp_step = 10
        for step in range(16):
            lost.update(step=11 + step, metrics={}, ready=False)
        self.assertEqual(lost.failure, "grasp_contact_lost")

    def test_pull_times_out_without_native_break(self):
        fsm = HarvestFSM()
        fsm.phase = "pull"
        fsm.grasp_step = 10
        for step in range(301):
            fsm.update(step=11 + step, metrics={"grasp_contact": True}, ready=False)
        self.assertEqual(fsm.failure, "pull_timeout")

    def test_placement_only_from_measured_success(self):
        fsm = HarvestFSM()
        fsm.phase = "verify_placement"
        fsm.grasp_step = 10
        fsm.break_step = 50
        fsm.update(step=100, metrics={"detached": True}, ready=True)
        self.assertEqual(fsm.phase, "verify_placement")
        fsm.update(step=101, metrics={"detached": True, "success": True}, ready=True)
        self.assertEqual(fsm.phase, "done")


if __name__ == "__main__":
    unittest.main()
