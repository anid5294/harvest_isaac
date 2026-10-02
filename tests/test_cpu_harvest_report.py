import importlib.util
from pathlib import Path
import sys
import unittest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("cpu_harvest_checker", scripts / "check_cpu_harvest_run.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class CpuHarvestReportTests(unittest.TestCase):
    def report(self):
        return {"task": "VLA-OrchardPick-G1-JointPos-v0", "action_dimension": 43,
                "passed": True, "success_count": 1, "termination_terms": {"success": True},
                "orchard": {"physics_device": "cpu", "metrics": {"success": True, "failure": None}}}

    def trajectory(self):
        return [
            {"step": 1, "phase": "close", "detached": False, "grasp_contact": True,
             "carried": False, "released": False, "position": [0, 0, 1]},
            {"step": 2, "phase": "pull", "detached": True, "grasp_contact": True,
             "carried": False, "released": False, "position": [0, 0, 1]},
            {"step": 3, "phase": "transfer", "detached": True, "grasp_contact": True,
             "carried": True, "released": False, "position": [0.11, 0, 1]},
            {"step": 4, "phase": "release", "detached": True, "grasp_contact": False,
             "carried": True, "released": True, "supported": True, "opened": True,
             "stable_steps": 30, "failure": None, "success": True, "position": [0.11, 0, 1]},
        ]

    def test_complete_measured_sequence(self):
        errors, summary = checker.check(self.report(), self.trajectory())
        self.assertEqual(errors, [])
        self.assertEqual(summary["break"]["step"], 2)
        self.assertEqual(summary["carry"]["horizontal_displacement_m"], 0.11)

    def test_policy_or_report_success_cannot_replace_missing_physics(self):
        rows = self.trajectory()
        rows[2]["grasp_contact"] = False
        errors, _ = checker.check(self.report(), rows)
        self.assertTrue(any("carry sample" in error for error in errors))
        self.assertTrue(checker.check(self.report(), [])[0])

    def test_requires_native_attached_state_and_cpu(self):
        report = self.report()
        report["orchard"]["physics_device"] = "cuda:0"
        rows = self.trajectory()
        rows[0]["detached"] = True
        errors, _ = checker.check(report, rows)
        self.assertTrue(any("CPU" in error for error in errors))
        self.assertTrue(any("attached native-stem" in error for error in errors))

    def test_contact_on_attached_sample_and_named_success(self):
        rows = self.trajectory()
        errors, _ = checker.check(self.report(), rows)
        self.assertEqual(errors, [])
        report = self.report()
        report["termination_terms"]["success"] = False
        self.assertTrue(checker.check(report, self.trajectory())[0])

    def test_stale_and_postbreak_only_contact_rejected(self):
        rows = self.trajectory()
        rows[1]["detached"] = False
        rows.insert(1, {"step": 2, "phase": "pull", "detached": False,
                        "grasp_contact": False, "carried": False, "released": False,
                        "position": [0, 0, 1]})
        rows.insert(2, {"step": 3, "phase": "pull", "detached": False,
                        "grasp_contact": False, "carried": False, "released": False,
                        "position": [0, 0, 1]})
        rows[0]["grasp_contact"] = True  # stale by more than two control steps
        rows[3]["detached"] = True
        rows[3]["phase"] = "pull"
        rows[3]["position"] = [0, 0, 1]
        rows[4]["position"] = [0.11, 0, 1]
        rows[-1].update(supported=True, opened=True, stable_steps=30,
                        failure=None, success=True)
        errors, _ = checker.check(self.report(), rows)
        self.assertTrue(any("attached sample within 2 steps" in error for error in errors))

        rows = self.trajectory()
        rows[0]["grasp_contact"] = False
        rows[1]["grasp_contact"] = False
        errors, _ = checker.check(self.report(), rows)
        self.assertTrue(any("attached sample within 2 steps" in error for error in errors))

    def test_release_must_follow_carry_and_final_trace_must_show_success(self):
        rows = self.trajectory()
        rows[2]["released"] = True
        rows[3]["released"] = False
        errors, _ = checker.check(self.report(), rows)
        self.assertTrue(any("release not observed after qualifying carry" in error for error in errors))
        self.assertTrue(any("Final trajectory sample lacks measured release" in error for error in errors))

    def test_malformed_rows_rejected_without_exception(self):
        errors, _ = checker.check(self.report(), [None])
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
