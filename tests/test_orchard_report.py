import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "orchard_check", Path(__file__).resolve().parents[1] / "scripts/check_orchard_run.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReportTests(unittest.TestCase):
    def report(self):
        return {"task": "VLA-OrchardPick-G1-JointPos-v0", "action_dimension": 43,
                "passed": True, "success_count": 1, "termination_terms": {"success": True},
                "orchard": {"metrics": {"success": True, "failure": None}}}

    def test_named_success_is_required_even_when_report_claims_pass(self):
        report = self.report()
        self.assertEqual(module.check(report), [])
        report["termination_terms"]["success"] = False
        self.assertTrue(module.check(report))

    def test_passive_hold_is_not_harvest_success(self):
        report = self.report()
        report.update(passed=False, success_count=0, termination_terms={"success": False})
        report["orchard"] = {
            "layout": {"apples": [[0, 0, 1]], "target_index": 0},
            "metrics": {"position": [0, 0, 1], "speed": 0, "detached": False},
        }
        self.assertEqual(module.check(report, passive=True), [])
        self.assertTrue(module.check(report))
        report["orchard"]["metrics"]["position"][2] = 0.5
        self.assertTrue(module.check(report, passive=True))

    def test_commercial_passive_checks_every_fruit(self):
        report = self.report()
        report["orchard"] = {
            "layout": {"tree_model": "commercial", "apples": [[0, 0, 1], [1, 0, 2]], "target_index": 0},
            "metrics": {"position": [0, 0, 1], "speed": 0, "detached": False, "detached_indices": []},
        }
        report["rigid_objects"] = {
            "object": {"position_m": [0, 0, 1], "finite": True, "stable": True},
            "apple_1": {"position_m": [1, 0, 2], "finite": True, "stable": True},
        }
        self.assertEqual(module.check(report, passive=True), [])
        report["rigid_objects"]["apple_1"]["position_m"][2] = 1.9
        self.assertTrue(module.check(report, passive=True))
