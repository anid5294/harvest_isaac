import importlib.util
from pathlib import Path
import sys
import unittest

scripts = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(scripts))
spec=importlib.util.spec_from_file_location("free_demo_checker", scripts/"check_free_pick_place_run.py")
checker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class FreeDemoReportTests(unittest.TestCase):
    def fixture(self):
        return {"task":"VLA-FreeApplePickPlace-G1-JointPos-v0", "action_dimension":43,
                "passed":True, "success_count":1, "termination_terms":{"success":True},
                "free_pick_place":{"physics_device":"cpu", "native_stem_present":False,
                  "harvest_success_claim":False,"metrics":dict.fromkeys(
                      ("finite","robot_stable","grasp_confirmed","lifted","carried","released","supported","success"),True)}}

    def test_complete_evidence(self):
        self.assertEqual(checker.check(self.fixture()),[])

    def test_missing_any_gate_rejected(self):
        for key in self.fixture()["free_pick_place"]["metrics"]:
            report=self.fixture()
            report["free_pick_place"]["metrics"][key]=False
            self.assertTrue(checker.check(report),key)

    def test_pose_or_policy_claim_not_success(self):
        report=self.fixture()
        report["termination_terms"]["success"]=False
        self.assertTrue(checker.check(report))
        report=self.fixture()
        report["free_pick_place"]["harvest_success_claim"]=True
        self.assertTrue(checker.check(report))


if __name__ == "__main__": unittest.main()
