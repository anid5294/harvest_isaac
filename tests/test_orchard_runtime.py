"""Regression checks for shutdown exit status and evidence-gated grasping."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load("scripts/run_checked.py")
contact = load("src/vla_isaaclab/envs/orchard_pick/contact.py")
progress = load("src/vla_isaaclab/envs/orchard_pick/progress.py")
summary = load("scripts/summarize_orchard_trace.py")


class RuntimeTests(unittest.TestCase):
    def test_shutdown_zero_does_not_erase_task_failure(self):
        self.assertEqual(runtime.resolve_exit(0, {"exit_code": 2}), 2)
        self.assertEqual(runtime.resolve_exit(0, {"exit_code": 0}), 0)
        self.assertEqual(runtime.resolve_exit(0, None), 1)
        self.assertEqual(runtime.resolve_exit(-11, {"exit_code": 0}), 139)
        self.assertEqual(runtime.resolve_exit(1, {"exit_code": 0}), 1)

    def test_touching_with_one_digit_is_not_a_grasp(self):
        thumb, finger = contact.FINGER_LINKS[2], contact.FINGER_LINKS[4]
        self.assertFalse(contact.opposing_contact({thumb: 1.0}))
        self.assertFalse(contact.opposing_contact({finger: 1.0}))
        self.assertTrue(contact.opposing_contact({thumb: 0.2, finger: 0.3}))

    def test_close_waits_for_contact_or_reports_specific_failure(self):
        controller = progress.Progress()
        controller.index = 3
        for _ in range(120):
            controller.update(reached=True, detached=False, carried=False, grasp_contact=False)
        self.assertEqual(controller.phase, "close")
        self.assertEqual(controller.failure, "grasp_contact_missing")
        controller = progress.Progress()
        controller.index = 3
        for _ in range(90):
            controller.update(reached=True, detached=False, carried=False, grasp_contact=True)
        self.assertEqual(controller.phase, "pull")

    def test_old_trace_does_not_imply_zero_measured_contact(self):
        result = summary.summarize([{"phase": "close", "hand_distance": 0.12}])
        self.assertFalse(result["close"]["contact_measurements_available"])
        self.assertEqual(result["close"]["hand_distance_min_m"], 0.12)


if __name__ == "__main__":
    unittest.main()
