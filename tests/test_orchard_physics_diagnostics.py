import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "orchard_physics_diagnostics",
    Path(__file__).resolve().parents[1]
    / "src/vla_isaaclab/envs/orchard_pick/diagnostics.py",
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class PhysicsDiagnosticsTests(unittest.TestCase):
    def test_buffers_are_bounded(self):
        buffer = diagnostics.PhysicsDiagnostics(capacity=2)
        for dt in (0.1, 0.2, 0.3):
            buffer.on_step(dt)
        self.assertEqual([row["step"] for row in buffer.snapshot()["steps"]], [2, 3])
        self.assertEqual(len(buffer.events), 0)

        for index in range(3):
            buffer.on_break(f"/Joint_{index}")
        self.assertEqual([row["joint_path"] for row in buffer.snapshot()["events"]],
                         ["/Joint_1", "/Joint_2"])

    def test_contacts_serialize_impulses_and_step_timestamp(self):
        buffer = diagnostics.PhysicsDiagnostics()
        buffer.on_contact(actor0="/A", actor1="/B", collider0="/A/Shape",
                           collider1="/B/Shape", event_type="found",
                           impulses_ns=[(1, 2.5, -3)])
        buffer.on_step(0.02)
        snapshot = buffer.snapshot()
        self.assertEqual(snapshot["timestamp_basis"],
                         "callback_observation_physics_step_count")
        self.assertEqual(snapshot["step_count"], 1)
        self.assertAlmostEqual(snapshot["observed_time_s"], 0.02)
        self.assertEqual(snapshot["events"][0]["observed_step"], 0)
        self.assertEqual(snapshot["events"][0]["observed_time_s"], 0.0)
        self.assertEqual(snapshot["steps"][0]["contacts"][0]["impulses_ns"],
                         [[1, 2.5, -3]])

    def test_unstepped_break_is_included_in_snapshot(self):
        buffer = diagnostics.PhysicsDiagnostics()
        buffer.on_break("/Orchard/Stem_0/FixedJoint")
        snapshot = buffer.snapshot()
        self.assertEqual(snapshot["pending_joint_breaks"],
                         [{"joint_path": "/Orchard/Stem_0/FixedJoint"}])
        self.assertEqual(snapshot["steps"], [])

    def test_callback_error_fails_snapshot(self):
        buffer = diagnostics.PhysicsDiagnostics()
        buffer.callback_error = "ValueError('bad native callback')"
        with self.assertRaisesRegex(RuntimeError, "bad native callback"):
            buffer.snapshot()


if __name__ == "__main__":
    unittest.main()
