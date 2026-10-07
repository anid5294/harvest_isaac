import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


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

    def test_install_uses_native_threshold_attr_and_retains_subscriptions(self):
        threshold_values = []

        class Prim:
            def IsValid(self):
                return True

        class Stage:
            def GetPrimAtPath(self, path):
                return Prim()

        class ContactReportAPI:
            def CreateThresholdAttr(self):
                return SimpleNamespace(Set=threshold_values.append)

        contact_subscription = object()
        step_subscription = object()
        simulation_interface = SimpleNamespace(
            subscribe_contact_report_events=lambda callback: contact_subscription)
        physics_interface = SimpleNamespace(
            subscribe_physics_step_events=lambda callback: step_subscription)

        omni = ModuleType("omni")
        omni.__path__ = []
        omni.physx = SimpleNamespace(
            get_physx_simulation_interface=lambda: simulation_interface,
            get_physx_interface=lambda: physics_interface)
        omni.usd = SimpleNamespace(
            get_context=lambda: SimpleNamespace(get_stage=lambda: Stage()))
        pxr = ModuleType("pxr")
        pxr.PhysicsSchemaTools = SimpleNamespace(intToSdfPath=str)
        pxr.PhysxSchema = SimpleNamespace(
            PhysxContactReportAPI=SimpleNamespace(Apply=lambda prim: ContactReportAPI()))
        pxr.Usd = SimpleNamespace(PrimRange=lambda prim: ())
        pxr.UsdPhysics = SimpleNamespace(RigidBodyAPI=object())
        env = SimpleNamespace(
            scene=SimpleNamespace(env_prim_paths=["/World/envs/env_0"]),
            cfg=SimpleNamespace(orchard_layout=SimpleNamespace(apples=[object()])))

        with patch.dict(sys.modules, {"omni": omni, "omni.physx": omni.physx,
                                     "omni.usd": omni.usd, "pxr": pxr}):
            diagnostics.install(env)

        self.assertEqual(threshold_values, [0.0, 0.0])
        self.assertIs(env._orchard_contact_report_subscription, contact_subscription)
        self.assertIs(env._orchard_physics_step_subscription, step_subscription)


if __name__ == "__main__":
    unittest.main()
