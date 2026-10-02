"""Static contract checks for the standalone stem diagnostic (no Isaac Sim required)."""

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/debug/minimal_stem_physx.py"
SPEC = importlib.util.spec_from_file_location("minimal_stem_physx", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_seed42_geometry_matches_selected_asset():
    layout = PROBE.load_seed42_layout()
    assert layout["radius"] == 0.03
    assert layout["mass"] == 0.16
    assert abs(layout["span"] - 0.055) < 1e-9
    assert abs(layout["stem_length"] - 0.025) < 1e-9
    assert abs(layout["break_force"] - 17.341074446037137) < 1e-12


def test_break_modes_vary_independent_native_limits():
    force = 17.341074446037137
    assert PROBE.thresholds("breakable", force) == (force, 0.35)
    assert PROBE.thresholds("unbreakable", force) == (1e30, 1e30)
    assert PROBE.thresholds("force-only", force) == (force, 1e30)
    assert PROBE.thresholds("torque-only", force) == (1e30, 0.35)


def test_nonfinite_diagnostics_remain_valid_json():
    import json

    report = PROBE.json_safe({"all_values_finite": False,
                              "position": [float("nan"), float("inf"), 1.0]})
    assert report["position"] == [None, None, 1.0]
    json.dumps(report, allow_nan=False)


class MinimalStemTests(unittest.TestCase):
    test_geometry = staticmethod(test_seed42_geometry_matches_selected_asset)
    test_modes = staticmethod(test_break_modes_vary_independent_native_limits)
    test_json = staticmethod(test_nonfinite_diagnostics_remain_valid_json)


if __name__ == "__main__":
    unittest.main()
