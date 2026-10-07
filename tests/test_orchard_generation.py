"""Offline reproducibility and geometry checks for multi-tree orchard specs."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
builder_path = ROOT / "scripts/build_orchardbench_usd.py"
builder_spec = importlib.util.spec_from_file_location("orchard_generation_builder_test", builder_path)
builder = importlib.util.module_from_spec(builder_spec)
builder_spec.loader.exec_module(builder)
adapter = builder.load_adapter()
generator_path = ROOT / "src/vla_isaaclab/envs/orchard_pick/orchard_generation.py"
generator_spec = importlib.util.spec_from_file_location("_offline_orchard.orchard_generation", generator_path)
generation = importlib.util.module_from_spec(generator_spec)
generator_spec.loader.exec_module(generation)


class OrchardGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pool = Path(self.temp.name)
        source = self.pool / "seed42"
        isaac = source / "isaac"
        isaac.mkdir(parents=True)
        tree = {"schema": "orchardbench_tree_v1", "seed": 42, "units": "m", "up_axis": "Z",
                "branches": [{"id": 0, "parent_id": -1,
                              "start": [0.0, 0.0, 1.1], "end": [1.0, 0.0, 1.1],
                              "radius_start": 0.01, "radius_end": 0.005}],
                "fruits": [{"id": 8, "parent_branch_id": 0, "anchor": [0.5, 0.0, 1.1],
                            "center": [0.5, 0.0, 1.04], "radius": 0.03,
                            "stem": {"mass_kg": 0.16, "break_force_n": 17.0}}],
                "leaves": [{"id": 0, "position": [0.7, 0.1, 1.1],
                            "length": 0.08, "width": 0.04}]}
        canonical = source / "tree.json"
        canonical.write_text(json.dumps(tree), encoding="utf-8")
        usd = isaac / "tree.usda"
        usd.write_text("#usda 1.0\n", encoding="utf-8")
        layout, _ = adapter.interaction_layout(adapter.load_tree(canonical, 42), 8)
        manifest = {"schema": "orchardbench_isaac_asset_v1", "seed": 42,
                    "tree_file": "tree.usda", "selected_fruit_id": 8,
                    "tree_sha256": hashlib.sha256(usd.read_bytes()).hexdigest(),
                    "canonical_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
                    "layout": json.loads(json.dumps(layout.metadata()))}
        (isaac / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_deterministic_native_geometry_and_separation(self):
        first = generation.generate_orchard_spec(self.pool, 101)
        second = generation.generate_orchard_spec(self.pool, 101)
        self.assertEqual(first, second)
        self.assertTrue(generation.validate_spec(first, self.pool))
        self.assertEqual(len(first["trees"]), 12)
        self.assertEqual(set(first["substream_seeds"]), set(generation.STREAM_NAMES))
        self.assertEqual(first["lighting"]["preset"], "morning")
        self.assertEqual(first["parameters"]["tree_scale"], 1.0)
        self.assertEqual(first["basket"]["yaw_deg"], 0.0)
        for tree in first["trees"]:
            self.assertEqual(tree["asset"]["canonical_path"], "seed42/tree.json")
            self.assertEqual(tree["selected_fruit_id"], 8)
        self.assertGreaterEqual(first["feasibility"]["minimum_crown_gap_m"], 0.20)
        self.assertGreater(first["feasibility"]["robot_body_wood_clearance_m"], 0)
        self.assertGreater(first["feasibility"]["basket_wood_clearance_m"], 0)
        self.assertEqual(first["feasibility"]["status"], "proxy_only")
        self.assertFalse(first["feasibility"]["full_collision_ik_verified"])

    def test_independent_substreams_and_tampering(self):
        first = generation.generate_orchard_spec(self.pool, 7)
        changed = generation.generate_orchard_spec(self.pool, 8)
        self.assertNotEqual(first["substream_seeds"], changed["substream_seeds"])
        self.assertNotEqual([tree["position"] for tree in first["trees"]],
                            [tree["position"] for tree in changed["trees"]])
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False)), first)
        self.assertTrue(generation.validate_spec(json.loads(json.dumps(first)), self.pool))
        tampered = copy.deepcopy(first)
        tampered["trees"][0]["position"][0] += 0.1
        with self.assertRaises(ValueError):
            generation.validate_spec(tampered, self.pool)
        tampered = copy.deepcopy(first)
        tampered["camera"]["eye"][0] = float("nan")
        with self.assertRaises(ValueError):
            generation.validate_spec(tampered, self.pool)

    def test_changed_asset_hash_rejected(self):
        spec = generation.generate_orchard_spec(self.pool, 4)
        (self.pool / "seed42/isaac/tree.usda").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            generation.validate_spec(spec, self.pool)

    def test_invalid_counts_and_seed_rejected(self):
        for count in (0, -1, True, 1.0, "2", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                generation.generate_orchard_spec(self.pool, 4, rows=count)
            with self.subTest(slots=count), self.assertRaises(ValueError):
                generation.generate_orchard_spec(self.pool, 4, slots_per_row=count)
        for seed in (True, 1.0, "4", -1, 2**32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                generation.generate_orchard_spec(self.pool, seed)

    def test_rejects_foliage_and_wood_when_all_stances_fail(self):
        with mock.patch.object(generation, "_foliage_clearance", return_value=-0.01):
            with self.assertRaisesRegex(ValueError, "positive wood, basket, foliage, and robot-basket clearance"):
                generation.generate_orchard_spec(self.pool, 4, rows=1, slots_per_row=1)
        with mock.patch.object(generation, "_body_clearance", return_value=-0.01):
            with self.assertRaisesRegex(ValueError, "feasible stance proxy"):
                generation.generate_orchard_spec(self.pool, 4, rows=1, slots_per_row=1)
        with mock.patch.object(generation, "_robot_basket_clearance", return_value=-0.01):
            with self.assertRaisesRegex(ValueError, "robot-basket clearance"):
                generation.generate_orchard_spec(self.pool, 4, rows=1, slots_per_row=1)

    def test_robot_basket_proxy_rejects_intersection(self):
        self.assertLess(generation._robot_basket_clearance((0.0, -0.43),
                        [-0.35, -0.40, 0.78]), 0)
        self.assertGreater(generation._robot_basket_clearance((0.0, -0.43),
                           [-0.43, -0.40, 0.78]), 0)

    def test_prepared_run_specs_have_clear_stances_and_current_hashes(self):
        pool = ROOT / "outputs/orchardbench/tree_pool"
        folder = ROOT / "outputs/orchardbench/stochastic_orchards"
        for seed in (101, 202, 303, 404):
            with self.subTest(seed=seed):
                spec = json.loads((folder / f"seed{seed}/orchard_spec.json").read_text())
                self.assertTrue(generation.validate_spec(spec, pool))
                self.assertGreater(spec["feasibility"]["robot_body_wood_clearance_m"], 0)
                self.assertGreater(spec["feasibility"]["robot_body_visual_foliage_clearance_m"], 0)
                self.assertGreater(spec["feasibility"]["basket_wood_clearance_m"], 0)
                self.assertGreater(spec["feasibility"]["robot_basket_clearance_m"], 0)
                self.assertIn(spec["parameters"]["robot_stance_local_offset_m"],
                              spec["parameters"]["robot_stance_candidate_offsets_m"])
                self.assertIn(spec["parameters"]["basket_local_offset_m"],
                              spec["parameters"]["basket_candidate_offsets_m"])
                if seed == 202:
                    self.assertNotEqual(spec["parameters"]["robot_stance_local_offset_m"],
                                        spec["parameters"]["robot_stance_candidate_offsets_m"][0])
                    self.assertEqual(spec["parameters"]["basket_local_offset_m"],
                                     [-0.43, -0.40, 0.78])

    def test_crown_radius_covers_all_source_extents(self):
        tree = adapter.load_tree(self.pool / "seed42/tree.json", 42)
        radius = generation._crown_radius(tree)
        self.assertGreaterEqual(radius, 1.01)
        tree["leaves"].append({"position": [3.0, 0.0, 1.0], "length": .1, "width": .1})
        self.assertGreaterEqual(generation._crown_radius(tree), 3.15)


if __name__ == "__main__":
    unittest.main()
