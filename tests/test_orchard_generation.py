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
        self.assertEqual(first["parameters"]["layout_model"], "structured_v4")
        self.assertEqual(len(first["trees"]), sum(map(len, first["parameters"]["occupied_slots"])))
        self.assertEqual(set(first["substream_seeds"]), set(generation.STREAM_NAMES))
        self.assertEqual(first["lighting"]["preset"], "morning")
        self.assertEqual(first["parameters"]["tree_scale"], 1.0)
        self.assertEqual(first["basket"]["yaw_deg"], 0.0)
        for tree in first["trees"]:
            self.assertEqual(tree["asset"]["canonical_path"], "seed42/tree.json")
            self.assertEqual(tree["selected_fruit_id"], 8)
        self.assertGreaterEqual(first["feasibility"]["minimum_crown_gap_m"],
                                -generation.MAX_CROWN_OVERLAP_M)
        self.assertIsNone(first["feasibility"]["minimum_major_wood_gap_m"])
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
        for count in (0, -1, True, 1.0, "2"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                generation.generate_orchard_spec(self.pool, 4, rows=count)
            with self.subTest(slots=count), self.assertRaises(ValueError):
                generation.generate_orchard_spec(self.pool, 4, slots_per_row=count)
        for seed in (True, 1.0, "4", -1, 2**32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                generation.generate_orchard_spec(self.pool, seed)

    def test_structured_planting_bounds_occupancy_and_footprint(self):
        seen_rows, seen_counts = set(), set()
        for seed in range(101, 111):
            spec = generation.generate_orchard_spec(self.pool, seed)
            params = spec["parameters"]
            rows = params["rows"]
            seen_rows.add(rows)
            seen_counts.update(params["row_slot_counts"])
            self.assertTrue(3 <= rows <= 5)
            self.assertEqual(len(params["row_slot_counts"]), rows)
            self.assertEqual(len(params["row_gap_m"]), rows - 1)
            self.assertEqual(len(params["along_row_gap_m"]), rows)
            self.assertEqual(len(spec["infrastructure"]["surfaces"]), rows + 1)
            self.assertTrue(all(6 <= n <= 9 for n in params["row_slot_counts"]))
            self.assertTrue(all(3.15 <= d <= 3.75 for d in params["row_gap_m"]))
            self.assertTrue(all(2.55 <= d <= 3.10 for gaps in params["along_row_gap_m"] for d in gaps))
            self.assertEqual(len(set(params["row_slot_counts"])), 1)
            self.assertEqual(params["missing_slot_probability"], 0.0)
            self.assertLessEqual(max(params["row_gap_m"]) - min(params["row_gap_m"]), 0.20 + 1e-9)
            self.assertLessEqual(max(d for gaps in params["along_row_gap_m"] for d in gaps) -
                                 min(d for gaps in params["along_row_gap_m"] for d in gaps), 0.19 + 1e-9)
            for i, gap in enumerate(params["row_gap_m"]):
                self.assertAlmostEqual(params["row_centers_m"][i + 1] -
                                       params["row_centers_m"][i], gap)
            tree_slots = {(t["row"], t["slot"]) for t in spec["trees"]}
            planned_slots = {(r, s) for r, slots in enumerate(params["occupied_slots"]) for s in slots}
            self.assertEqual(tree_slots, planned_slots)
            self.assertEqual(len(tree_slots), len(spec["trees"]))
            self.assertTrue(all(slots == list(range(n)) for slots, n in
                                zip(params["occupied_slots"], params["row_slot_counts"])))
            for row, (count, gaps, occupied) in enumerate(zip(params["row_slot_counts"],
                    params["along_row_gap_m"], params["occupied_slots"])):
                self.assertEqual(len(gaps), count - 1)
                self.assertEqual(occupied, sorted(set(occupied)))
                self.assertTrue(all(0 <= slot < count for slot in occupied))
                xs = [0.0]
                for gap in gaps:
                    xs.append(xs[-1] + gap)
                center = (xs[0] + xs[-1]) / 2
                for tree in (t for t in spec["trees"] if t["row"] == row):
                    local_x, local_y = generation._rotate(*tree["position"][:2],
                                                            -params["global_yaw_deg"])
                    self.assertLessEqual(abs(local_x - (xs[tree["slot"]] - center)),
                                         params["planting_jitter_m"] + 1e-9)
                    self.assertLessEqual(abs(local_y - params["row_centers_m"][row]),
                                         params["planting_jitter_m"] + 1e-9)
            ground = spec["infrastructure"]["surfaces"][0]
            gx, gy = generation._rotate(*ground["position"][:2], -params["global_yaw_deg"])
            half_x, half_y = ground["size"][0] / 2, ground["size"][1] / 2
            for tree in spec["trees"]:
                x, y = generation._rotate(*tree["position"][:2], -params["global_yaw_deg"])
                self.assertLessEqual(abs(x - gx) + tree["crown_radius_m"], half_x)
                self.assertLessEqual(abs(y - gy) + tree["crown_radius_m"], half_y)
            self.assertTrue(generation.validate_spec(spec, self.pool))
        self.assertGreater(len(seen_rows), 1)
        self.assertGreater(len(seen_counts), 1)

    @unittest.skipUnless((ROOT / "outputs/orchardbench/tree_pool").is_dir() and all((ROOT / "outputs/orchardbench/dense_orchards" / f"seed{seed}/orchard_spec.json").is_file() for seed in (101, 202, 303, 404)), "Requires local generated orchard fixtures")
    def test_cached_dense_v3_specs_still_replay(self):
        pool = ROOT / "outputs/orchardbench/tree_pool"
        folder = ROOT / "outputs/orchardbench/dense_orchards"
        for seed in (101, 202, 303, 404):
            with self.subTest(seed=seed):
                spec = json.loads((folder / f"seed{seed}/orchard_spec.json").read_text())
                self.assertEqual(spec["parameters"]["layout_model"], "structured_v3")
                self.assertTrue(generation.validate_spec(spec, pool))

    @unittest.skipUnless((ROOT / "outputs/orchardbench/tree_pool").is_dir(), "Requires local generated orchard fixtures")
    def test_coherent_pool_seeds_avoid_neighbor_morphology_repeats(self):
        pool = ROOT / "outputs/orchardbench/tree_pool"
        for seed in (101, 202, 303, 404):
            with self.subTest(seed=seed):
                spec = generation.generate_orchard_spec(pool, seed)
                trees = spec["trees"]
                self.assertGreaterEqual(len({tree["asset_id"] for tree in trees}), 5)
                self.assertFalse(any(a["asset_id"] == b["asset_id"]
                                     for a, b in zip(trees, trees[1:]) if a["row"] == b["row"]))
                self.assertEqual(len(trees), sum(spec["parameters"]["row_slot_counts"]))
                self.assertGreaterEqual(spec["feasibility"]["minimum_major_wood_gap_m"],
                                        generation.MIN_MAJOR_WOOD_GAP_M)

    def test_explicit_dimensions_and_legacy_replay(self):
        fixed = generation.generate_orchard_spec(self.pool, 9, rows=2, slots_per_row=3)
        self.assertEqual(fixed["parameters"]["row_slot_counts"], [3, 3])
        self.assertEqual(fixed["parameters"]["requested_rows"], 2)
        self.assertEqual(fixed["parameters"]["requested_slots_per_row"], 3)
        self.assertTrue(generation.validate_spec(fixed, self.pool))
        legacy = generation.generate_orchard_spec(self.pool, 9, rows=2, slots_per_row=3,
                                                  _legacy_layout=True)
        self.assertNotIn("layout_model", legacy["parameters"])
        self.assertEqual(len(legacy["trees"]), 6)
        self.assertTrue(generation.validate_spec(legacy, self.pool))
        prior = generation.generate_orchard_spec(self.pool, 9, rows=2, slots_per_row=3,
                                                 _structured_v2=True)
        self.assertEqual(prior["parameters"]["layout_model"], "structured_v2")
        self.assertTrue(generation.validate_spec(prior, self.pool))

    def test_major_wood_capsules_and_bounded_crown_overlap(self):
        self.assertAlmostEqual(generation._segment_segment_distance(
            (0, 0, 0), (1, 0, 0), (0.5, -1, 0), (0.5, 1, 0)), 0)
        self.assertAlmostEqual(generation._segment_segment_distance(
            (0, 0, 0), (1, 0, 0), (0.5, -1, 0.3), (0.5, 1, 0.3)), 0.3)
        self.assertAlmostEqual(generation._segment_segment_distance(
            (0, 0, 0), (1, 0, 0), (2, 1, 0), (2, 2, 0)), 2**0.5)
        self.assertAlmostEqual(generation._segment_segment_distance(
            (0, 0, 0), (1, 0, 0), (0.5, 0.4, 0), (1.5, 0.4, 0)), 0.4)
        self.assertAlmostEqual(generation._segment_segment_distance(
            (0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0)), 1)
        trees = [{"instance_id": "a", "position": [0, 0, 0], "yaw_deg": 0,
                  "crown_radius_m": 1.0},
                 {"instance_id": "b", "position": [1.8, 0, 0], "yaw_deg": 0,
                  "crown_radius_m": 1.0}]
        wood = [([0, 0, 0], [0, 0, 1], 0.03)]
        lookup = {key: {"major_wood": wood, "major_radius": 0.03} for key in ("a", "b")}
        crown_gap, wood_gap = generation._dense_separation(trees, lookup)
        self.assertAlmostEqual(crown_gap, -0.2)
        self.assertGreater(wood_gap, generation.MIN_MAJOR_WOOD_GAP_M)
        trees[1]["position"][0] = 0.07
        with self.assertRaises(ValueError):
            generation._dense_separation(trees, lookup)

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

    @unittest.skipUnless((ROOT / "outputs/orchardbench/tree_pool").is_dir() and all((ROOT / "outputs/orchardbench/stochastic_orchards" / f"seed{seed}/orchard_spec.json").is_file() for seed in (101, 202, 303, 404)), "Requires local generated orchard fixtures")
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
