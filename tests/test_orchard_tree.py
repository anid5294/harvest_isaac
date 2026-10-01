"""Simulator-free checks for original procedural tree visuals and USD export."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_DIR = Path(__file__).resolve().parents[1] / "src/vla_isaaclab/envs/orchard_pick"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


layout_module = _load("orchard_pick_layout_tree_test", "layout.py")
tree = _load("orchard_pick_tree_test", "tree_geometry.py")


class OrchardTreeTests(unittest.TestCase):
    def setUp(self):
        self.layout = layout_module.make_layout(42)

    def test_determinism_variation_and_complexity_bounds(self):
        first = tree.generate_tree(self.layout, 79)
        repeated = tree.generate_tree(self.layout, 79)
        other = tree.generate_tree(self.layout, 80)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other)
        self.assertGreaterEqual(len(first.leaf_centers), 250)
        self.assertLessEqual(len(first.leaf_centers), 320)
        self.assertLess(len(first.wood.faces), 15000)
        self.assertLess(sum(len(mesh.faces) for mesh in first.leaves), 2500)
        self.assertEqual(len(first.petiole_segments), len(first.leaf_centers))

    def test_finite_nondegenerate_mesh_and_target_clearance(self):
        generated = tree.generate_tree(self.layout, 42)
        for mesh in (generated.wood, *generated.leaves):
            self.assertTrue(all(math.isfinite(v) for p in mesh.points for v in p))
            for a, b, c in mesh.faces:
                self.assertTrue(all(0 <= i < len(mesh.points) for i in (a, b, c)))
                ab = tree._sub(mesh.points[b], mesh.points[a])
                ac = tree._sub(mesh.points[c], mesh.points[a])
                self.assertGreater(tree._dot(tree._cross(ab, ac),
                                             tree._cross(ab, ac)), 1e-14)
        for start, end in generated.secondary_segments:
            self.assertGreaterEqual(tree._segment_distance(start, end, generated.target),
                                    tree.PARAMS["target_clearance_radius"])
        leaf_vertices = {p for mesh in generated.leaves for p in mesh.points}
        branch_tips = {end for _, end in generated.secondary_segments}
        for start, heel in generated.petiole_segments:
            self.assertIn(start, branch_tips)
            self.assertIn(heel, leaf_vertices)
            self.assertGreaterEqual(tree._segment_distance(start, heel, generated.target),
                                    tree.PARAMS["target_clearance_radius"])
        for mesh in generated.leaves:
            self.assertTrue(all(math.dist(p, generated.target) >= 0.20
                                for p in mesh.points))

    def test_closed_tube_winding(self):
        mesh = tree.Mesh()
        import random
        tree._tube(mesh, (0, 0, 0), (0, 0, 1), 0.1, 0.05,
                   random.Random(0), roughness=0)
        for a, b, c in mesh.faces[:14]:
            points = [mesh.points[i] for i in (a, b, c)]
            centroid = tuple(sum(p[k] for p in points)/3 for k in range(3))
            normal = tree._cross(tree._sub(points[1], points[0]),
                                 tree._sub(points[2], points[0]))
            self.assertGreater(tree._dot(normal, (centroid[0], centroid[1], 0)), 0)

    def test_usda_manifest_hash_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp) / "asset"
            manifest = tree.write_tree_asset(target_dir, self.layout, seed=17)
            content = (target_dir / "tree.usda").read_text()
            on_disk = json.loads((target_dir / "manifest.json").read_text())
            self.assertEqual(manifest, on_disk)
            self.assertEqual(manifest["layout"], json.loads(json.dumps(self.layout.metadata())))
            self.assertEqual(manifest["tree_file"], "tree.usda")
            self.assertEqual(manifest["sha256"], hashlib.sha256(content.encode()).hexdigest())
            self.assertEqual(manifest["collision_status"], "visual_only_existing_scaffold")
            self.assertIn('defaultPrim = "Tree"', content)
            self.assertIn('metersPerUnit = 1', content)
            self.assertIn('upAxis = "Z"', content)
            self.assertIn('bool doubleSided = true', content)
            self.assertIn('prepend apiSchemas = ["MaterialBindingAPI"]', content)
            self.assertIn('UsdPreviewSurface', content)
            with self.assertRaises(FileExistsError):
                tree.write_tree_asset(target_dir, self.layout, seed=17)


if __name__ == "__main__":
    unittest.main()
