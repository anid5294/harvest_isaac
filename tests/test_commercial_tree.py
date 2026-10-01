"""Simulator-free topology and export checks for the trained orchard tree."""

import importlib.util
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest


MODULE = (Path(__file__).resolve().parents[1] /
          "src/vla_isaaclab/envs/orchard_pick/commercial_tree.py")
spec = importlib.util.spec_from_file_location("commercial_tree_test", MODULE)
tree_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tree_module
spec.loader.exec_module(tree_module)


class CommercialTreeTests(unittest.TestCase):
    @staticmethod
    def _point_segment_distance(point, start, end):
        delta = tuple(end[i]-start[i] for i in range(3))
        along = sum((point[i]-start[i])*delta[i] for i in range(3))
        length_squared = sum(v*v for v in delta)
        t = max(0., min(1., along/length_squared))
        return math.dist(point, tuple(start[i]+t*delta[i] for i in range(3)))

    def test_seeded_shape_and_reachable_target(self):
        first = tree_module.generate_commercial_tree(42)
        self.assertEqual(first, tree_module.generate_commercial_tree(42))
        self.assertNotEqual(first, tree_module.generate_commercial_tree(43))
        self.assertEqual(len(first.fruits), 13)
        self.assertEqual(sum(f.is_target for f in first.fruits), 1)
        self.assertTrue(first.fruits[0].is_target)
        self.assertLess(math.dist(first.fruits[0].center, (-.10, -.29, 1.0)), .04)
        self.assertGreater(max(f.center[2] for f in first.fruits), 2.3)
        self.assertAlmostEqual(max(b.end[2] for b in first.branches), 2.85)
        for fruit in first.fruits:
            self.assertEqual(fruit.anchor[:2], fruit.center[:2])
            self.assertAlmostEqual(fruit.anchor[2]-fruit.center[2],
                                   fruit.radius + tree_module.STEM_LENGTH)

    def test_connected_tapered_wood_and_attached_foliage(self):
        generated = tree_module.generate_commercial_tree(42)
        self.assertGreaterEqual(len(generated.leaves), 150)
        self.assertLessEqual(len(generated.leaves), 182)
        for index, segment in enumerate(generated.branches):
            self.assertGreater(math.dist(segment.start, segment.end), 0)
            self.assertGreaterEqual(segment.radius_start, segment.radius_end)
            self.assertGreater(segment.radius_end, 0)
            self.assertTrue(all(math.isfinite(v) for p in (segment.start, segment.end)
                                for v in p))
            if index == 0:
                self.assertEqual(segment.parent, -1)
            else:
                self.assertGreaterEqual(segment.parent, 0)
                self.assertLess(segment.parent, index)
                self.assertEqual(segment.start,
                                 generated.branches[segment.parent].end)
        for fruit in generated.fruits:
            self.assertEqual(generated.branches[fruit.branch_index].end,
                             fruit.anchor)
            self.assertEqual(generated.branches[fruit.branch_index].kind,
                             "fruit_spur")
        for leaf in generated.leaves:
            twig = generated.branches[leaf.branch_index]
            self.assertEqual(twig.kind, "twig")
            self.assertLess(math.dist(leaf.anchor, twig.start) +
                            math.dist(leaf.anchor, twig.end) -
                            math.dist(twig.start, twig.end), 1e-8)
            self.assertLess(math.dist(leaf.anchor, leaf.center), .1)
        self.assertEqual(sum(s.kind == "stake" for s in generated.trellis), 1)
        self.assertEqual(sum(s.kind == "wire" for s in generated.trellis), 4)

    def test_foliage_asset_and_metadata(self):
        generated = tree_module.generate_commercial_tree(7)
        self.assertEqual(generated.metadata()["seed"], 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "foliage.usda"
            self.assertEqual(tree_module.write_foliage_usda(generated, path), path)
            content = path.read_text()
            self.assertIn('defaultPrim = "Foliage"', content)
            self.assertIn('bool doubleSided = true', content)
            self.assertIn('UsdPreviewSurface', content)
            self.assertEqual(content.count("def Mesh"), 1)
            point_match = re.search(r"point3f\[\] points = \[(.*?)\]", content)
            count_match = re.search(r"int\[\] faceVertexCounts = \[(.*?)\]", content)
            index_match = re.search(r"int\[\] faceVertexIndices = \[(.*?)\]", content)
            self.assertIsNotNone(point_match)
            self.assertIsNotNone(count_match)
            self.assertIsNotNone(index_match)
            points = re.findall(r"\([^()]+\)", point_match.group(1))
            counts = [int(x) for x in count_match.group(1).split(", ")]
            indices = [int(x) for x in index_match.group(1).split(", ")]
            self.assertEqual(len(points), 4*len(generated.leaves))
            self.assertEqual(len(counts), 2*len(generated.leaves))
            self.assertEqual(sum(counts), len(indices))
            self.assertTrue(all(0 <= i < len(points) for i in indices))

    def test_seed_validation_and_100_seed_physical_clearance(self):
        with self.assertRaises(ValueError):
            tree_module.generate_commercial_tree(True)
        with self.assertRaises(ValueError):
            tree_module.generate_commercial_tree(1.5)
        for seed in range(100):
            generated = tree_module.generate_commercial_tree(seed)
            for j, wood in enumerate(generated.branches):
                if j == 0:
                    self.assertEqual(wood.parent, -1)
                else:
                    self.assertGreaterEqual(wood.parent, 0)
                    self.assertLess(wood.parent, j)
                    self.assertEqual(wood.start,
                                     generated.branches[wood.parent].end)
            for i, fruit in enumerate(generated.fruits):
                for j, other in enumerate(generated.fruits[:i]):
                    self.assertGreater(
                        math.dist(fruit.center, other.center),
                        fruit.radius+other.radius,
                        f"seed={seed} fruit pair={i},{j}")
                for j, wood in enumerate(generated.branches):
                    clearance = self._point_segment_distance(
                        fruit.center, wood.start, wood.end)
                    self.assertGreater(
                        clearance, fruit.radius+max(wood.radius_start, wood.radius_end),
                        f"seed={seed} fruit={i} wood={j} kind={wood.kind}")
                for j, trellis in enumerate(generated.trellis):
                    clearance = self._point_segment_distance(
                        fruit.center, trellis.start, trellis.end)
                    self.assertGreater(
                        clearance, fruit.radius+trellis.radius,
                        f"seed={seed} fruit={i} trellis={j}")


if __name__ == "__main__":
    unittest.main()
