"""Checks for OrchardBench's pinned offline geometry export."""

import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "scripts/export_orchardbench_tree.py"
ARTIFACT = ROOT / "assets/orchard/orchardbench_seed42/tree.json"
SOURCE = Path(os.environ["ORCHARDBENCH_SOURCE"]).expanduser() if os.environ.get(
    "ORCHARDBENCH_SOURCE") else None
PIN = "6313313db8b1a7d23fb2cc3afd67cac46f29399a"
EXPORT_PYTHON = os.environ.get("ORCHARDBENCH_PYTHON", sys.executable)


def distance_to_segment(point, a, b):
    delta = [b[i]-a[i] for i in range(3)]
    t = sum((point[i]-a[i])*delta[i] for i in range(3))/sum(x*x for x in delta)
    t = min(1., max(0., t))
    return math.dist(point, [a[i]+t*delta[i] for i in range(3)])


def rotate(q, vector):
    x, y, z, w = q
    u = (x, y, z)
    cross = lambda a, b: (a[1]*b[2]-a[2]*b[1],
                          a[2]*b[0]-a[0]*b[2],
                          a[0]*b[1]-a[1]*b[0])
    first = cross(u, vector)
    second = cross(u, tuple(first[i]+w*vector[i] for i in range(3)))
    return [vector[i]+2*second[i] for i in range(3)]


def multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw,
            aw*bw-ax*bx-ay*by-az*bz]


class OrchardBenchExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = json.loads(ARTIFACT.read_text())

    def test_pinned_source_and_native_preset(self):
        tree = self.tree
        self.assertEqual(tree["schema"], "orchardbench_tree_v1")
        self.assertEqual(tree["seed"], 42)
        self.assertEqual(tree["units"], "m")
        self.assertEqual(tree["up_axis"], "Z")
        self.assertEqual(tree["source"]["revision"], PIN)
        self.assertEqual(tree["source"]["license"], "Apache-2.0")
        self.assertEqual(tree["source"]["preset"], "apple")
        self.assertEqual(tree["source"]["lsystem_params"]["n"], 4)
        self.assertEqual(tree["source"]["lsystem_params"]["target_height"], 2.6)
        self.assertEqual(tree["source"]["foliage_density"], 0.6)
        self.assertEqual(tree["source"]["foliage_params"]["leaves_per_terminal"], 4)
        self.assertEqual(tree["source"]["fruit_params"]["max_count"], 40)
        self.assertEqual(tree["source"]["license_file"], "LICENSE")
        self.assertIn("Apache License", (ARTIFACT.parent / "LICENSE").read_text()[:200])
        self.assertEqual(set(tree["source"]["source_files_sha256"]),
                         {"LICENSE", "treesim/config.py", "treesim/skeleton.py",
                          "treesim/lsystem.py", "treesim/foliage.py", "treesim/fruit.py",
                          "treesim/builder.py"})
        if SOURCE is not None and SOURCE.is_dir():
            revision = subprocess.check_output(
                ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
            if revision != PIN:
                return
            for name, digest in tree["source"]["source_files_sha256"].items():
                self.assertEqual(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest(), digest)

    def test_topology_and_source_placement(self):
        tree = self.tree
        branches = tree["branches"]
        self.assertEqual(len(branches), 168)
        self.assertEqual(len(tree["fruits"]), 40)
        self.assertEqual(len(tree["leaves"]), 344)
        self.assertEqual(branches[0]["parent_id"], -1)
        for i, branch in enumerate(branches):
            self.assertEqual(branch["id"], i)
            self.assertGreater(branch["radius_start"], 0)
            self.assertGreater(branch["radius_end"], 0)
            self.assertGreater(math.dist(branch["start"], branch["end"]), 0)
            if i:
                self.assertLess(branch["parent_id"], i)
                self.assertEqual(branch["start"], branches[branch["parent_id"]]["end"])
        for fruit in tree["fruits"]:
            branch = branches[fruit["parent_branch_id"]]
            self.assertLess(distance_to_segment(fruit["anchor"],
                                                branch["start"], branch["end"]), 1e-8)
            self.assertEqual(fruit["center"][:2], fruit["anchor"][:2])
            self.assertAlmostEqual(fruit["anchor"][2]-fruit["center"][2],
                                   fruit["radius"]+fruit["stem"]["length"])
            for world_key, local_key in (("anchor", "anchor_parent_local"),
                                         ("center", "center_parent_local")):
                local = fruit[local_key]
                reconstructed = [branch["start"][k] + v for k, v in enumerate(
                    rotate(branch["frame_xyzw"], local["position"]))]
                self.assertLess(math.dist(reconstructed, fruit[world_key]), 1e-8)
                world_q = multiply(branch["frame_xyzw"], local["orientation_xyzw"])
                self.assertLess(math.dist(world_q, [0, 0, 0, 1]), 1e-8)
        for leaf in tree["leaves"]:
            branch = branches[leaf["parent_branch_id"]]
            self.assertLess(distance_to_segment(leaf["position"],
                                                branch["start"], branch["end"]), 1e-8)
            self.assertIn(leaf["mesh_class"], range(len(tree["leaf_meshes"])))
            reconstructed = [branch["start"][k] + v for k, v in enumerate(
                rotate(branch["frame_xyzw"], leaf["parent_local"]["position"]))]
            self.assertLess(math.dist(reconstructed, leaf["position"]), 1e-8)
            world_q = multiply(branch["frame_xyzw"],
                               leaf["parent_local"]["orientation_xyzw"])
            self.assertLess(math.dist(world_q, leaf["orientation_xyzw"]), 1e-8)
        bounds = tree["bounds"]
        self.assertAlmostEqual(bounds["max"][2]-bounds["min"][2], bounds["height"])
        self.assertAlmostEqual(sum(math.dist(b["start"], b["end"]) for b in branches),
                               bounds["total_wood_length"])

    def test_source_leaf_meshes(self):
        for mesh in self.tree["leaf_meshes"]:
            self.assertEqual(len(mesh["vertices"]), 18)
            self.assertEqual(len(mesh["triangle_indices"]), 120)
            self.assertEqual(mesh["double_sided"], True)
            self.assertTrue(all(0 <= i < 18 for i in mesh["triangle_indices"]))

    def test_selected_physical_fruit_28_clears_all_source_wood(self):
        fruit = self.tree["fruits"][28]
        clearances = [distance_to_segment(fruit["center"],
                                          branch["start"], branch["end"])
                      - fruit["radius"]
                      - max(branch["radius_start"], branch["radius_end"])
                      for branch in self.tree["branches"]]
        self.assertGreater(min(clearances), 0.007)
        self.assertEqual(clearances.index(min(clearances)), fruit["parent_branch_id"])
        self.assertAlmostEqual(fruit["anchor"][2]-fruit["center"][2],
                               fruit["radius"]+fruit["stem"]["length"])
        for other in self.tree["fruits"]:
            if other["id"] != 28:
                self.assertGreater(math.dist(fruit["center"], other["center"]),
                                   fruit["radius"]+other["radius"])

    def test_reproduces_checked_in_artifact(self):
        if SOURCE is None or not SOURCE.is_dir() or not shutil.which("git"):
            self.skipTest("set ORCHARDBENCH_SOURCE to a pinned OrchardBench checkout")
        if not shutil.which(EXPORT_PYTHON):
            self.skipTest("ORCHARDBENCH_PYTHON interpreter unavailable")
        numpy_probe = subprocess.run([EXPORT_PYTHON, "-c", "import numpy"],
                                     capture_output=True, text=True)
        if numpy_probe.returncode:
            self.skipTest("export interpreter does not provide NumPy")
        revision = subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
        if revision != PIN:
            self.skipTest("external source checkout differs from pinned revision")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "tree.json"
            subprocess.run([EXPORT_PYTHON, str(EXPORTER),
                            "--source", str(SOURCE), "--output", str(output),
                            "--seed", "42"], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(output.read_text()), self.tree)


if __name__ == "__main__":
    unittest.main()
