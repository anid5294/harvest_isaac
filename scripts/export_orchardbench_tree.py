#!/usr/bin/env python3
"""Export OrchardBench's pinned apple generator to simulator-independent JSON.

Run with a Python that has NumPy (Blender's bundled Python works). The source
checkout is read-only. Only OrchardBench's geometry/config modules execute;
the fruit placement function is extracted from fruit.py because importing that
whole module would initialize Warp runtime code unrelated to placement.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types


SOURCE_URL = "https://github.com/humphreymunn/orchardbench"
PINNED_REVISION = "6313313db8b1a7d23fb2cc3afd67cac46f29399a"
PACKAGE = "_orchardbench_tree_export_source"


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_source(checkout: Path):
    import numpy as np

    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != PINNED_REVISION:
        raise ValueError(f"OrchardBench revision {revision} differs from pinned {PINNED_REVISION}")
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True).strip()
    if dirty:
        raise ValueError("OrchardBench checkout has modified tracked files")
    if "Apache License" not in (checkout / "LICENSE").read_text()[:200]:
        raise ValueError("Expected OrchardBench Apache 2.0 license not found")
    folder = checkout / "treesim"
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(folder)]
    sys.modules[PACKAGE] = package
    config = _load_file(f"{PACKAGE}.config", folder / "config.py")
    skeleton = _load_file(f"{PACKAGE}.skeleton", folder / "skeleton.py")
    lsystem = _load_file(f"{PACKAGE}.lsystem", folder / "lsystem.py")
    foliage = _load_file(f"{PACKAGE}.foliage", folder / "foliage.py")

    # fruit.py also defines Warp kernels and a runtime AppleField. Its pure
    # ApplePlacement/place_apples AST nodes use only NumPy and the config.
    parsed = ast.parse((folder / "fruit.py").read_text(), filename=str(folder / "fruit.py"))
    pure_nodes = [node for node in parsed.body
                  if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                  and node.name in {"ApplePlacement", "place_apples"}]
    if {node.name for node in pure_nodes} != {"ApplePlacement", "place_apples"}:
        raise ValueError("Pinned OrchardBench fruit placement API changed")
    fruit = types.ModuleType(f"{PACKAGE}.fruit_placement_only")
    sys.modules[fruit.__name__] = fruit
    fruit.__dict__.update({"np": np, "dataclass": dataclass,
                           "FruitParams": config.FruitParams,
                           "TreeSkeleton": skeleton.TreeSkeleton})
    tree = ast.Module(body=[ast.ImportFrom(module="__future__",
                                           names=[ast.alias(name="annotations")], level=0),
                            *pure_nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(folder / "fruit.py"), "exec"),
         fruit.__dict__)
    # Reuse builder.py's exact quaternion convention for parent-local records.
    frame_nodes = [node for node in ast.parse((folder / "builder.py").read_text()).body
                   if isinstance(node, ast.FunctionDef)
                   and node.name in {"_qconj", "_qmul", "_qrot"}]
    if {node.name for node in frame_nodes} != {"_qconj", "_qmul", "_qrot"}:
        raise ValueError("Pinned OrchardBench frame helper API changed")
    frame_math = types.ModuleType(f"{PACKAGE}.frame_math_only")
    frame_math.np = np
    sys.modules[frame_math.__name__] = frame_math
    exec(compile(ast.fix_missing_locations(ast.Module(body=frame_nodes, type_ignores=[])),
                 str(folder / "builder.py"), "exec"), frame_math.__dict__)
    return np, config, lsystem, foliage, fruit, frame_math


def _leaf_meshes(foliage, fp):
    # Source leaf_mesh only needs newton.Mesh as a value container. Supplying a
    # collector preserves its exact vertices and double-sided triangles.
    class MeshCollector:
        def __init__(self, vertices, indices, **_kwargs):
            self.vertices = vertices
            self.indices = indices

    previous = sys.modules.get("newton")
    stub = types.ModuleType("newton")
    stub.Mesh = MeshCollector
    sys.modules["newton"] = stub
    try:
        return foliage.leaf_meshes(fp)
    finally:
        if previous is None:
            del sys.modules["newton"]
        else:
            sys.modules["newton"] = previous


def export_tree(checkout: Path, seed: int = 42, foliage_density: float = 0.6,
                max_apples: int = 40) -> dict:
    """Run native OrchardBench geometry and return the canonical tree record."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not (0 < foliage_density <= 2.0):
        raise ValueError("foliage_density must be in (0, 2]")
    if type(max_apples) is not int or max_apples < 1:
        raise ValueError("max_apples must be a positive integer")
    checkout = Path(checkout).resolve()
    np, config, lsystem, foliage, fruit, frame_math = _load_source(checkout)
    source_files = ["LICENSE", "treesim/config.py", "treesim/skeleton.py",
                    "treesim/lsystem.py", "treesim/foliage.py", "treesim/fruit.py",
                    "treesim/builder.py"]
    source_hashes = {name: hashlib.sha256((checkout/name).read_bytes()).hexdigest()
                     for name in source_files}
    params = config.preset("apple")
    skel = lsystem.generate(params, seed=seed)
    fp = config.FoliageParams()
    fp.set_density(foliage_density)
    fr = config.FruitParams(enabled=True, max_count=max_apples)
    source_leaves = foliage.place_leaves(skel, fp, seed=seed)
    source_fruits = fruit.place_apples(skel, fr, seed=seed)
    source_meshes = _leaf_meshes(foliage, fp)

    def parent_local(parent_seg, world_position, world_quat=None):
        parent = skel[parent_seg]
        conjugate = frame_math._qconj(parent.frame)
        local = frame_math._qrot(conjugate, world_position - parent.start)
        orientation = frame_math._qmul(
            conjugate, np.array([0., 0., 0., 1.]) if world_quat is None else world_quat)
        return {"position": local.tolist(), "orientation_xyzw": orientation.tolist()}

    # Match builder.py's independent seed+313 leaf colour/mesh class stream.
    leaf_rng = np.random.default_rng(seed + 313)
    leaves = []
    for i, lp in enumerate(source_leaves):
        brightness = float(np.clip(leaf_rng.normal(1.0, 0.16), 0.6, 1.5))
        yellow = float(np.clip(leaf_rng.normal(1.0, 0.14), 0.7, 1.5))
        mesh_class = int(leaf_rng.integers(0, len(source_meshes)))
        color = [float(np.clip(fp.leaf_color[0]*brightness*yellow, 0, 1)),
                 float(np.clip(fp.leaf_color[1]*brightness, 0, 1)),
                 float(np.clip(fp.leaf_color[2]*brightness/yellow, 0, 1))]
        leaves.append({"id": i, "parent_branch_id": int(lp.parent_seg),
                       "position": lp.attach.tolist(),
                       "orientation_xyzw": lp.frame.tolist(),
                       "parent_local": parent_local(lp.parent_seg, lp.attach, lp.frame),
                       "length": float(lp.length), "width": float(lp.width),
                       "mesh_class": mesh_class, "color": color})

    apple_rng = np.random.default_rng(seed + 99)
    fruits = []
    for i, ap in enumerate(source_fruits):
        drop = fr.stem_length + ap.radius
        center = ap.attach - np.array([0.0, 0.0, drop])
        fruits.append({"id": i, "parent_branch_id": int(ap.parent_seg),
                       "anchor": ap.attach.tolist(),
                       "center": center.tolist(),
                       "anchor_parent_local": parent_local(ap.parent_seg, ap.attach),
                       "center_parent_local": parent_local(ap.parent_seg, center),
                       "radius": float(ap.radius), "color": list(ap.color),
                       "stem": {"length": float(fr.stem_length),
                                "break_force_n": float(apple_rng.uniform(*fr.detach_force)),
                                "mass_kg": float(fr.mass)}})

    branches = [{"id": int(s.index), "parent_id": int(s.parent),
                 "start": s.start.tolist(), "end": s.end.tolist(),
                 "radius_start": float(s.radius_start),
                 "radius_end": float(s.radius_end),
                 "frame_xyzw": s.frame.tolist(), "order": int(s.order),
                 "depth": int(s.depth)} for s in skel.segments]
    meshes = [{"class": i, "vertices": mesh.vertices.tolist(),
               "triangle_indices": mesh.indices.tolist(),
               "double_sided": True} for i, mesh in enumerate(source_meshes)]
    return {
        "schema": "orchardbench_tree_v1", "tree_id": f"orchardbench_apple_seed{seed}",
        "seed": seed, "training_system": "orchardbench_apple_central_leader",
        "units": "m", "up_axis": "Z",
        "source": {"url": SOURCE_URL, "revision": PINNED_REVISION,
                   "license": "Apache-2.0", "preset": "apple",
                   "license_file": "LICENSE",
                   "source_files_sha256": source_hashes,
                   "geometry_modules": ["config.py", "skeleton.py", "lsystem.py",
                                        "foliage.py", "fruit.py", "builder.py"],
                   "lsystem_params": asdict(params),
                   "foliage_params": asdict(fp),
                   "fruit_params": asdict(fr),
                   "foliage_density": foliage_density,
                   "max_apples": max_apples,
                   "notes": "Native source placement; no robot reach adjustment by exporter."},
        "bounds": {"min": skel.bounds()[0].tolist(),
                   "max": skel.bounds()[1].tolist(),
                   "height": float(skel.height()),
                   "total_wood_length": float(skel.total_length())},
        "branches": branches, "fruits": fruits, "leaves": leaves,
        "leaf_meshes": meshes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="OrchardBench checkout at the pinned revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--foliage-density", type=float, default=0.6)
    parser.add_argument("--max-apples", type=int, default=40)
    args = parser.parse_args(argv)
    record = export_tree(args.source, args.seed, args.foliage_density, args.max_apples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    shutil.copyfile(args.source / "LICENSE", args.output.parent / "LICENSE")
    print(f"Wrote {args.output}: {len(record['branches'])} branches, "
          f"{len(record['fruits'])} fruit, {len(record['leaves'])} leaves")


if __name__ == "__main__":
    main()
