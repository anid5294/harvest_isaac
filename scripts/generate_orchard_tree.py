#!/usr/bin/env python3
"""Export original orchard tree visuals without importing Isaac Sim."""

import argparse
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "src/vla_isaaclab/envs/orchard_pick"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for tree.usda and manifest.json")
    parser.add_argument("--layout-seed", type=int, default=42)
    parser.add_argument("--tree-seed", type=int, default=42)
    args = parser.parse_args()
    layout = _load("orchard_pick_layout_export", MODULE_DIR / "layout.py")
    tree = _load("orchard_pick_tree_export", MODULE_DIR / "tree_geometry.py")
    manifest = tree.write_tree_asset(
        args.output_dir, layout.make_layout(args.layout_seed), args.tree_seed)
    print(f"{args.output_dir / manifest['tree_file']} ({manifest['leaf_count']} leaves, "
          f"{manifest['wood_triangles']} wood triangles)")


if __name__ == "__main__":
    main()
