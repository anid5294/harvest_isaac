#!/usr/bin/env python3
"""Generate or validate an orchard spec offline, without importing Isaac Lab."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import types


def generator_module():
    folder = Path(__file__).resolve().parents[1] / "src/vla_isaaclab/envs/orchard_pick"
    name = "_offline_orchard_spec"
    package = types.ModuleType(name)
    package.__path__ = [str(folder)]
    sys.modules[name] = package
    return importlib.import_module(name + ".orchard_generation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchard-seed", type=int, default=101)
    parser.add_argument("--pool", type=Path, default=Path("outputs/orchardbench/tree_pool"))
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--trees-per-row", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    module = generator_module()
    if args.validate_only:
        spec = json.loads(args.output.read_text())
    else:
        spec = module.generate_orchard_spec(args.pool, args.orchard_seed,
            rows=args.rows, slots_per_row=args.trees_per_row)
    module.validate_spec(spec, args.pool)
    if not args.validate_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"validated": True, "spec": str(args.output),
                      "seed": spec["seed"], "target": spec["target"],
                      "feasibility": spec["feasibility"]}, indent=2))


if __name__ == "__main__":
    main()
