#!/usr/bin/env python3
"""Prepare a deterministic pool of OrchardBench trees and Isaac USD assets.

The exporter and USD builder remain the sources of truth for generation and
validation. This script only coordinates their existing command line tools
and records per-seed outcomes so a failed candidate does not hide good ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (1, 7, 21, 42, 53, 67, 83, 97)


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True)


def dependency_error(stage: str, result: subprocess.CompletedProcess[str]) -> str | None:
    output = f"{result.stdout}\n{result.stderr}"
    if "ModuleNotFoundError" not in output and "ImportError" not in output:
        return None
    if stage == "export" and ("numpy" in output.lower() or "No module named 'numpy'" in output):
        return "Exporter requires NumPy; choose a Python with NumPy installed."
    if stage == "build" and ("pxr" in output.lower() or "isaaclab" in output.lower()):
        return "USD builder requires pxr; choose a Python with Pixar USD (pxr) installed."
    return None


def error_text(stage: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = dependency_error(stage, result)
    if detail:
        return detail
    combined = (result.stderr or result.stdout).strip()
    if not combined:
        combined = f"process exited with status {result.returncode}"
    return combined[-6000:]


def prepare_seed(seed: int, source: Path, output: Path, python: str) -> dict:
    pool_dir = output / f"seed{seed}"
    canonical = pool_dir / "tree.json"
    isaac_dir = pool_dir / "isaac"
    pool_dir.mkdir(parents=True, exist_ok=True)

    # Reuse validated prepared assets. Never overwrite a half-built/stale pool
    # and then silently mix it with newly generated geometry.
    if canonical.exists() or isaac_dir.exists():
        try:
            from build_orchardbench_usd import load_adapter
            _, _, manifest = load_adapter().load_asset(isaac_dir, seed)
            tree = json.loads(canonical.read_text())
            return ready_record(seed, canonical, tree, manifest)
        except (OSError, ValueError, KeyError) as exc:
            return {"seed": seed, "status": "existing_asset_invalid",
                    "error": f"Use a new output directory or repair this asset explicitly: {exc}"}

    export = run([
        python, str(PROJECT / "scripts/export_orchardbench_tree.py"),
        "--source", str(source), "--output", str(canonical), "--seed", str(seed),
    ], cwd=PROJECT)
    if export.returncode:
        return {"seed": seed, "status": "export_failed", "error": error_text("export", export)}

    try:
        tree = json.loads(canonical.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"seed": seed, "status": "export_failed", "error": f"Exporter did not produce valid JSON: {exc}"}

    build = run([
        python, str(PROJECT / "scripts/build_orchardbench_usd.py"),
        "--tree", str(canonical), "--output", str(isaac_dir),
    ], cwd=PROJECT)
    if build.returncode:
        return {
            "seed": seed,
            "status": "build_failed",
            "error": error_text("build", build),
            "tree_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
            "branch_count": len(tree.get("branches", [])),
            "fruit_count": len(tree.get("fruits", [])),
            "leaf_count": len(tree.get("leaves", [])),
        }

    try:
        asset_manifest = json.loads((isaac_dir / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"seed": seed, "status": "build_failed", "error": f"Builder did not produce a valid manifest: {exc}"}

    return ready_record(seed, canonical, tree, asset_manifest)


def ready_record(seed, canonical, tree, asset_manifest):
    return {
        "seed": seed,
        "status": "ready",
        "tree_json": f"seed{seed}/tree.json",
        "isaac_directory": f"seed{seed}/isaac",
        "tree_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
        "tree_usd_sha256": asset_manifest["tree_sha256"],
        "selected_fruit_id": asset_manifest["selected_fruit_id"],
        "selected_fruit_wood_clearance_m": asset_manifest["selected_fruit_wood_clearance_m"],
        "branch_count": asset_manifest["branch_count"],
        "fruit_count": asset_manifest["fruit_count"],
        "leaf_count": asset_manifest["leaf_count"],
        "collider_branch_count": asset_manifest.get("collider_branch_count"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Pinned, clean OrchardBench checkout with its LICENSE file")
    parser.add_argument("--output", type=Path, required=True,
                        help="Pool output directory (for example outputs/orchardbench/tree_pool)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Seeds to prepare (default: 1 7 21 42 53 67 83 97)")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable with NumPy and pxr available")
    args = parser.parse_args(argv)

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"OrchardBench source directory does not exist: {source}")
    if not (source / "LICENSE").is_file():
        parser.error(f"OrchardBench LICENSE file is missing: {source / 'LICENSE'}")
    if not args.seeds or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique nonnegative integers")

    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "LICENSE", output / "LICENSE")
    results = []
    for seed in args.seeds:
        print(f"Preparing OrchardBench seed {seed}...", flush=True)
        item = prepare_seed(seed, source, output, args.python)
        results.append(item)
        print(f"  {item['status']}: {item.get('error', 'asset ready')}", flush=True)

    ready = sum(item["status"] == "ready" for item in results)
    pool_manifest = {
        "schema": "orchardbench_tree_pool_v1",
        "source": {
            "url": "https://github.com/humphreymunn/orchardbench",
            "revision": "6313313db8b1a7d23fb2cc3afd67cac46f29399a",
            "license": "Apache-2.0",
            "license_file": "LICENSE",
        },
        "seeds_requested": args.seeds,
        "ready_count": ready,
        "failed_count": len(results) - ready,
        "candidates": results,
    }
    (output / "manifest.json").write_text(json.dumps(pool_manifest, indent=2) + "\n")
    print(f"Pool manifest: {output / 'manifest.json'} ({ready}/{len(results)} ready)")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
