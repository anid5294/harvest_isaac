"""Validate a prepared visual tree without loading the simulator."""

import hashlib
import json
from pathlib import Path


def validate_tree_asset(directory, layout):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "tree.usda"
    expected_layout = json.loads(json.dumps(layout.metadata()))
    if manifest.get("layout") != expected_layout:
        raise ValueError("Tree asset layout differs from this run; regenerate for its seed")
    if manifest.get("tree_file") != "tree.usda":
        raise ValueError("Tree manifest must reference tree.usda")
    if manifest.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Tree asset SHA-256 mismatch; regenerate the asset")
    if manifest.get("collision_status") != "visual_only_existing_scaffold":
        raise ValueError("Unsupported tree collision profile")
    return path, manifest
