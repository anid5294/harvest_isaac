"""Simulator-independent validation/layout for an offline topology-bearing tree."""

import hashlib
import json
import math
from pathlib import Path

from .layout import OrchardLayout


def load_tree(path, seed=None):
    tree = json.loads(Path(path).read_text())
    if tree.get("units") != "m" or tree.get("up_axis") != "Z":
        raise ValueError("Expected unscaled metre, Z-up tree")
    if seed is not None and tree["seed"] != seed:
        raise ValueError("Tree seed differs from run seed; regenerate or select matching seed")
    branches = {b["id"]: b for b in tree["branches"]}
    if not branches or len(branches) != len(tree["branches"]):
        raise ValueError("Empty or duplicate branch IDs")
    for branch in branches.values():
        for key in ("start", "end"):
            if len(branch[key]) != 3 or not all(math.isfinite(v) for v in branch[key]):
                raise ValueError("Invalid branch coordinates")
        if math.dist(branch["start"], branch["end"]) <= 1e-8:
            raise ValueError("Zero-length branch")
        if not (math.isfinite(branch["radius_start"]) and math.isfinite(branch["radius_end"])
                and 0 < branch["radius_end"] <= branch["radius_start"] + 1e-12):
            raise ValueError("Invalid branch taper")
        parent = branch["parent_id"]
        if parent not in (None, -1):
            if parent not in branches:
                raise ValueError("Missing parent branch")
            if math.dist(branch["start"], branches[parent]["end"]) > 1e-6:
                raise ValueError("Disconnected branch graph")
        seen = {branch["id"]}
        while parent not in (None, -1):
            if parent in seen or parent not in branches:
                raise ValueError("Cyclic or incomplete graph")
            seen.add(parent)
            parent = branches[parent]["parent_id"]
    errors = []
    for fruit in tree["fruits"]:
        parent = branches[fruit["parent_branch_id"]]
        delta = [b-a for a, b in zip(parent["start"], parent["end"])]
        norm2 = sum(v*v for v in delta)
        t = sum((a-b)*d for a,b,d in zip(fruit["anchor"], parent["start"], delta))/norm2
        projected = [a+t*d for a,d in zip(parent["start"], delta)]
        residual = math.dist(projected, fruit["anchor"])
        if not -1e-6 <= t <= 1+1e-6 or residual > 1e-6:
            raise ValueError("Fruit anchor is not on its parent branch")
        if math.dist(fruit["center"], fruit["anchor"]) <= fruit["radius"]:
            raise ValueError("Fruit stem must have positive length")
        errors.append(residual)
    if not errors:
        raise ValueError("No fruit")
    tree["topology_validation"] = {"max_fruit_anchor_error_m": max(errors),
                                   "validated_branch_count": len(branches)}
    return tree


def interaction_layout(tree, fruit_id=None):
    """Choose existing fruit; never move/scale a tree to satisfy robot reach.

    This selection is a candidate only, not a reachability or grasp guarantee.
    Robot placement/control and collisions still require target-runtime review.
    """
    fruits = [f for f in tree["fruits"] if wood_clearance(tree, f) > 0]
    if not fruits:
        raise ValueError("No non-intersecting fruit available; do not move the tree to manufacture one")
    if fruit_id is None:
        fruit = min(fruits, key=lambda f: abs(f["center"][2]-1.05))
    else:
        fruit = next((f for f in fruits if str(f["id"]) == str(fruit_id)), None)
        if fruit is None:
            raise ValueError(f"Unknown fruit ID: {fruit_id}")
    x, y, _ = fruit["center"]
    stem = fruit["stem"]
    layout = OrchardLayout(
        seed=tree["seed"], apples=(tuple(fruit["center"]),), target_index=0,
        basket=(x-0.35, y-0.40, 0.78), radius=fruit["radius"],
        mass=stem["mass_kg"], stem_break_force=stem["break_force_n"],
        stem_anchors=(tuple(fruit["anchor"]),), tree_model="orchardbench",
    )
    return layout, fruit


def wood_clearance(tree, fruit):
    """Conservative sphere-to-tapered-wood clearance, in metres."""
    gaps = []
    for branch in tree["branches"]:
        start, end = branch["start"], branch["end"]
        delta = [b-a for a,b in zip(start,end)]
        t = sum((p-a)*d for p,a,d in zip(fruit["center"], start, delta))/sum(d*d for d in delta)
        point = [a+max(0,min(1,t))*d for a,d in zip(start,delta)]
        gaps.append(math.dist(fruit["center"],point)-fruit["radius"]-
                    max(branch["radius_start"],branch["radius_end"]))
    return min(gaps)


def load_asset(directory, seed):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "orchardbench_isaac_asset_v1" or manifest.get("seed") != seed:
        raise ValueError("Unsupported tree asset or seed mismatch")
    if manifest.get("tree_file") != "tree.usda":
        raise ValueError("Unexpected tree file")
    path = directory / "tree.usda"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["tree_sha256"]:
        raise ValueError("Tree USD hash mismatch; rebuild")
    canonical = directory.parent / "tree.json"
    if hashlib.sha256(canonical.read_bytes()).hexdigest() != manifest["canonical_sha256"]:
        raise ValueError("Canonical tree hash mismatch; rebuild")
    expected, _ = interaction_layout(load_tree(canonical, seed), manifest["selected_fruit_id"])
    if json.loads(json.dumps(expected.metadata())) != manifest["layout"]:
        raise ValueError("Physical layout does not match canonical fruit")
    layout = OrchardLayout(**manifest["layout"])
    if layout.tree_model != "orchardbench" or len(layout.apples) != 1 or layout.target_index != 0:
        raise ValueError("External tree must have exactly one dynamic target")
    return layout, path, manifest
