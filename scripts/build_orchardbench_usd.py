#!/usr/bin/env python3
"""Compile canonical OrchardBench tree to metre/Z-up USD.

Collision modes:

    none
        No branch collision geometry. Intended as a diagnostic to determine
        whether branch convex hulls are responsible for a PhysX failure.

    coarse
        Create collision only for sufficiently large branch segments. Fine
        twigs remain visible but are not separate PhysX convex hulls.

    all
        Preserve the original behavior: create an eight-sided convex-hull
        collider for every branch segment.

The chosen fruit is omitted from the static tree because Isaac creates it as
an independent dynamic body. Other fruit and foliage remain visual-only for
the current single-fruit harvesting gate.

This script can run in two environments:

1. A Python environment where pxr is already importable (for example Blender).
2. The Songkhla Isaac Lab Conda environment, where pxr becomes available only
   after Isaac/Kit is initialized through AppLauncher.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types


PROJECT = Path(__file__).resolve().parents[1]

DEFAULT_MIN_COLLIDER_RADIUS_M = 0.010
DEFAULT_MIN_COLLIDER_LENGTH_M = 0.040


def load_adapter():
    folder = PROJECT / "src/vla_isaaclab/envs/orchard_pick"

    package = types.ModuleType("_offline_orchard")
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package

    spec = importlib.util.spec_from_file_location(
        "_offline_orchard.external_tree",
        folder / "external_tree.py",
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load external tree adapter from {folder / 'external_tree.py'}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def segment_length(start, end):
    return math.sqrt(
        (end[0] - start[0]) ** 2
        + (end[1] - start[1]) ** 2
        + (end[2] - start[2]) ** 2
    )


def validate_branch(branch):
    start = branch["start"]
    end = branch["end"]
    r0 = float(branch["radius_start"])
    r1 = float(branch["radius_end"])

    values = [
        *[float(x) for x in start],
        *[float(x) for x in end],
        r0,
        r1,
    ]

    if not all(math.isfinite(x) for x in values):
        raise ValueError(
            f"Branch {branch['id']} contains non-finite geometry: {values}"
        )

    if r0 <= 0.0 or r1 <= 0.0:
        raise ValueError(
            f"Branch {branch['id']} has non-positive radius: "
            f"r0={r0}, r1={r1}"
        )

    length = segment_length(start, end)

    if not math.isfinite(length) or length <= 1.0e-6:
        raise ValueError(
            f"Branch {branch['id']} has invalid/near-zero length: {length}"
        )

    return length


def build(
    tree_path,
    output,
    fruit_id=None,
    collision_mode="coarse",
    min_collider_radius_m=DEFAULT_MIN_COLLIDER_RADIUS_M,
    min_collider_length_m=DEFAULT_MIN_COLLIDER_LENGTH_M,
):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    adapter = load_adapter()

    tree = adapter.load_tree(tree_path)
    layout, fruit = adapter.interaction_layout(tree, fruit_id)

    output.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output / "tree.usda"))

    if stage is None:
        raise RuntimeError(
            f"Failed to create USD stage at {output / 'tree.usda'}"
        )

    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/Tree")
    stage.SetDefaultPrim(root.GetPrim())

    def material(name, color):
        mat = UsdShade.Material.Define(
            stage,
            f"/Tree/Materials/{name}",
        )

        shader = UsdShade.Shader.Define(
            stage,
            mat.GetPath().AppendChild("Surface"),
        )

        shader.CreateIdAttr("UsdPreviewSurface")

        shader.CreateInput(
            "diffuseColor",
            Sdf.ValueTypeNames.Color3f,
        ).Set(
            Gf.Vec3f(*color)
        )

        shader.CreateInput(
            "roughness",
            Sdf.ValueTypeNames.Float,
        ).Set(0.8)

        mat.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(),
            "surface",
        )

        return mat

    wood = material(
        "Bark",
        (0.26, 0.13, 0.055),
    )

    leaf_materials = [
        material(f"Leaf{i}", color)
        for i, color in enumerate(
            (
                (0.08, 0.24, 0.025),
                (0.12, 0.32, 0.04),
                (0.19, 0.36, 0.045),
            )
        )
    ]

    def mesh(path, vertices, indices, mat=None):
        obj = UsdGeom.Mesh.Define(stage, path)

        obj.CreatePointsAttr(
            [
                Gf.Vec3f(*vertex)
                for vertex in vertices
            ]
        )

        obj.CreateFaceVertexCountsAttr(
            [3] * (len(indices) // 3)
        )

        obj.CreateFaceVertexIndicesAttr(indices)
        obj.CreateSubdivisionSchemeAttr("none")
        obj.CreateDoubleSidedAttr(True)

        if mat is not None:
            UsdShade.MaterialBindingAPI.Apply(
                obj.GetPrim()
            ).Bind(mat)

        return obj

    def tapered(start, end, r0, r1, sides):
        start_vec = Gf.Vec3d(*start)
        end_vec = Gf.Vec3d(*end)

        z = end_vec - start_vec
        length = z.GetLength()

        if not math.isfinite(length) or length <= 1.0e-9:
            raise ValueError(
                f"Cannot construct tapered segment with length {length}"
            )

        z.Normalize()

        ref = (
            Gf.Vec3d(0.0, 0.0, 1.0)
            if abs(z[2]) < 0.9
            else Gf.Vec3d(1.0, 0.0, 0.0)
        )

        x = Gf.Cross(ref, z).GetNormalized()
        y = Gf.Cross(z, x)

        points = [
            tuple(
                Gf.Vec3d(*point)
                + radius
                * (
                    math.cos(
                        2.0 * math.pi * i / sides
                    )
                    * x
                    + math.sin(
                        2.0 * math.pi * i / sides
                    )
                    * y
                )
            )
            for point, radius in (
                (start, r0),
                (end, r1),
            )
            for i in range(sides)
        ]

        faces = []

        for i in range(sides):
            j = (i + 1) % sides

            faces += [
                i,
                j,
                sides + i,
                j,
                sides + j,
                sides + i,
            ]

        for i in range(1, sides - 1):
            faces += [
                0,
                i + 1,
                i,
                sides,
                sides + i,
                sides + i + 1,
            ]

        return points, faces

    collider_branch_ids = []
    skipped_collider_branch_ids = []
    branch_diagnostics = []

    #
    # Wood
    #
    for branch in tree["branches"]:
        length = validate_branch(branch)

        start = branch["start"]
        end = branch["end"]
        r0 = float(branch["radius_start"])
        r1 = float(branch["radius_end"])

        # Visual geometry is always preserved.
        mesh(
            f"/Tree/Wood/B{branch['id']}",
            *tapered(
                start,
                end,
                r0,
                r1,
                16,
            ),
            wood,
        )

        max_radius = max(r0, r1)
        min_radius = min(r0, r1)

        if collision_mode == "none":
            create_collider = False

        elif collision_mode == "coarse":
            create_collider = (
                length >= min_collider_length_m
                and max_radius >= min_collider_radius_m
            )

        elif collision_mode == "all":
            create_collider = True

        else:
            raise ValueError(
                f"Unknown collision mode: {collision_mode}"
            )

        branch_diagnostics.append(
            {
                "id": branch["id"],
                "length_m": length,
                "radius_start_m": r0,
                "radius_end_m": r1,
                "min_radius_m": min_radius,
                "max_radius_m": max_radius,
                "aspect_ratio_length_over_max_radius": (
                    length / max_radius
                ),
                "collider": create_collider,
            }
        )

        if not create_collider:
            skipped_collider_branch_ids.append(
                branch["id"]
            )
            continue

        collider = mesh(
            f"/Tree/Colliders/B{branch['id']}",
            *tapered(
                start,
                end,
                r0,
                r1,
                8,
            ),
        )

        collider.CreateVisibilityAttr(
            "invisible"
        )

        UsdPhysics.CollisionAPI.Apply(
            collider.GetPrim()
        )

        UsdPhysics.MeshCollisionAPI.Apply(
            collider.GetPrim()
        ).CreateApproximationAttr(
            "convexHull"
        )

        collider_branch_ids.append(
            branch["id"]
        )

    #
    # Leaves are visual-only.
    #
    source_meshes = {
        item["class"]: item
        for item in tree["leaf_meshes"]
    }

    for i, leaf in enumerate(
        tree["leaves"]
    ):
        source = source_meshes[
            leaf["mesh_class"]
        ]

        q = leaf[
            "orientation_xyzw"
        ]

        rotation = Gf.Rotation(
            Gf.Quatd(
                q[3],
                Gf.Vec3d(
                    *q[:3]
                ),
            )
        )

        points = [
            tuple(
                rotation.TransformDir(
                    Gf.Vec3d(*vertex)
                )
                + Gf.Vec3d(
                    *leaf["position"]
                )
            )
            for vertex in source[
                "vertices"
            ]
        ]

        mesh(
            f"/Tree/Leaves/L{i}",
            points,
            source[
                "triangle_indices"
            ],
            leaf_materials[
                i % len(
                    leaf_materials
                )
            ],
        )

    #
    # All fruit except the selected target remains visual-only.
    # The selected target fruit is created dynamically by the Isaac task.
    #
    for item in tree["fruits"]:
        if item["id"] == fruit["id"]:
            continue

        sphere = UsdGeom.Sphere.Define(
            stage,
            f"/Tree/VisualFruit/F{item['id']}",
        )

        sphere.CreateRadiusAttr(
            item["radius"]
        )

        sphere.AddTranslateOp().Set(
            Gf.Vec3d(
                *item["center"]
            )
        )

        UsdShade.MaterialBindingAPI.Apply(
            sphere.GetPrim()
        ).Bind(
            material(
                f"Fruit{item['id']}",
                item["color"],
            )
        )

        bottom = list(
            item["center"]
        )

        bottom[2] += item["radius"]

        mesh(
            f"/Tree/VisualStems/S{item['id']}",
            *tapered(
                bottom,
                item["anchor"],
                0.0015,
                0.0015,
                6,
            ),
            wood,
        )

    stage.GetRootLayer().Save()

    #
    # Collision-review USD.
    #
    # This references the exact generated collision geometry rather than
    # constructing a second approximation.
    #
    debug = Usd.Stage.CreateNew(
        str(
            output
            / "collision_review.usda"
        )
    )

    if debug is None:
        raise RuntimeError(
            "Failed to create collision_review.usda"
        )

    UsdGeom.SetStageMetersPerUnit(
        debug,
        1.0,
    )

    UsdGeom.SetStageUpAxis(
        debug,
        UsdGeom.Tokens.z,
    )

    debug_root = debug.DefinePrim(
        "/Tree",
        "Xform",
    )

    debug_root.GetReferences().AddReference(
        "./tree.usda"
    )

    debug.SetDefaultPrim(
        debug_root
    )

    for branch_id in collider_branch_ids:
        collider_prim = debug.GetPrimAtPath(
            f"/Tree/Colliders/B{branch_id}"
        )

        if collider_prim.IsValid():
            UsdGeom.Imageable(
                collider_prim
            ).GetVisibilityAttr().Set(
                "inherited"
            )

        wood_prim = debug.GetPrimAtPath(
            f"/Tree/Wood/B{branch_id}"
        )

        if wood_prim.IsValid():
            UsdGeom.Imageable(
                wood_prim
            ).GetVisibilityAttr().Set(
                "invisible"
            )

    debug.GetRootLayer().Save()

    #
    # Collision diagnostics.
    #
    (
        output
        / "branch_collision_diagnostics.json"
    ).write_text(
        json.dumps(
            branch_diagnostics,
            indent=2,
        )
        + "\n"
    )

    if collision_mode == "none":
        collision_description = (
            "no static branch colliders; "
            "visual-only tree collision ablation"
        )

    elif collision_mode == "coarse":
        collision_description = (
            "static eight-sided tapered convex hulls for "
            f"segments with length >= {min_collider_length_m:.4f} m "
            f"and max radius >= {min_collider_radius_m:.4f} m; "
            "smaller twigs visual-only"
        )

    else:
        collision_description = (
            "static eight-sided tapered convex hull per segment; "
            "original collision behavior"
        )

    manifest = {
        "schema": (
            "orchardbench_isaac_asset_v1"
        ),
        "tree_file": "tree.usda",
        "tree_sha256": hashlib.sha256(
            (
                output
                / "tree.usda"
            ).read_bytes()
        ).hexdigest(),
        "canonical_sha256": hashlib.sha256(
            Path(
                tree_path
            ).read_bytes()
        ).hexdigest(),
        "source": tree["source"],
        "seed": tree["seed"],
        "selected_fruit_id": (
            fruit["id"]
        ),
        "layout": layout.metadata(),
        "topology_validation": (
            tree[
                "topology_validation"
            ]
        ),
        "branch_count": len(
            tree["branches"]
        ),
        "fruit_count": len(
            tree["fruits"]
        ),
        "leaf_count": len(
            tree["leaves"]
        ),
        "selected_fruit_wood_clearance_m": (
            adapter.wood_clearance(
                tree,
                fruit,
            )
        ),
        "collision_mode": (
            collision_mode
        ),
        "collision": (
            collision_description
        ),
        "collider_branch_count": len(
            collider_branch_ids
        ),
        "skipped_collider_branch_count": len(
            skipped_collider_branch_ids
        ),
        "min_collider_radius_m": (
            min_collider_radius_m
        ),
        "min_collider_length_m": (
            min_collider_length_m
        ),
        "branch_collision_diagnostics": (
            "branch_collision_diagnostics.json"
        ),
        "dynamic_fruit_count": 1,
    }

    (
        output
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )


def make_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--tree",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fruit-id",
    )

    parser.add_argument(
        "--collision-mode",
        choices=(
            "none",
            "coarse",
            "all",
        ),
        default="coarse",
        help=(
            "Tree collision representation. "
            "'none' disables branch colliders; "
            "'coarse' excludes small twigs; "
            "'all' preserves the original collider behavior."
        ),
    )

    parser.add_argument(
        "--min-collider-radius-m",
        type=float,
        default=DEFAULT_MIN_COLLIDER_RADIUS_M,
        help=(
            "For --collision-mode coarse, skip branch "
            "segments whose maximum endpoint radius is "
            "below this value."
        ),
    )

    parser.add_argument(
        "--min-collider-length-m",
        type=float,
        default=DEFAULT_MIN_COLLIDER_LENGTH_M,
        help=(
            "For --collision-mode coarse, skip branch "
            "segments shorter than this value."
        ),
    )

    return parser


def main():
    #
    # Blender/local USD Python already exposes pxr.
    #
    # Songkhla's Conda environment does not expose pxr until the Isaac/Kit
    # application has initialized its extension/runtime paths.
    #
    try:
        import pxr  # noqa: F401

        pxr_already_available = True

    except ModuleNotFoundError:
        pxr_already_available = False

    app = None

    parser = make_parser()

    if not pxr_already_available:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(
            parser
        )

    args = parser.parse_args()

    if (
        args.min_collider_radius_m
        <= 0.0
    ):
        parser.error(
            "--min-collider-radius-m must be > 0"
        )

    if (
        args.min_collider_length_m
        <= 0.0
    ):
        parser.error(
            "--min-collider-length-m must be > 0"
        )

    try:
        if not pxr_already_available:
            #
            # Match the existing project's Isaac-side asset scripts:
            # initialize a minimal headless Kit application before importing
            # USD/PXR modules.
            #
            args.headless = True
            args.enable_cameras = False
            args.experience = str(
                PROJECT
                / "configs/ycb.python.headless.kit"
            )

            app = AppLauncher(
                args
            ).app

            #
            # Explicitly verify that initialization made pxr available.
            #
            from pxr import Usd  # noqa: F401

        build(
            tree_path=args.tree,
            output=args.output,
            fruit_id=args.fruit_id,
            collision_mode=(
                args.collision_mode
            ),
            min_collider_radius_m=(
                args.min_collider_radius_m
            ),
            min_collider_length_m=(
                args.min_collider_length_m
            ),
        )

    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
