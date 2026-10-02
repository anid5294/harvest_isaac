#!/usr/bin/env python3
"""Compile canonical tree to metre/Z-up USD.

Collision modes:

    none
        No branch collision geometry. Use this to determine whether the
        OrchardBench branch colliders are responsible for PhysX GPU crashes.

    coarse
        Generate collision only for sufficiently large branch segments.
        This is the recommended default for the current harvesting milestone.

    all
        Preserve the original behavior: an eight-sided convex hull collider
        for every branch segment.

Requires pxr (Isaac Python or Blender's bundled Python). No simulator is started.

The chosen fruit is omitted from the static tree because Isaac creates it as an
independent dynamic body. Other fruit and foliage remain visual-only in the
single-fruit harvesting gate.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types


# Conservative initial physics LOD thresholds.
#
# These are deliberately larger than the smallest visible OrchardBench twigs.
# Thin twigs still render, but they do not become separate PhysX convex hulls.
DEFAULT_MIN_COLLIDER_RADIUS_M = 0.010
DEFAULT_MIN_COLLIDER_LENGTH_M = 0.040


def load_adapter():
    folder = (
        Path(__file__).resolve().parents[1]
        / "src/vla_isaaclab/envs/orchard_pick"
    )

    package = types.ModuleType("_offline_orchard")
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package

    spec = importlib.util.spec_from_file_location(
        "_offline_orchard.external_tree",
        folder / "external_tree.py",
    )
    module = importlib.util.module_from_spec(spec)
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
    r0 = branch["radius_start"]
    r1 = branch["radius_end"]

    values = [*start, *end, r0, r1]

    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError(
            f"Branch {branch['id']} contains non-finite geometry: {values}"
        )

    if r0 <= 0 or r1 <= 0:
        raise ValueError(
            f"Branch {branch['id']} has non-positive radius: "
            f"r0={r0}, r1={r1}"
        )

    length = segment_length(start, end)

    if length <= 1e-6:
        raise ValueError(
            f"Branch {branch['id']} has near-zero length: {length}"
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
        ).Set(Gf.Vec3f(*color))

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
        material(f"Leaf{i}", c)
        for i, c in enumerate(
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
            [Gf.Vec3f(*v) for v in vertices]
        )

        obj.CreateFaceVertexCountsAttr(
            [3] * (len(indices) // 3)
        )

        obj.CreateFaceVertexIndicesAttr(indices)
        obj.CreateSubdivisionSchemeAttr("none")
        obj.CreateDoubleSidedAttr(True)

        if mat:
            UsdShade.MaterialBindingAPI.Apply(
                obj.GetPrim()
            ).Bind(mat)

        return obj

    def tapered(start, end, r0, r1, sides):
        z = Gf.Vec3d(*end) - Gf.Vec3d(*start)

        length = z.GetLength()
        if length <= 1e-9:
            raise ValueError(
                f"Cannot construct tapered segment with length {length}"
            )

        z.Normalize()

        ref = (
            Gf.Vec3d(0, 0, 1)
            if abs(z[2]) < 0.9
            else Gf.Vec3d(1, 0, 0)
        )

        x = Gf.Cross(ref, z).GetNormalized()
        y = Gf.Cross(z, x)

        points = [
            tuple(
                Gf.Vec3d(*p)
                + r
                * (
                    math.cos(2 * math.pi * i / sides) * x
                    + math.sin(2 * math.pi * i / sides) * y
                )
            )
            for p, r in ((start, r0), (end, r1))
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

    for b in tree["branches"]:
        length = validate_branch(b)

        start = b["start"]
        end = b["end"]
        r0 = float(b["radius_start"])
        r1 = float(b["radius_end"])

        # Always generate the visual branch.
        visual_args = (
            start,
            end,
            r0,
            r1,
        )

        mesh(
            f"/Tree/Wood/B{b['id']}",
            *tapered(*visual_args, 16),
            wood,
        )

        max_radius = max(r0, r1)
        min_radius = min(r0, r1)

        create_collider = False

        if collision_mode == "all":
            create_collider = True

        elif collision_mode == "coarse":
            create_collider = (
                length >= min_collider_length_m
                and max_radius >= min_collider_radius_m
            )

        elif collision_mode == "none":
            create_collider = False

        else:
            raise ValueError(
                f"Unknown collision mode: {collision_mode}"
            )

        diagnostic = {
            "id": b["id"],
            "length_m": length,
            "radius_start_m": r0,
            "radius_end_m": r1,
            "max_radius_m": max_radius,
            "min_radius_m": min_radius,
            "aspect_ratio": length / max_radius,
            "collider": create_collider,
        }

        branch_diagnostics.append(diagnostic)

        if not create_collider:
            skipped_collider_branch_ids.append(b["id"])
            continue

        collider = mesh(
            f"/Tree/Colliders/B{b['id']}",
            *tapered(
                start,
                end,
                r0,
                r1,
                8,
            ),
        )

        collider.CreateVisibilityAttr("invisible")

        UsdPhysics.CollisionAPI.Apply(
            collider.GetPrim()
        )

        UsdPhysics.MeshCollisionAPI.Apply(
            collider.GetPrim()
        ).CreateApproximationAttr("convexHull")

        collider_branch_ids.append(b["id"])

    source_meshes = {
        m["class"]: m
        for m in tree["leaf_meshes"]
    }

    for i, leaf in enumerate(tree["leaves"]):
        source = source_meshes[leaf["mesh_class"]]

        q = leaf["orientation_xyzw"]

        rotation = Gf.Rotation(
            Gf.Quatd(
                q[3],
                Gf.Vec3d(*q[:3]),
            )
        )

        points = [
            tuple(
                rotation.TransformDir(
                    Gf.Vec3d(*v)
                )
                + Gf.Vec3d(*leaf["position"])
            )
            for v in source["vertices"]
        ]

        mesh(
            f"/Tree/Leaves/L{i}",
            points,
            source["triangle_indices"],
            leaf_materials[i % 3],
        )

    for f in tree["fruits"]:
        # Selected fruit is created dynamically by the Isaac task.
        if f["id"] == fruit["id"]:
            continue

        sphere = UsdGeom.Sphere.Define(
            stage,
            f"/Tree/VisualFruit/F{f['id']}",
        )

        sphere.CreateRadiusAttr(f["radius"])
        sphere.AddTranslateOp().Set(
            Gf.Vec3d(*f["center"])
        )

        UsdShade.MaterialBindingAPI.Apply(
            sphere.GetPrim()
        ).Bind(
            material(
                f"Fruit{f['id']}",
                f["color"],
            )
        )

        bottom = list(f["center"])
        bottom[2] += f["radius"]

        mesh(
            f"/Tree/VisualStems/S{f['id']}",
            *tapered(
                bottom,
                f["anchor"],
                0.0015,
                0.0015,
                6,
            ),
            wood,
        )

    stage.GetRootLayer().Save()

    #
    # Diagnostic collision-review layer.
    #
    # It references the same tree.usda and simply makes existing colliders
    # visible while hiding visual wood.
    #
    debug = Usd.Stage.CreateNew(
        str(output / "collision_review.usda")
    )

    UsdGeom.SetStageMetersPerUnit(debug, 1.0)
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

    debug.SetDefaultPrim(debug_root)

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

    diagnostics_path = (
        output / "branch_collision_diagnostics.json"
    )

    diagnostics_path.write_text(
        json.dumps(
            branch_diagnostics,
            indent=2,
        )
        + "\n"
    )

    if collision_mode == "none":
        collision_description = (
            "no static branch colliders; "
            "visual-only tree for collision ablation"
        )

    elif collision_mode == "coarse":
        collision_description = (
            "static eight-sided tapered convex hulls only for "
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
        "schema": "orchardbench_isaac_asset_v1",
        "tree_file": "tree.usda",
        "tree_sha256": hashlib.sha256(
            (output / "tree.usda").read_bytes()
        ).hexdigest(),
        "canonical_sha256": hashlib.sha256(
            Path(tree_path).read_bytes()
        ).hexdigest(),
        "source": tree["source"],
        "seed": tree["seed"],
        "selected_fruit_id": fruit["id"],
        "layout": layout.metadata(),
        "topology_validation": tree[
            "topology_validation"
        ],
        "branch_count": len(tree["branches"]),
        "fruit_count": len(tree["fruits"]),
        "leaf_count": len(tree["leaves"]),
        "selected_fruit_wood_clearance_m": (
            adapter.wood_clearance(
                tree,
                fruit,
            )
        ),
        "collision_mode": collision_mode,
        "collision": collision_description,
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

    (output / "manifest.json").write_text(
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
    )

    p.add_argument(
        "--tree",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--fruit-id",
    )

    p.add_argument(
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
            "'all' preserves the old behavior."
        ),
    )

    p.add_argument(
        "--min-collider-radius-m",
        type=float,
        default=DEFAULT_MIN_COLLIDER_RADIUS_M,
        help=(
            "For --collision-mode coarse, skip branch "
            "segments whose maximum radius is below "
            "this value."
        ),
    )

    p.add_argument(
        "--min-collider-length-m",
        type=float,
        default=DEFAULT_MIN_COLLIDER_LENGTH_M,
        help=(
            "For --collision-mode coarse, skip branch "
            "segments shorter than this value."
        ),
    )

    a = p.parse_args()

    build(
        tree_path=a.tree,
        output=a.output,
        fruit_id=a.fruit_id,
        collision_mode=a.collision_mode,
        min_collider_radius_m=(
            a.min_collider_radius_m
        ),
        min_collider_length_m=(
            a.min_collider_length_m
        ),
    )