"""Build the YCB apple assets used by the environment (no simulator needed).

1. Converts YCB ``013_apple`` (google_16k: textured.obj + texture_map.png) to USD:
   ``assets/ycb_013_apple/apple_visual.usda``  - textured mesh, origin at its bounding-box
   centre, Z up (stem up), no physics;
   ``assets/ycb_013_apple/apple_physics.usda`` - the same mesh as a rigid body with a
   convex-hull collider and harvest_isaac's apple friction (1.4 / 1.1, restitution 0).
2. Writes ``tree_ycb.usda`` next to the OrchardBench ``tree.usda``: every visual fruit sphere is
   replaced by the YCB apple at real size, hanging from its tree.json stem anchor with the
   stem pointing at the anchor, with a random spin about its own axis. ``tree.usda`` is kept.

Download 013_apple_google_16k.tgz from the YCB benchmark site, extract it, then:

    C:\\isaac\\env_isaaclab3\\Scripts\\python.exe scripts\\build_ycb_apple.py <path>\\013_apple\\google_16k
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[1] / "harvest_rl" / "assets"
OUT = ROOT / "ycb_013_apple"
TREE_DIR = ROOT / "tree_orchardbench_seed42"


def read_obj(path):
    verts, uvs, normals, faces = [], [], [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append(tuple(float(x) for x in parts[1:4]))
        elif parts[0] == "vt":
            uvs.append(tuple(float(x) for x in parts[1:3]))
        elif parts[0] == "vn":
            normals.append(tuple(float(x) for x in parts[1:4]))
        elif parts[0] == "f":
            faces.append([tuple(int(i) - 1 for i in corner.split("/")) for corner in parts[1:]])
    return verts, uvs, normals, faces


def new_stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def build_visual(ycb_dir):
    verts, uvs, normals, faces = read_obj(Path(ycb_dir) / "textured.obj")
    lo = [min(v[i] for v in verts) for i in range(3)]
    hi = [max(v[i] for v in verts) for i in range(3)]
    center = [(a + b) / 2 for a, b in zip(lo, hi)]
    half = [(b - a) / 2 for a, b in zip(lo, hi)]
    shutil.copy(Path(ycb_dir) / "texture_map.png", OUT / "texture_map.png")

    stage = new_stage(OUT / "apple_visual.usda")
    root = UsdGeom.Xform.Define(stage, "/Apple")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Apple/Mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*(v[i] - center[i] for i in range(3))) for v in verts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in faces]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([c[0] for f in faces for c in f]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*(-h for h in half)), Gf.Vec3f(*half)]))
    pv = UsdGeom.PrimvarsAPI(mesh)
    st = pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in uvs]))
    st.SetIndices(Vt.IntArray([c[1] for f in faces for c in f]))
    nrm = pv.CreatePrimvar("normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.faceVarying)
    nrm.Set(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
    nrm.SetIndices(Vt.IntArray([c[2] for f in faces for c in f]))

    material = UsdShade.Material.Define(stage, "/Apple/Looks/AppleSkin")
    surface = UsdShade.Shader.Define(stage, "/Apple/Looks/AppleSkin/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    reader = UsdShade.Shader.Define(stage, "/Apple/Looks/AppleSkin/StReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, "/Apple/Looks/AppleSkin/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./texture_map.png")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb", UsdShade.AttributeType.Output)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    return half


def build_physics():
    stage = new_stage(OUT / "apple_physics.usda")
    root = stage.DefinePrim("/Apple", "Xform")
    root.GetReferences().AddReference("./apple_visual.usda")
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root)
    mesh = stage.OverridePrim("/Apple/Mesh")
    UsdPhysics.CollisionAPI.Apply(mesh)
    UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    phys = UsdShade.Material.Define(stage, "/Apple/PhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(phys.GetPrim())
    api.CreateStaticFrictionAttr(1.4)
    api.CreateDynamicFrictionAttr(1.1)
    api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()


def build_tree(half, seed=42):
    """Copy tree.usda to tree_ycb.usda with every visual fruit sphere swapped for the YCB apple."""
    tree = json.loads((TREE_DIR / "tree.json").read_text())
    fruits = {int(f["id"]): f for f in tree["fruits"]}
    src = TREE_DIR / "isaac" / "tree.usda"
    dst = TREE_DIR / "isaac" / "tree_ycb.usda"
    shutil.copy(src, dst)
    stage = Usd.Stage.Open(str(dst))
    rng = random.Random(seed)
    apple_ref = Path("../../ycb_013_apple/apple_visual.usda").as_posix()
    swapped = 0
    for prim in list(stage.GetPrimAtPath("/Tree/VisualFruit").GetChildren()):
        fid = int(prim.GetName()[1:])
        fruit = fruits[fid]
        old_center = prim.GetAttribute("xformOp:translate").Get()
        if (Gf.Vec3d(*fruit["center"]) - old_center).GetLength() > 1e-6:
            raise RuntimeError(f"{prim.GetPath()} does not match tree.json fruit {fid}")
        anchor = Gf.Vec3d(*fruit["anchor"])
        down = (Gf.Vec3d(*fruit["center"]) - anchor).GetNormalized()
        center = anchor + down * (fruit["stem"]["length"] + half[2])
        align = Gf.Rotation(Gf.Vec3d(0, 0, 1), -down)  # apple +Z (stem) points at the anchor
        spin = Gf.Rotation(Gf.Vec3d(0, 0, 1), rng.uniform(0.0, 360.0))
        q = (spin * align).GetQuat()
        path = prim.GetPath()
        stage.RemovePrim(path)
        xform = UsdGeom.Xform.Define(stage, path)
        xform.GetPrim().GetReferences().AddReference(apple_ref)
        xform.AddTranslateOp().Set(center)
        xform.AddOrientOp().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
        swapped += 1
    stage.GetRootLayer().Save()
    return swapped


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ycb_dir", help="Path to the extracted 013_apple/google_16k folder.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    half = build_visual(args.ycb_dir)
    build_physics()
    (OUT / "half_extents.json").write_text(json.dumps({"half_extents_m": half}, indent=1))
    swapped = build_tree(half)
    print(f"apple half extents (m): {[round(h, 4) for h in half]}; swapped {swapped} visual fruits")


if __name__ == "__main__":
    main()
