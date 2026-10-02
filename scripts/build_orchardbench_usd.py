#!/usr/bin/env python3
"""Compile canonical tree to metre/Z-up USD with separate rigid wood colliders.

Requires pxr (Isaac Python or Blender's bundled Python). No simulator is started.
The chosen fruit is omitted: Isaac creates it as an independent dynamic body.
Other fruit and foliage are explicitly visual-only in this single-fruit gate.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types


def load_adapter():
    folder = Path(__file__).resolve().parents[1] / "src/vla_isaaclab/envs/orchard_pick"
    package = types.ModuleType("_offline_orchard")
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location("_offline_orchard.external_tree", folder / "external_tree.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(tree_path, output, fruit_id=None):
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
        mat = UsdShade.Material.Define(stage, f"/Tree/Materials/{name}")
        shader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Surface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    wood = material("Bark", (0.26, 0.13, 0.055))
    leaf_materials = [material(f"Leaf{i}", c) for i,c in enumerate(
        ((0.08,0.24,0.025), (0.12,0.32,0.04), (0.19,0.36,0.045)))]

    def mesh(path, vertices, indices, mat=None):
        obj = UsdGeom.Mesh.Define(stage, path)
        obj.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
        obj.CreateFaceVertexCountsAttr([3]*(len(indices)//3))
        obj.CreateFaceVertexIndicesAttr(indices)
        obj.CreateSubdivisionSchemeAttr("none")
        obj.CreateDoubleSidedAttr(True)
        if mat:
            UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)
        return obj

    def tapered(start, end, r0, r1, sides):
        z = Gf.Vec3d(*end)-Gf.Vec3d(*start)
        z.Normalize()
        ref = Gf.Vec3d(0,0,1) if abs(z[2]) < .9 else Gf.Vec3d(1,0,0)
        x = Gf.Cross(ref,z).GetNormalized()
        y = Gf.Cross(z,x)
        points = [tuple(Gf.Vec3d(*p)+r*(math.cos(2*math.pi*i/sides)*x+
                  math.sin(2*math.pi*i/sides)*y))
                  for p,r in ((start,r0),(end,r1)) for i in range(sides)]
        faces = []
        for i in range(sides):
            j=(i+1)%sides
            faces += [i,j,sides+i, j,sides+j,sides+i]
        for i in range(1,sides-1):
            faces += [0,i+1,i, sides,sides+i,sides+i+1]
        return points, faces

    for b in tree["branches"]:
        args = (b["start"], b["end"], b["radius_start"], b["radius_end"])
        mesh(f"/Tree/Wood/B{b['id']}", *tapered(*args,16), wood)
        collider = mesh(f"/Tree/Colliders/B{b['id']}", *tapered(*args,8))
        collider.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("convexHull")
    source_meshes = {m["class"]:m for m in tree["leaf_meshes"]}
    for i,leaf in enumerate(tree["leaves"]):
        source = source_meshes[leaf["mesh_class"]]
        q=leaf["orientation_xyzw"]
        rotation=Gf.Rotation(Gf.Quatd(q[3],Gf.Vec3d(*q[:3])))
        points=[tuple(rotation.TransformDir(Gf.Vec3d(*v))+Gf.Vec3d(*leaf["position"]))
                for v in source["vertices"]]
        mesh(f"/Tree/Leaves/L{i}", points, source["triangle_indices"], leaf_materials[i%3])
    for f in tree["fruits"]:
        if f["id"] == fruit["id"]:
            continue
        sphere = UsdGeom.Sphere.Define(stage, f"/Tree/VisualFruit/F{f['id']}")
        sphere.CreateRadiusAttr(f["radius"])
        sphere.AddTranslateOp().Set(Gf.Vec3d(*f["center"]))
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material(f"Fruit{f['id']}",f["color"]))
        bottom=list(f["center"])
        bottom[2]+=f["radius"]
        mesh(f"/Tree/VisualStems/S{f['id']}", *tapered(bottom,f["anchor"],.0015,.0015,6), wood)
    stage.GetRootLayer().Save()
    # Diagnostic layer: the same collision hulls made visible, not a second
    # independently generated approximation. Never load this as the task asset.
    debug = Usd.Stage.CreateNew(str(output / "collision_review.usda"))
    UsdGeom.SetStageMetersPerUnit(debug, 1.0)
    UsdGeom.SetStageUpAxis(debug, UsdGeom.Tokens.z)
    debug_root = debug.DefinePrim("/Tree", "Xform")
    debug_root.GetReferences().AddReference("./tree.usda")
    debug.SetDefaultPrim(debug_root)
    for b in tree["branches"]:
        UsdGeom.Imageable(debug.GetPrimAtPath(f"/Tree/Colliders/B{b['id']}")).GetVisibilityAttr().Set("inherited")
        UsdGeom.Imageable(debug.GetPrimAtPath(f"/Tree/Wood/B{b['id']}")).GetVisibilityAttr().Set("invisible")
    debug.GetRootLayer().Save()
    manifest={"schema":"orchardbench_isaac_asset_v1", "tree_file":"tree.usda",
              "tree_sha256":hashlib.sha256((output/"tree.usda").read_bytes()).hexdigest(),
              "canonical_sha256":hashlib.sha256(Path(tree_path).read_bytes()).hexdigest(),
              "source":tree["source"],"seed":tree["seed"],"selected_fruit_id":fruit["id"],
              "layout":layout.metadata(),"topology_validation":tree["topology_validation"],
              "branch_count":len(tree["branches"]),"fruit_count":len(tree["fruits"]),
              "leaf_count":len(tree["leaves"]),
              "selected_fruit_wood_clearance_m":adapter.wood_clearance(tree,fruit),
              "collision":"static eight-sided tapered convex hull per segment; visual-only leaves and other fruit",
              "dynamic_fruit_count":1}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tree",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fruit-id")
    a=p.parse_args()
    build(a.tree,a.output,a.fruit_id)
