#!/usr/bin/env python3
"""Render canonical OrchardBench tree JSON in Blender, at its source scale.

Run with Blender's Python, for example:
  Blender --background --factory-startup --python scripts/render_tree_review.py -- \
    --tree outputs/orchardbench/seed42/tree.json --output outputs/orchardbench/seed42/review \
    --robot-usd assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd

This is a visual geometry gate. It does not simulate fruit contact or Isaac Lab.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Quaternion, Vector


def arguments():
    cli = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-usd", type=Path)
    parser.add_argument("--exclude-fruit-id", help="Omit one fruit from visual export/review")
    parser.add_argument("--resolution", type=int, default=640)
    return parser.parse_args(cli)


def material(name, color, roughness=0.8):
    item = bpy.data.materials.new(name)
    item.diffuse_color = (*color[:3], 1.0)
    item.use_nodes = True
    bsdf = item.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
    return item


def mesh_object(name, vertices, faces, mats, indices=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for mat in mats:
        obj.data.materials.append(mat)
    if indices is not None:
        for polygon, idx in zip(mesh.polygons, indices):
            polygon.material_index = idx
    return obj


def tube(vertices, faces, start, end, radius0, radius1, sides=10):
    start, end = Vector(start), Vector(end)
    axis = end - start
    if axis.length < 1e-8:
        return
    axis.normalize()
    reference = Vector((0, 0, 1)) if abs(axis.z) < 0.9 else Vector((0, 1, 0))
    u = axis.cross(reference).normalized()
    v = axis.cross(u).normalized()
    base = len(vertices)
    for point, radius in ((start, radius0), (end, radius1)):
        for j in range(sides):
            direction = u * math.cos(2 * math.pi * j / sides) + v * math.sin(2 * math.pi * j / sides)
            vertices.append(tuple(point + radius * direction))
    for j in range(sides):
        next_j = (j + 1) % sides
        faces.append((base + j, base + next_j, base + sides + next_j, base + sides + j))
    for j in range(1, sides - 1):
        faces.append((base, base + j + 1, base + j))
        faces.append((base + sides, base + sides + j, base + sides + j + 1))


def branches(data, wood):
    verts, faces = [], []
    for segment in data["branches"]:
        tube(verts, faces, segment["start"], segment["end"],
             float(segment["radius_start"]), float(segment["radius_end"]))
    return mesh_object("OrchardBench wood (source branch graph)", verts, faces, [wood])


def leaves(data):
    # The canonical JSON contains OrchardBench's three exact source-generated
    # mesh classes, so use their vertices and triangle order without re-meshing.
    verts, faces, indices = [], [], []
    leaf_mats = []
    classes = {item["class"]: item for item in data["leaf_meshes"]}
    for index, leaf in enumerate(data["leaves"]):
        px, py, pz = leaf["position"]
        qx, qy, qz, qw = leaf["orientation_xyzw"]
        rotation = Quaternion((qw, qx, qy, qz))
        source = classes[leaf["mesh_class"]]
        base = len(verts)
        for point in source["vertices"]:
            verts.append(tuple(Vector((px, py, pz)) + rotation @ Vector(point)))
        triangles = source["triangle_indices"]
        for k in range(0, len(triangles), 3):
            faces.append(tuple(base + triangles[k + offset] for offset in range(3)))
            indices.append(index)
        leaf_mats.append(material(f"Leaf {leaf['id']}", leaf["color"]))
    return mesh_object("OrchardBench leaves (source mesh classes/transforms)",
                       verts, faces, leaf_mats, indices)


def sphere(vertices, faces, center, radius, lat=10, lon=16):
    base = len(vertices)
    center = Vector(center)
    for i in range(lat + 1):
        phi = math.pi * i / lat
        for j in range(lon):
            theta = 2 * math.pi * j / lon
            point = center + radius * Vector((math.sin(phi) * math.cos(theta),
                                               math.sin(phi) * math.sin(theta), math.cos(phi)))
            vertices.append(tuple(point))
    for i in range(lat):
        for j in range(lon):
            next_j = (j + 1) % lon
            faces.append((base + i * lon + j, base + i * lon + next_j,
                          base + (i + 1) * lon + next_j, base + (i + 1) * lon + j))


def fruits(data, wood, excluded):
    shown = [fruit for fruit in data["fruits"] if str(fruit["id"]) != excluded]
    stem_verts, stem_faces, fruit_verts, fruit_faces, face_indices = [], [], [], [], []
    fruit_mats = []
    for fruit in shown:
        center, anchor = fruit["center"], fruit["anchor"]
        radius = float(fruit["radius"])
        # Source uses spherical apples suspended by a short pedicel.
        before = len(fruit_faces)
        sphere(fruit_verts, fruit_faces, center, radius)
        fruit_mats.append(material(f"Fruit {fruit['id']}", fruit["color"], 0.42))
        face_indices.extend([len(fruit_mats) - 1] * (len(fruit_faces) - before))
        tube(stem_verts, stem_faces, anchor,
             (center[0], center[1], center[2] + radius), 0.0025, 0.0020, sides=6)
    if shown:
        mesh_object("OrchardBench fruit (source spheres)", fruit_verts, fruit_faces,
                    fruit_mats, face_indices)
        mesh_object("OrchardBench pedicels", stem_verts, stem_faces, [wood])
    return shown


def bbox(data):
    points = [point for segment in data["branches"]
              for point in (segment["start"], segment["end"])]
    points.extend(leaf["position"] for leaf in data["leaves"])
    points.extend(fruit["center"] for fruit in data["fruits"])
    low = tuple(min(p[i] for p in points) for i in range(3))
    high = tuple(max(p[i] for p in points) for i in range(3))
    return low, high


def import_robot(path, position):
    if not path.is_file():
        raise FileNotFoundError(path)
    previous = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=str(path.resolve()))
    imported = set(bpy.data.objects) - previous
    if not any(obj.type == "MESH" for obj in imported):
        raise RuntimeError("USD import created no G1 mesh objects")
    holder = bpy.data.objects.new("G1 exact USD pose (root z=0.8, yaw=90deg)", None)
    bpy.context.collection.objects.link(holder)
    for obj in imported:
        if obj.parent not in imported:
            obj.parent = holder
    holder.location = position
    holder.rotation_euler.z = math.pi / 2
    return len(imported)


def render_view(output, name, eye, target, ortho_scale, resolution):
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new(name)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ortho_scale
    camera = bpy.data.objects.new(name, camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = eye
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(output / f"{name}.png")
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(camera, do_unlink=True)


def main():
    args = arguments()
    tree = json.loads(args.tree.read_text(encoding="utf-8"))
    if tree.get("units") != "m" or tree.get("up_axis") != "Z":
        raise ValueError("Expected canonical tree JSON in metres with Z up")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Review directory must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.background_type = "WORLD"
    scene.world.color = (0.80, 0.86, 0.91)
    scene.render.image_settings.file_format = "PNG"
    wood = material("Bark", (0.30, 0.16, 0.075))
    branches(tree, wood)
    leaves(tree)
    shown = fruits(tree, wood, args.exclude_fruit_id)
    low, high = bbox(tree)
    center = tuple((lo + hi) / 2 for lo, hi in zip(low, high))
    width = max(high[0] - low[0], high[1] - low[1])
    height = high[2] - min(0.0, low[2])
    span = max(height * 1.18, width * 1.5)
    distance = max(4.0, 2.2 * span)
    target = (center[0], center[1], max(0.0, low[2]) + height * 0.52)
    views = {
        "front": ((center[0], center[1] + distance, target[2] + 0.25 * height), span),
        "side": ((center[0] + distance, center[1], target[2] + 0.15 * height), span),
        "three_quarter": ((center[0] + 0.75 * distance, center[1] + 0.75 * distance,
                            target[2] + 0.32 * height), span),
    }
    for name, (eye, scale) in views.items():
        render_view(args.output, name, eye, target, scale, args.resolution)
    robot_status = "not_requested"
    selected = next((f for f in tree["fruits"] if f["id"] == 28), tree["fruits"][0])
    robot_position = (selected["center"][0]+0.10, selected["center"][1]-0.43, 0.8)
    if args.robot_usd:
        try:
            count = import_robot(args.robot_usd, robot_position)
            robot_status = f"imported_{count}_objects"
            # Include both tree and G1 at the actual unscaled Isaac root pose.
            robot_target = ((center[0]+robot_position[0])/2,
                            (center[1]+robot_position[1])/2, height * 0.45)
            robot_span = max(span, height * 1.35, width * 1.7)
            eye = (robot_target[0] + distance * 0.72,
                   robot_target[1] + distance * 0.72, robot_target[2] + 0.28 * height)
            render_view(args.output, "g1_scale", eye, robot_target, robot_span,
                        args.resolution)
        except Exception as exc:
            robot_status = f"incomplete: {type(exc).__name__}: {exc}"
            traceback.print_exc()
    scene.render.filepath = ""
    bpy.ops.wm.save_as_mainfile(filepath=str((args.output / "tree_review.blend").resolve()))
    report = {
        "tree_json": str(args.tree.resolve()), "tree_id": tree.get("tree_id"),
        "seed": tree.get("seed"), "source": tree.get("source"),
        "units": "m", "up_axis": "Z", "bounds_min_m": low, "bounds_max_m": high,
        "branch_count": len(tree["branches"]), "leaf_count": len(tree["leaves"]),
        "fruit_count": len(tree["fruits"]), "rendered_fruit_count": len(shown),
        "excluded_fruit_id": args.exclude_fruit_id,
        "views": [f"{name}.png" for name in views] +
                 (["g1_scale.png"] if robot_status.startswith("imported_") else []),
        "robot_usd": str(args.robot_usd.resolve()) if args.robot_usd else None,
        "robot_status": robot_status,
        "robot_root_position_m": robot_position,
        "robot_pose_note": "Unscaled source USD pose, not simulated standing/control evidence",
        "render_engine": "Blender Workbench geometry diagnostic",
        "validation_scope": "Blender geometry/scale only; not Isaac physics or rendering",
    }
    (args.output / "review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if args.robot_usd and not robot_status.startswith("imported_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
