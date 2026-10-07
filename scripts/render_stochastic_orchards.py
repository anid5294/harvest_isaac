#!/usr/bin/env python3
"""Render four OrchardSpec scenes with source OrchardBench trees and the G1 USD.

Run with Blender, for example:
  /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
    --python scripts/render_stochastic_orchards.py -- \
    --spec-dir outputs/orchardbench/stochastic_orchards

This is a geometry and composition review, not a physics or grasp validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_tree_review as review


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (101, 202, 303, 404)
DEFAULT_OUTPUT = ROOT / "outputs/orchardbench/stochastic_orchards"
DEFAULT_ROBOT = ROOT / "assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"


def arguments():
    cli = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-dir", type=Path,
                        default=ROOT / "outputs/orchardbench/tree_pool",
                        help="Base directory for relative tree asset paths")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--robot-usd", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--resolution", type=int, default=640,
                        help="Width of each individual render; overview uses 4:3 aspect")
    parser.add_argument("--seed", type=int, choices=SEEDS,
                        help="Render one seed without making the four seed contact sheet")
    return parser.parse_args(cli)


def spec_path(directory: Path, seed: int) -> Path:
    candidates = (directory / f"seed{seed}" / "orchard_spec.json",
                  directory / f"seed{seed}.json",
                  directory / f"seed{seed}" / "spec.json")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"OrchardSpec for seed {seed}; checked {candidates}")


def asset_path(raw: str, spec_file: Path, spec_dir: Path, pool_dir: Path | None) -> Path:
    path = Path(raw)
    candidates = ([path] if path.is_absolute() else
                  ([pool_dir / path] if pool_dir else []) +
                  [spec_file.parent / path, spec_dir / path, ROOT / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Tree asset {raw}; checked {candidates}")


def cuboid(name: str, location, dimensions, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    return obj


def noise_material(name, dark, light, scale):
    mat = review.material(name, light)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 3.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.28
    ramp.color_ramp.elements[0].color = (*dark, 1)
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = (*light, 1)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
    return mat


def setup_scene(lighting):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in tuple(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in tuple(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (*lighting["dome_color"], 1)
    background.inputs["Strength"].default_value = 0.8 * float(lighting["dome_intensity"]) / 950.0
    sun_data = bpy.data.lights.new("Seeded afternoon sun", type="SUN")
    sun_data.energy = 2.2 * float(lighting["sun_intensity"]) / 3000.0
    sun_data.angle = 0.09
    sun_data.color = lighting["sun_color"]
    sun = bpy.data.objects.new("Seeded afternoon sun", sun_data)
    bpy.context.collection.objects.link(sun)
    azimuth = math.radians(float(lighting["azimuth_deg"]))
    elevation = math.radians(float(lighting["elevation_deg"]))
    incoming = Vector((-math.cos(elevation)*math.cos(azimuth),
                       -math.cos(elevation)*math.sin(azimuth), -math.sin(elevation)))
    sun.rotation_euler = incoming.to_track_quat("-Z", "Y").to_euler()
    fill_data = bpy.data.lights.new("Soft sky fill", type="AREA")
    fill_data.energy = 450
    fill_data.shape = "DISK"
    fill_data.size = 9
    fill = bpy.data.objects.new("Soft sky fill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (-4, -5, 8)
    fill.rotation_euler = (Vector((0, 0, 0)) - fill.location).to_track_quat("-Z", "Y").to_euler()
    return scene


def tree_instances(trees, spec_file: Path, spec_dir: Path, pool_dir: Path | None):
    templates = {}
    counts = {}
    wood = review.material("OrchardBench bark", (0.30, 0.16, 0.075))
    for item in trees:
        asset = item["asset"]
        path = asset_path(asset["canonical_path"], spec_file, spec_dir, pool_dir)
        if path not in templates:
            tree = json.loads(path.read_text(encoding="utf-8"))
            if tree.get("units") != "m" or tree.get("up_axis") != "Z":
                raise ValueError(f"Noncanonical tree units/axis: {path}")
            before = set(bpy.data.objects)
            review.branches(tree, wood)
            review.leaves(tree)
            review.fruits(tree, wood, excluded=None)
            source = tuple(set(bpy.data.objects) - before)
            for obj in source:
                obj.hide_render = True
                obj.hide_viewport = True
            templates[path] = source
            counts[str(path)] = {"branches": len(tree["branches"]),
                                 "leaves": len(tree["leaves"]),
                                 "fruits": len(tree["fruits"])}
        root = bpy.data.objects.new(f"Tree {item['instance_id']} source-scale root", None)
        bpy.context.collection.objects.link(root)
        root.location = item["position"]
        root.rotation_euler.z = math.radians(float(item["yaw_deg"]))
        for original in templates[path]:
            copy = original.copy()
            copy.data = original.data  # Shared exact source mesh, individual world pose.
            copy.hide_render = False
            copy.hide_viewport = False
            bpy.context.collection.objects.link(copy)
            copy.parent = root
    return counts


def ground_rows(trees, surfaces):
    xs = [float(item["position"][0]) for item in trees]
    ys = [float(item["position"][1]) for item in trees]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    grass = noise_material("Managed alley grass", (0.19, 0.28, 0.10),
                           (0.37, 0.48, 0.19), 22)
    soil = noise_material("Managed under-tree strips", (0.26, 0.19, 0.10),
                          (0.48, 0.36, 0.21), 30)
    for index, surface in enumerate(surfaces):
        obj = cuboid("Grass alley and border" if index == 0 else
                     f"Managed strip row {index-1}", surface["position"],
                     surface["size"], grass if index == 0 else soil)
        obj.rotation_euler.z = math.radians(float(surface["yaw_deg"]))
    return xmin, xmax, ymin, ymax


def basket(position, yaw_deg):
    """Open basket at its specified world pose, with visible collection cavity."""
    wicker = review.material("Basket woven ochre", (0.43, 0.25, 0.095), 0.82)
    rim = review.material("Basket rim", (0.29, 0.13, 0.045), 0.76)
    root = bpy.data.objects.new("Harvest basket pose", None)
    bpy.context.collection.objects.link(root)
    root.location = position
    root.rotation_euler.z = math.radians(float(yaw_deg))
    # OrchardSpec gives the basket's center at z=0.78 m.
    base_z = -0.15
    for name, loc, dim, mat in (
        ("basket base", (0, 0, base_z), (0.32, 0.28, 0.04), wicker),
        ("basket long wall near", (0, -0.125, base_z+0.15), (0.34, 0.025, 0.29), wicker),
        ("basket long wall far", (0, 0.125, base_z+0.15), (0.34, 0.025, 0.29), wicker),
        ("basket short wall left", (-0.145, 0, base_z+0.15), (0.025, 0.28, 0.29), wicker),
        ("basket short wall right", (0.145, 0, base_z+0.15), (0.025, 0.28, 0.29), wicker),
        ("basket rim near", (0, -0.125, base_z+0.31), (0.38, 0.04, 0.04), rim),
        ("basket rim far", (0, 0.125, base_z+0.31), (0.38, 0.04, 0.04), rim),
        ("basket rim left", (-0.145, 0, base_z+0.31), (0.04, 0.31, 0.04), rim),
        ("basket rim right", (0.145, 0, base_z+0.31), (0.04, 0.31, 0.04), rim),
        ("basket support left front", (-0.13, -0.11, -0.42), (0.035, 0.035, 0.72), rim),
        ("basket support right front", (0.13, -0.11, -0.42), (0.035, 0.035, 0.72), rim),
        ("basket support left back", (-0.13, 0.11, -0.42), (0.035, 0.035, 0.72), rim),
        ("basket support right back", (0.13, 0.11, -0.42), (0.035, 0.035, 0.72), rim),
    ):
        obj = cuboid(name, loc, dim, mat)
        obj.parent = root
    return root


def render_camera(scene, path: Path, name: str, eye, target, lens, width, height,
                  fit_scene=False):
    camera_data = bpy.data.cameras.new(name)
    camera_data.type = "ORTHO" if fit_scene else "PERSP"
    if not fit_scene:
        camera_data.lens = lens
    camera = bpy.data.objects.new(name, camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = eye
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    if fit_scene:
        bpy.context.view_layer.update()
        rotation = camera.rotation_euler.to_matrix()
        right, up = rotation @ Vector((1, 0, 0)), rotation @ Vector((0, 1, 0))
        projected = []
        for obj in scene.objects:
            if obj.type != "MESH" or obj.hide_render:
                continue
            for corner in obj.bound_box:
                point = obj.matrix_world @ Vector(corner)
                projected.append((point.dot(right), point.dot(up)))
        if not projected:
            raise RuntimeError("No visible geometry to frame")
        left, bottom = (min(point[axis] for point in projected) for axis in (0, 1))
        right_edge, top = (max(point[axis] for point in projected) for axis in (0, 1))
        offset = right * ((left + right_edge)/2 - Vector(target).dot(right))
        offset += up * ((bottom + top)/2 - Vector(target).dot(up))
        camera.location += offset
        camera_data.ortho_scale = 1.10 * max(right_edge-left,
                                              (top-bottom)*width/height)
    scene.camera = camera
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(camera, do_unlink=True)


def render_seed(seed: int, args):
    spec_file = spec_path(args.spec_dir, seed)
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    if int(spec["seed"]) != seed:
        raise ValueError(f"Spec seed mismatch: {spec_file}")
    trees = spec["trees"]
    if not trees:
        raise ValueError(f"No trees: {spec_file}")
    scene = setup_scene(spec["lighting"])
    counts = tree_instances(trees, spec_file, args.spec_dir, args.pool_dir)
    bounds = ground_rows(trees, spec["infrastructure"]["surfaces"])
    robot_pose = spec["robot"]
    before = set(bpy.data.objects)
    imported_count = review.import_robot(args.robot_usd, tuple(robot_pose["position"]))
    imported = set(bpy.data.objects) - before
    holder = next((obj for obj in imported if obj.name.startswith("G1 exact USD pose")), None)
    if holder is None or not any(obj.type == "MESH" for obj in imported):
        raise RuntimeError("G1 USD mesh/pose holder unavailable")
    # render_tree_review uses a diagnostic fixed 90 degree yaw; OrchardSpec owns this pose.
    holder.rotation_euler.z = math.radians(float(robot_pose["yaw_deg"]))
    basket(spec["basket"]["position"], spec["basket"]["yaw_deg"])
    out = args.output / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    xmin, xmax, ymin, ymax = bounds
    center_x, center_y = (xmin+xmax)/2, (ymin+ymax)/2
    width, depth = xmax-xmin, ymax-ymin
    # Fixed three-quarter direction; the orthographic view fits actual visible bounds.
    overview_eye = (center_x + 12.0, center_y + 22.0, 18.0)
    overview_target = (center_x, center_y, 1.35)
    render_camera(scene, out / "overview.png", "overview", overview_eye,
                  overview_target, 20, round(args.resolution*4/3), args.resolution,
                  fit_scene=True)
    alley_eye = (xmax+3.7, center_y, 2.6)
    alley_target = (center_x, center_y, 1.5)
    render_camera(scene, out / "alley.png", "alley", alley_eye,
                  alley_target, 32, args.resolution, args.resolution)
    report = {"seed": seed, "spec": str(spec_file.resolve()),
              "spec_sha256": hashlib.sha256(spec_file.read_bytes()).hexdigest(),
              "tree_count": len(trees),
              "source_tree_counts": counts, "robot_usd": str(args.robot_usd.resolve()),
              "robot_import_object_count": imported_count,
              "robot_position_m": robot_pose["position"],
              "robot_yaw_deg": robot_pose["yaw_deg"],
              "basket_pose": spec["basket"],
              "views": ["overview.png", "alley.png"],
              "render_engine": "Cycles CPU; 16 samples; denoising; AgX; exposure 0",
              "scope": "Blender visual geometry only; no Isaac physics or task validation"}
    (out / "render_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rendered": seed, "output": str(out), "robot_objects": imported_count}), flush=True)


def contact_sheet(output: Path, resolution: int):
    import numpy as np
    paths = [output / f"seed{seed}" / "overview.png" for seed in SEEDS]
    images = [bpy.data.images.load(str(path.resolve()), check_existing=False) for path in paths]
    width, height = images[0].size
    pixels = np.ones((2*height, 2*width, 4), dtype=np.float32)
    for index, img in enumerate(images):
        if tuple(img.size) != (width, height):
            raise ValueError("Overview resolutions differ")
        tile = np.empty(width*height*4, dtype=np.float32)
        img.pixels.foreach_get(tile)
        row, col = divmod(index, 2)
        pixels[(1-row)*height:(2-row)*height, col*width:(col+1)*width, :] = tile.reshape(height, width, 4)
        bpy.data.images.remove(img)
    montage = bpy.data.images.new("Stochastic orchards 101 202 303 404",
                                  width=2*width, height=2*height, alpha=True)
    montage.pixels.foreach_set(pixels.ravel())
    montage.filepath_raw = str((output / "grid_2x2.png").resolve())
    montage.file_format = "PNG"
    montage.save()
    bpy.data.images.remove(montage)


def main():
    args = arguments()
    args.spec_dir = args.spec_dir.resolve()
    args.pool_dir = args.pool_dir.resolve() if args.pool_dir else None
    args.output = args.output.resolve()
    args.robot_usd = args.robot_usd.resolve()
    if not args.robot_usd.is_file():
        raise FileNotFoundError(args.robot_usd)
    seeds = (args.seed,) if args.seed else SEEDS
    for seed in seeds:
        render_seed(seed, args)
    if args.seed is None:
        contact_sheet(args.output, args.resolution)
        print(json.dumps({"grid": str(args.output / "grid_2x2.png"), "seeds": SEEDS}), flush=True)


if __name__ == "__main__":
    main()
