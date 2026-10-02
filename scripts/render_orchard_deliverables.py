#!/usr/bin/env python3
"""Export and render OrchardBench seed comparison and a small orchard preview.

Export: Blender's bundled Python, without Blender UI:
  python scripts/render_orchard_deliverables.py export --source CHECKOUT
Render: Blender --background --factory-startup --python THIS_FILE -- render --mode grid
The orchard mode is only run after the grid has been reviewed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
SEEDS = (1, 7, 21, 42)
GRID = ROOT / "outputs/orchardbench/stochastic_tree_grid"
ORCHARD = ROOT / "outputs/orchardbench/orchard_preview"


def export(source: Path):
    from export_orchardbench_tree import export_tree

    GRID.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "LICENSE", GRID / "LICENSE")
    for seed in SEEDS:
        tree = export_tree(source, seed=seed)
        path = GRID / f"seed{seed}.json"
        path.write_text(json.dumps(tree, indent=2, allow_nan=False) + "\n")
        print(f"{path}: {len(tree['branches'])} branches, "
              f"{len(tree['leaves'])} leaves, {len(tree['fruits'])} fruit", flush=True)


def load_trees():
    trees = []
    for seed in SEEDS:
        data = json.loads((GRID / f"seed{seed}.json").read_text())
        if data["seed"] != seed or data["units"] != "m" or data["up_axis"] != "Z":
            raise ValueError(f"Invalid canonical record for seed {seed}")
        trees.append(data)
    return trees


def setup_scene():
    import bpy
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
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    return scene


def create_tree(review, tree, location=(0, 0, 0)):
    """Use the source-scale review meshes and move only the tree assembly."""
    import bpy
    wood = review.material(f"Bark seed {tree['seed']}", (0.30, 0.16, 0.075))
    before = set(bpy.data.objects)
    review.branches(tree, wood)
    review.leaves(tree)
    review.fruits(tree, wood, excluded=None)
    for obj in set(bpy.data.objects) - before:
        obj.location = location


def common_camera(review, output, name, resolution):
    # Fixed world-space pose and scale for every seed. None of these values is
    # derived from an individual tree's bounds.
    review.render_view(output, name, (6.5, 8.5, 5.1), (0, 0, 1.75), 5.6, resolution)


def grid_render(trees, resolution):
    import bpy
    import numpy as np
    import render_tree_review as review

    GRID.mkdir(parents=True, exist_ok=True)
    for tree in trees:
        setup_scene()
        create_tree(review, tree)
        common_camera(review, GRID, f"seed{tree['seed']}", resolution)

    # Assemble losslessly inside Blender; each quadrant is an independent PNG.
    size = resolution
    pixels = np.ones((2 * size, 2 * size, 4), dtype=np.float32)
    for index, seed in enumerate(SEEDS):
        image = bpy.data.images.load(str((GRID / f"seed{seed}.png").resolve()), check_existing=False)
        rgba = np.empty(size * size * 4, dtype=np.float32)
        image.pixels.foreach_get(rgba)
        tile = rgba.reshape(size, size, 4)
        row, col = divmod(index, 2)
        pixels[(1-row)*size:(2-row)*size, col*size:(col+1)*size, :] = tile
        bpy.data.images.remove(image)
    montage = bpy.data.images.new("OrchardBench seeds 1 7 21 42", width=2*size,
                                  height=2*size, alpha=True, float_buffer=False)
    montage.pixels.foreach_set(pixels.ravel())
    montage.filepath_raw = str((GRID / "grid_2x2.png").resolve())
    montage.file_format = "PNG"
    montage.save()
    report = {
        "kind": "stochastic_tree_grid", "seeds_top_left_to_bottom_right": SEEDS,
        "tree_json": [f"seed{s}.json" for s in SEEDS],
        "individual_renders": [f"seed{s}.png" for s in SEEDS],
        "grid_render": "grid_2x2.png", "tile_resolution_px": size,
        "camera_eye_m": [6.5, 8.5, 5.1], "camera_target_m": [0, 0, 1.75],
        "orthographic_scale_m": 5.6, "tree_scale": 1.0,
        "source_revision": trees[0]["source"]["revision"],
        "per_seed": [{"seed": t["seed"], "branch_count": len(t["branches"]),
                      "leaf_count": len(t["leaves"]), "fruit_count": len(t["fruits"]),
                      "bounds": t["bounds"]} for t in trees],
        "scope": "Blender visual geometry only; no Isaac physics or contact validation",
    }
    (GRID / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def orchard_render(trees, resolution):
    import bpy
    from mathutils import Vector
    import render_tree_review as review

    if not (GRID / "grid_2x2.png").is_file():
        raise RuntimeError("Seed grid must exist before orchard render")
    ORCHARD.mkdir(parents=True, exist_ok=True)
    scene = setup_scene()
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.66, 0.78, 0.96, 1)
    background.inputs["Strength"].default_value = 0.8
    sun_data = bpy.data.lights.new("Soft afternoon sunlight", type="SUN")
    sun_data.energy = 2.2
    sun_data.angle = 0.09
    sun = bpy.data.objects.new("Soft afternoon sunlight", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (0.45, -0.5, -0.6)
    # Two rows run along X, four unscaled source trees per row. The 4 m grass
    # alley lies between the row centers at y=+-2 m.
    positions = []
    for row, y in enumerate((-2.0, 2.0)):
        for column, x in enumerate((-6.0, -2.0, 2.0, 6.0)):
            seed = SEEDS[(column + row) % len(SEEDS)]
            tree = trees[SEEDS.index(seed)]
            create_tree(review, tree, (x, y, 0))
            positions.append({"row": row, "column": column, "seed": seed,
                              "root_position_m": [x, y, 0], "scale": 1.0})
    grass = review.material("Grass alley", (0.29, 0.40, 0.17))
    strip = review.material("Managed under-tree strip", (0.43, 0.34, 0.20))
    def mottled(mat, dark, light, scale):
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        texture = nodes.new("ShaderNodeTexNoise")
        texture.inputs["Scale"].default_value = scale
        texture.inputs["Detail"].default_value = 3
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.28
        ramp.color_ramp.elements[0].color = (*dark, 1)
        ramp.color_ramp.elements[1].position = 0.72
        ramp.color_ramp.elements[1].color = (*light, 1)
        links.new(texture.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
    mottled(grass, (0.19, 0.28, 0.10), (0.37, 0.48, 0.19), 22)
    mottled(strip, (0.26, 0.19, 0.10), (0.48, 0.36, 0.21), 30)
    def ground(name, center_y, width, mat, top_z=0.0):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, center_y, top_z - 0.03))
        obj = bpy.context.object
        obj.name = name
        obj.dimensions = (17.5, width, 0.06)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.data.materials.append(mat)
    ground("Grass alley and borders", 0, 8.5, grass)
    for row, y in enumerate((-2.0, 2.0)):
        ground(f"Managed row strip {row+1}", y, 1.15, strip, top_z=0.008)
    def perspective(name, eye, target, lens, width, height):
        camera_data = bpy.data.cameras.new(name)
        camera_data.type = "PERSP"
        camera_data.lens = lens
        camera = bpy.data.objects.new(name, camera_data)
        bpy.context.collection.objects.link(camera)
        camera.location = eye
        camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.camera = camera
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100
        scene.render.filepath = str(ORCHARD / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(camera, do_unlink=True)
    perspective("orchard_overview_cycles", (11.5, 13, 8), (0, 0, 1.3), 31,
                round(resolution * 4 / 3), round(resolution * 0.8))
    perspective("orchard_alley_eye_level", (9.7, 0, 2.6), (0, 0, 1.5), 32,
                resolution, resolution)
    report = {"kind": "orchard_visual_prototype", "row_count": 2,
              "trees_per_row": 4, "tree_count": len(positions),
              "spacing_along_row_m": 4.0, "row_center_spacing_m": 4.0,
              "managed_strip_width_m": 1.15,
              "positions": positions, "views": ["orchard_overview_cycles.png", "orchard_alley_eye_level.png"],
              "render_engine": "Cycles CPU, 16 samples with denoising",
              "scope": "Blender visual layout only; no physics, collision, control or Isaac validation"}
    (ORCHARD / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    cli = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "render"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--mode", choices=("grid", "orchard"), default="grid")
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args(cli)
    if args.action == "export":
        if args.source is None:
            parser.error("export requires --source")
        export(args.source)
    elif args.mode == "grid":
        grid_render(load_trees(), args.resolution)
    else:
        orchard_render(load_trees(), args.resolution)


if __name__ == "__main__":
    main()
