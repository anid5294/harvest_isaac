"""Opt-in OrchardSpec adapter; native single-fruit mechanics stay in env_cfg/mdp."""

from dataclasses import replace
import math
from pathlib import Path


def yaw_quaternion(degrees):
    half = math.radians(degrees) / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def selected_tree(spec):
    return next(t for t in spec["trees"] if t["instance_id"] == spec["target"]["tree_instance_id"])


def physical_layout(spec, pool):
    from .orchard_generation import validate_spec
    from .external_tree import load_asset
    validate_spec(spec, pool)
    tree = selected_tree(spec)
    directory = Path(pool) / tree["asset"]["manifest_path"]
    layout, usd, manifest = load_asset(directory.parent, int(tree["asset_id"].removeprefix("seed")))
    physics = spec["physics"]
    layout = replace(layout, seed=spec["seed"],
                     apples=(tuple(spec["target"]["position"]),),
                     stem_anchors=(tuple(spec["target"]["stem_anchor"]),),
                     basket=tuple(spec["basket"]["position"]),
                     mass=physics["mass_kg"], stem_break_force=physics["stem_break_force_n"])
    return layout, usd, manifest


def populate(cfg, add_asset):
    """Add background trees/floor and apply the spec's robot, lights and camera."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from ..common import camera_cfg, make_g1_cfg
    from ..orchard_preview.layout import cylinder_between
    from .orchard_generation import transform_point
    import json

    spec, pool = cfg.orchard_spec, Path(cfg.orchard_pool)
    target = selected_tree(spec)
    scene = cfg.scene
    canonical_target = json.loads((pool / target["asset"]["canonical_path"]).read_text())
    target_fruit = next(f for f in canonical_target["fruits"] if f["id"] == target["selected_fruit_id"])
    scene.object.spawn.visual_material.diffuse_color = tuple(target_fruit["color"])
    for tree in spec["trees"]:
        if tree["instance_id"] == target["instance_id"]:
            continue
        name = tree["instance_id"]
        add_asset(name, AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Orchard/{name}",
            spawn=sim_utils.UsdFileCfg(usd_path=str((pool / tree["asset"]["usd_path"]).resolve())),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(tree["position"]), rot=yaw_quaternion(tree["yaw_deg"])),
        ))
        # Compiled assets omit their selected apple. Restore it visually only
        # on non-manipulation trees, avoiding a hidden missing-fruit pattern.
        canonical = json.loads((pool / tree["asset"]["canonical_path"]).read_text())
        fruit = next(f for f in canonical["fruits"] if f["id"] == tree["selected_fruit_id"])
        position = transform_point(fruit["center"], tree["position"], tree["yaw_deg"])
        add_asset(name + "_fruit", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Orchard/{name}_Fruit",
            spawn=sim_utils.SphereCfg(radius=fruit["radius"],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(fruit["color"]))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(position)),
        ))
        bottom = list(fruit["center"])
        bottom[2] += fruit["radius"]
        stem = cylinder_between(
            transform_point(bottom, tree["position"], tree["yaw_deg"]),
            transform_point(fruit["anchor"], tree["position"], tree["yaw_deg"]), 0.0015)
        add_asset(name + "_stem", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Orchard/{name}_Stem",
            spawn=sim_utils.CylinderCfg(radius=stem.radius, height=stem.length,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.16, 0.075))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=stem.center, rot=stem.orientation_wxyz),
        ))
    for index, strip in enumerate(spec["infrastructure"]["surfaces"]):
        add_asset(f"orchard_surface_{index}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Orchard/Surface_{index}",
            spawn=sim_utils.CuboidCfg(size=tuple(strip["size"]),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(strip["color"]), roughness=0.95)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(strip["position"]), rot=yaw_quaternion(strip["yaw_deg"])),
        ))
    scene.robot = make_g1_cfg(tuple(spec["robot"]["position"]))
    scene.robot.init_state.rot = tuple(spec["robot"]["orientation_wxyz"])
    lighting = spec["lighting"]
    scene.dome_light.spawn.intensity = lighting["dome_intensity"]
    scene.dome_light.spawn.color = tuple(lighting["dome_color"])
    scene.key_light.spawn.intensity = lighting["sun_intensity"]
    scene.key_light.spawn.color = tuple(lighting["sun_color"])
    # USD distant lights emit along local -Z. Rotate it toward the ground.
    az, polar = math.radians(lighting["azimuth_deg"]), math.radians(90 - lighting["elevation_deg"])
    scene.key_light.init_state.rot = (math.cos(az/2)*math.cos(polar/2),
        -math.sin(az/2)*math.sin(polar/2), math.cos(az/2)*math.sin(polar/2),
        math.sin(az/2)*math.cos(polar/2))
    scene.cam_left_high = camera_cfg(tuple(spec["camera"]["eye"]), tuple(spec["camera"]["target"]))
    cfg.camera_eye, cfg.camera_target = tuple(spec["camera"]["eye"]), tuple(spec["camera"]["target"])
    cfg.commercial_tree_summary = {"orchard_spec_schema": spec["schema"],
                                   "orchard_seed": spec["seed"], "tree_count": len(spec["trees"]),
                                   "feasibility": spec["feasibility"]}
