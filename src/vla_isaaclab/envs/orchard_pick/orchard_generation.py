"""Deterministic, simulator-free OrchardBench scene specifications.

Trees retain their native metres, Z-up morphology and scale. Crown circles use
the complete branch, fruit and leaf extents; they are deliberately conservative.
The stance test below is a body/wood/foliage proxy, not an IK or simulator collision test.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from .external_tree import load_asset, load_tree, wood_clearance

SCHEMA = "orchard_scene_spec_v1"
STREAM_NAMES = ("layout", "tree_selection", "target", "robot", "lighting", "basket")
MIN_CROWN_GAP_M = 0.20
STANCE_OFFSETS_M = ((0.10, -0.43), (0.10, -0.50), (0.0, -0.43),
                    (-0.10, -0.43), (0.0, -0.50))
BASKET_OFFSETS_M = ((-0.35, -0.40, 0.78), (-0.43, -0.40, 0.78))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _streams(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32); negative Isaac seeds are nondeterministic")
    values = {name: int.from_bytes(hashlib.sha256(
        f"{SCHEMA}:{seed}:{name}".encode()).digest(), "big") for name in STREAM_NAMES}
    return values, {name: random.Random(value) for name, value in values.items()}


def _vec3(value):
    if len(value) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value):
        raise ValueError("Expected finite 3-vector")
    return value


def _crown_radius(tree):
    radii = []
    for branch in tree["branches"]:
        radius = max(branch["radius_start"], branch["radius_end"])
        radii.extend(math.hypot(*_vec3(branch[key])[:2]) + radius for key in ("start", "end"))
    for fruit in tree["fruits"]:
        radii.append(math.hypot(*_vec3(fruit["center"])[:2]) + fruit["radius"])
    for leaf in tree.get("leaves", []):
        # The half diagonal bounds any orientation of this blade around its position.
        radii.append(math.hypot(*_vec3(leaf["position"])[:2]) +
                     leaf["length"] + 0.5 * leaf["width"])
    result = max(radii)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Invalid crown radius")
    return result


def _pool_assets(pool_dir):
    pool = Path(pool_dir).resolve()
    assets = []
    for canonical in sorted(pool.glob("seed*/tree.json")):
        identity = canonical.parent.name
        if not identity[4:].isdigit():
            continue
        seed = int(identity[4:])
        directory = canonical.parent / "isaac"
        layout, usd, manifest = load_asset(directory, seed)
        tree = load_tree(canonical, seed)
        fruit = next(f for f in tree["fruits"] if str(f["id"]) == str(manifest["selected_fruit_id"]))
        if wood_clearance(tree, fruit) <= 0:
            raise ValueError(f"Selected fruit intersects wood in {identity}")
        files = {"canonical_path": canonical.relative_to(pool).as_posix(),
                 "usd_path": usd.relative_to(pool).as_posix(),
                 "manifest_path": (directory / "manifest.json").relative_to(pool).as_posix(),
                 "canonical_sha256": _sha(canonical), "usd_sha256": _sha(usd),
                 "manifest_sha256": _sha(directory / "manifest.json")}
        assets.append({"id": identity, "asset": files, "radius": _crown_radius(tree),
                       "tree": tree, "fruit": fruit, "layout": layout})
    if not assets:
        raise ValueError(f"No prepared seedN/isaac tree assets in {pool}")
    return assets


def _rotate(x, y, yaw_deg):
    a = math.radians(yaw_deg)
    return (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a))


def _world(local, instance):
    x, y = _rotate(local[0], local[1], instance["yaw_deg"])
    return [instance["position"][0] + x, instance["position"][1] + y,
            instance["position"][2] + local[2]]


def transform_point(point, position, yaw_deg):
    """Transform a source-tree point by an upright, unscaled tree pose."""
    _vec3(point)
    _vec3(position)
    x, y = _rotate(point[0], point[1], yaw_deg)
    return [position[0] + x, position[1] + y, position[2] + point[2]]


def _surfaces(rows, slots, along_pitch, row_pitch, yaw_deg):
    length = (slots - 1) * along_pitch + 3.0
    width = (rows - 1) * row_pitch + 3.0
    surfaces = [{"size": [length + 3.0, width + 3.0, 0.02],
                 "position": [0.0, 0.0, -0.008], "yaw_deg": yaw_deg,
                 "color": [0.18, 0.31, 0.12]}]
    for row in range(rows):
        y = (row - (rows - 1) / 2) * row_pitch
        xw, yw = _rotate(0.0, y, yaw_deg)
        surfaces.append({"size": [length, 1.0, 0.02],
                         "position": [xw, yw, -0.005], "yaw_deg": yaw_deg,
                         "color": [0.30, 0.23, 0.15]})
    return surfaces


def _lighting(seed, generator):
    # Distinct day conditions cycle across adjacent run seeds; each condition
    # retains an independent named intensity stream.
    presets = (
        ("overcast", 2600.0, 750.0, [0.85, 0.91, 1.0], [1.0, 0.96, 0.90], 35.0, 58.0),
        ("morning", 850.0, 3300.0, [0.72, 0.82, 1.0], [1.0, 0.76, 0.53], 65.0, 25.0),
        ("midday", 1050.0, 3400.0, [0.86, 0.92, 1.0], [1.0, 0.98, 0.90], 15.0, 65.0),
        ("late_afternoon", 750.0, 3000.0, [0.75, 0.83, 1.0], [1.0, 0.72, 0.48], -70.0, 22.0),
    )
    name, dome, sun, dome_color, sun_color, azimuth, elevation = presets[seed % 4]
    return {"preset": name, "dome_intensity": dome * generator.uniform(0.95, 1.05),
            "sun_intensity": sun * generator.uniform(0.95, 1.05),
            "dome_color": dome_color, "sun_color": sun_color,
            "azimuth_deg": azimuth + generator.uniform(-3.0, 3.0),
            "elevation_deg": elevation + generator.uniform(-2.0, 2.0)}


def _point_segment_distance(point, start, end):
    delta = [b - a for a, b in zip(start, end)]
    norm2 = sum(d * d for d in delta)
    t = max(0.0, min(1.0, sum((p - a) * d for p, a, d in zip(point, start, delta)) / norm2))
    return math.dist(point, [a + t * d for a, d in zip(start, delta)])


def _body_clearance(robot_xy, instance, tree):
    # Spheres conservatively approximate a G1 torso and two leg volumes. The
    # fixed-base G1 initial position uses z=0.80; these are world heights.
    proxies = _body_proxies(robot_xy)
    gap = math.inf
    for branch in tree["branches"]:
        start, end = _world(branch["start"], instance), _world(branch["end"], instance)
        wood = max(branch["radius_start"], branch["radius_end"])
        for px, py, pz, radius in proxies:
            gap = min(gap, _point_segment_distance((px, py, pz), start, end) - wood - radius)
    return gap


def _body_proxies(robot_xy):
    x, y = robot_xy
    return ((x, y, 0.98, 0.22), (x, y, 1.30, 0.19),
            (x - 0.10, y, 0.37, 0.12), (x + 0.10, y, 0.37, 0.12))


def _foliage_clearance(robot_xy, instance, tree):
    # Leaves in the prepared asset are visual-only, but a stance embedded in
    # visible foliage is rejected for an unobstructed initial robot pose.
    proxies = _body_proxies(robot_xy)
    gap = math.inf
    for leaf in tree.get("leaves", []):
        center = _world(leaf["position"], instance)
        blade = leaf["length"] + 0.5 * leaf["width"]
        for px, py, pz, radius in proxies:
            gap = min(gap, math.dist((px, py, pz), center) - blade - radius)
    return gap if math.isfinite(gap) else None


def _basket_clearance(center, instance, tree):
    # A circumscribed sphere around the supported basket's 0.32 x 0.28 m floor.
    radius = 0.5 * math.hypot(0.32, 0.28) + 0.03
    gap = math.inf
    for branch in tree["branches"]:
        start, end = _world(branch["start"], instance), _world(branch["end"], instance)
        gap = min(gap, _point_segment_distance(center, start, end) -
                  max(branch["radius_start"], branch["radius_end"]) - radius)
    return gap


def _robot_basket_clearance(robot_xy, basket_center):
    # Enclose the basket floor and walls in the same sphere used for wood
    # clearance. Also bound the 0.09 m square pedestal from ground to floor.
    basket_radius = 0.5 * math.hypot(0.32, 0.28) + 0.03
    pedestal_radius = 0.5 * math.hypot(0.09, 0.09)
    pedestal_bottom = (basket_center[0], basket_center[1], 0.0)
    pedestal_top = (basket_center[0], basket_center[1], basket_center[2])
    return min(min(math.dist((px, py, pz), basket_center) - radius - basket_radius,
                   _point_segment_distance((px, py, pz), pedestal_bottom, pedestal_top) -
                   radius - pedestal_radius)
               for px, py, pz, radius in _body_proxies(robot_xy))


def _separation(trees):
    minimum = math.inf
    for i, a in enumerate(trees):
        for b in trees[i + 1:]:
            gap = math.dist(a["position"][:2], b["position"][:2]) - a["crown_radius_m"] - b["crown_radius_m"]
            minimum = min(minimum, gap)
            if gap < MIN_CROWN_GAP_M - 1e-9:
                raise ValueError(f"Tree crowns overlap or lack {MIN_CROWN_GAP_M} m clearance")
    return minimum if math.isfinite(minimum) else None


def generate_orchard_spec(pool_dir, seed, rows=2, slots_per_row=6):
    """Return a JSON-serializable orchard with bounded, reproducible candidates.

    A prepared asset's selected fruit alone is eligible, because its USD omits
    exactly that fruit for a dynamic replacement. Reusing an asset is permitted.
    """
    if (isinstance(rows, bool) or not isinstance(rows, int) or rows < 1 or
            isinstance(slots_per_row, bool) or not isinstance(slots_per_row, int) or slots_per_row < 1):
        raise ValueError("rows and slots_per_row must be positive integers")
    seeds, rng = _streams(seed)
    assets = _pool_assets(pool_dir)
    along_min, along_max = 4.2, 4.8
    row_min, row_max = 4.6, 6.1
    if min(asset["radius"] for asset in assets) * 2 + MIN_CROWN_GAP_M > along_max:
        raise ValueError("No pool tree fits the along-row spacing range")
    along_pitch = rng["layout"].uniform(along_min, along_max)
    row_pitch = rng["layout"].uniform(row_min, row_max)
    # Reserve clearance for the smallest possible neighbour, including 0.12 m
    # of opposing lateral jitter. An unusually broad source may be skipped.
    min_radius = min(asset["radius"] for asset in assets)
    selectable = [asset for asset in assets if asset["radius"] + min_radius +
                  MIN_CROWN_GAP_M + 0.12 <= along_pitch]
    targetable = [asset for asset in selectable if _body_clearance(
        (asset["fruit"]["center"][0] + 0.10, asset["fruit"]["center"][1] - 0.43),
        {"position": [0.0, 0.0, 0.0], "yaw_deg": 0.0}, asset["tree"]) > 0.04 and
        0.85 <= asset["fruit"]["center"][2] <= 1.25 and
        _basket_clearance([asset["fruit"]["center"][0] - 0.35,
                           asset["fruit"]["center"][1] - 0.40, 0.78],
                          {"position": [0.0, 0.0, 0.0], "yaw_deg": 0.0}, asset["tree"]) > 0.0]
    if not targetable:
        raise ValueError("Pool has no prepared selected fruit with a feasible stance proxy")
    guaranteed_target_asset = rng["target"].choice(targetable)
    global_yaw = rng["layout"].uniform(-3.0, 3.0)
    trees = []
    lookup = {}
    rejected = []
    for row in range(rows):
        for slot in range(slots_per_row):
            local_x = (slot - (slots_per_row - 1) / 2) * along_pitch
            local_y = (row - (rows - 1) / 2) * row_pitch + rng["layout"].uniform(-0.06, 0.06)
            x, y = _rotate(local_x, local_y, global_yaw)
            instance_id = f"tree_r{row}_s{slot}"
            yaw = global_yaw + rng["layout"].uniform(-5.0, 5.0)
            choices = selectable.copy()
            rng["tree_selection"].shuffle(choices)
            if row == 0 and slot == 0:
                choices = [guaranteed_target_asset]
            instance = None
            for asset in choices:
                candidate = {"instance_id": instance_id, "row": row, "slot": slot,
                             "asset_id": asset["id"], "tree_seed": asset["tree"]["seed"],
                             "asset": asset["asset"], "position": [x, y, 0.0],
                             "yaw_deg": yaw, "crown_radius_m": asset["radius"],
                             "selected_fruit_id": asset["fruit"]["id"]}
                try:
                    _separation(trees + [candidate])
                except ValueError:
                    if len(rejected) < 48:
                        rejected.append({"tree_instance_id": instance_id, "asset_id": asset["id"],
                                         "reason": "crown_clearance"})
                    continue
                instance = candidate
                break
            if instance is None:
                raise ValueError(f"No source tree fits slot {instance_id} with native crown bounds")
            trees.append(instance)
            lookup[instance_id] = asset
    minimum_gap = _separation(trees)
    candidates = list(range(len(trees)))
    rng["target"].shuffle(candidates)
    selected = None
    for index in candidates:
        instance = trees[index]
        asset = lookup[instance["instance_id"]]
        fruit = asset["fruit"]
        fruit_world = _world(fruit["center"], instance)
        anchor_world = _world(fruit["anchor"], instance)
        for attempt in range(8 * len(STANCE_OFFSETS_M)):
            stance_offset = STANCE_OFFSETS_M[attempt // 8]
            dx = rng["robot"].uniform(-0.03, 0.03)
            dy = rng["robot"].uniform(-0.03, 0.03)
            offset = _rotate(stance_offset[0] + dx, stance_offset[1] + dy,
                             instance["yaw_deg"])
            robot_xy = (fruit_world[0] + offset[0], fruit_world[1] + offset[1])
            yaw = instance["yaw_deg"] + 90.0 + rng["robot"].uniform(-3.0, 3.0)
            body_gap = min(_body_clearance(robot_xy, other, lookup[other["instance_id"]]["tree"])
                           for other in trees)
            foliage_gap = min(
                (gap for other in trees if (gap := _foliage_clearance(
                    robot_xy, other, lookup[other["instance_id"]]["tree"])) is not None),
                default=math.inf)
            for basket_local in BASKET_OFFSETS_M:
                basket_offset = _rotate(basket_local[0], basket_local[1], instance["yaw_deg"])
                basket_center = [fruit_world[0] + basket_offset[0],
                                 fruit_world[1] + basket_offset[1], basket_local[2]]
                basket_distance = math.dist(fruit_world[:2], basket_center[:2])
                if not 0.35 <= basket_distance <= 0.65:
                    continue
                basket_gap = min(_basket_clearance(basket_center, other,
                                                   lookup[other["instance_id"]]["tree"])
                                 for other in trees)
                robot_basket_gap = _robot_basket_clearance(robot_xy, basket_center)
                if body_gap <= 0 or basket_gap <= 0 or foliage_gap <= 0 or robot_basket_gap <= 0:
                    if len(rejected) < 48:
                        rejected.append({"tree_instance_id": instance["instance_id"],
                                         "attempt": attempt,
                                         "basket_local_offset_m": list(basket_local),
                                         "reason": ("body_wood_proxy" if body_gap <= 0 else
                                                    "basket_wood_proxy" if basket_gap <= 0 else
                                                    "body_foliage_proxy" if foliage_gap <= 0 else
                                                    "robot_basket_proxy"),
                                         "body_clearance_m": body_gap,
                                         "basket_clearance_m": basket_gap,
                                         "foliage_clearance_m": foliage_gap,
                                         "robot_basket_clearance_m": robot_basket_gap})
                    continue
                selected = (instance, fruit, fruit_world, anchor_world, robot_xy, yaw,
                            basket_center, body_gap, basket_gap, foliage_gap,
                            robot_basket_gap, stance_offset, basket_local)
                break
            if selected:
                break
        if selected:
            break
    if selected is None:
        raise ValueError("No selected fruit has positive wood, basket, foliage, and robot-basket clearance in bounded stance candidates")
    instance, fruit, fruit_world, anchor_world, robot_xy, yaw, basket_center, body_gap, basket_gap, foliage_gap, robot_basket_gap, stance_offset, basket_local = selected
    angle = math.radians(yaw) / 2
    return {"schema": SCHEMA, "seed": seed, "substream_seeds": seeds,
            "parameters": {"rows": rows, "slots_per_row": slots_per_row,
                           "along_row_pitch_m": along_pitch, "row_pitch_m": row_pitch,
                           "global_yaw_deg": global_yaw,
                           "along_row_range_m": [along_min, along_max],
                           "row_range_m": [row_min, row_max],
                           "individual_yaw_range_deg": [-5.0, 5.0],
                           "row_lateral_jitter_m": 0.06,
                           "crown_gap_min_m": MIN_CROWN_GAP_M,
                           "tree_scale": 1.0, "tree_up_axis": "Z",
                           "robot_stance_local_offset_m": list(stance_offset),
                           "robot_stance_candidate_offsets_m": [list(offset) for offset in STANCE_OFFSETS_M],
                           "robot_stance_jitter_m": 0.03,
                           "robot_yaw_relative_deg": 90.0,
                           "robot_yaw_jitter_deg": 3.0,
                           "basket_local_offset_m": list(basket_local),
                           "basket_candidate_offsets_m": [list(offset) for offset in BASKET_OFFSETS_M]},
            "trees": trees,
            "target": {"tree_instance_id": instance["instance_id"], "fruit_id": fruit["id"],
                       "position": fruit_world, "radius_m": fruit["radius"],
                       "stem_anchor": anchor_world,
                       "source_stem": {"mass_kg": fruit["stem"]["mass_kg"],
                                       "break_force_n": fruit["stem"]["break_force_n"]}},
            "robot": {"position": [robot_xy[0], robot_xy[1], 0.80], "yaw_deg": yaw,
                      "orientation_wxyz": [math.cos(angle), 0.0, 0.0, math.sin(angle)]},
            "basket": {"position": basket_center, "yaw_deg": 0.0},
            "physics": {"mass_kg": 0.16, "stem_break_force_n": 17.341074446037137,
                        "fruit_radius_m": fruit["radius"]},
            "infrastructure": {"surfaces": _surfaces(rows, slots_per_row, along_pitch, row_pitch, global_yaw)},
            "camera": {"eye": [0.0, -max(((slots_per_row - 1) * along_pitch + 3.0) * 0.9, 10.0),
                               max(((slots_per_row - 1) * along_pitch + 3.0) * 0.5, 5.4)],
                       "target": [0.0, 0.0, 1.2]},
            "lighting": _lighting(seed, rng["lighting"]),
            "feasibility": {"status": "proxy_only", "full_collision_ik_verified": False,
                            "minimum_crown_gap_m": minimum_gap,
                            "robot_body_wood_clearance_m": body_gap,
                            "robot_body_visual_foliage_clearance_m": (
                                foliage_gap if math.isfinite(foliage_gap) else None),
                            "basket_wood_clearance_m": basket_gap,
                            "robot_basket_clearance_m": robot_basket_gap,
                            "basket_target_horizontal_distance_m": math.dist(fruit_world[:2], basket_center[:2]),
                            "selected_fruit_wood_clearance_m": wood_clearance(lookup[instance["instance_id"]]["tree"], fruit)},
            "rejected_candidates": rejected}


def validate_spec(spec, pool_dir):
    """Reject stale assets, nonfinite values, changed parameters or edited poses."""
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise ValueError("Unsupported orchard scene schema")
    parameters = spec.get("parameters", {})
    expected = generate_orchard_spec(pool_dir, spec["seed"],
                                     parameters["rows"], parameters["slots_per_row"])
    # JSON round-trip also rejects NaN and infinity, and normalizes integer keys.
    try:
        actual = json.loads(json.dumps(spec, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Non-JSON or nonfinite orchard specification") from exc
    if actual != expected:
        raise ValueError("Orchard specification differs from deterministic validated assets and parameters")
    return True
