"""Deterministic, simulator-free OrchardBench scene specifications.

Trees retain their native metres, Z-up morphology and scale. Presentation layouts
allow bounded crown overlap while requiring clearance between major wood capsules.
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
ROW_COUNT_BOUNDS = (2, 4)
SLOT_COUNT_BOUNDS = (4, 7)
ALONG_GAP_BOUNDS_M = (4.2, 4.8)
ROW_GAP_BOUNDS_M = (4.6, 6.1)
PLANTING_JITTER_M = 0.04
MISSING_SLOT_PROBABILITY = 0.18
DENSE_ROW_COUNT_BOUNDS = (3, 5)
DENSE_SLOT_COUNT_BOUNDS = (6, 9)
DENSE_ALONG_GAP_BOUNDS_M = (2.55, 3.10)
DENSE_ROW_GAP_BOUNDS_M = (3.15, 3.75)
DENSE_PLANTING_JITTER_M = 0.11
DENSE_MISSING_SLOT_PROBABILITY = 0.055
MAX_CROWN_OVERLAP_M = 1.65
MIN_MAJOR_WOOD_GAP_M = 0.06
MAJOR_WOOD_RADIUS_M = 0.02
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


def _major_wood(tree):
    """The trunk and structural branches, represented by tapered capsule bounds."""
    return [(branch["start"], branch["end"],
             max(branch["radius_start"], branch["radius_end"]))
            for branch in tree["branches"]
            if max(branch["radius_start"], branch["radius_end"]) >= MAJOR_WOOD_RADIUS_M]


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
        major_wood = _major_wood(tree)
        major_radius = max((math.hypot(*point[:2]) + radius
                            for start, end, radius in major_wood for point in (start, end)), default=0.0)
        assets.append({"id": identity, "asset": files, "radius": _crown_radius(tree),
                       "major_wood": major_wood, "major_radius": major_radius,
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


def _structured_surfaces(row_centers, row_xs, yaw_deg, crown_margin, jitter=PLANTING_JITTER_M):
    """Cover the entire planting grid, including intentionally empty slots."""
    margin = max(1.5, crown_margin) + jitter
    lo_x = min(min(xs) for xs in row_xs) - margin
    hi_x = max(max(xs) for xs in row_xs) + margin
    lo_y = min(row_centers) - margin
    hi_y = max(row_centers) + margin
    center_x, center_y = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
    gx, gy = _rotate(center_x, center_y, yaw_deg)
    surfaces = [{"size": [hi_x - lo_x + 3.0, hi_y - lo_y + 3.0, 0.02],
                 "position": [gx, gy, -0.008], "yaw_deg": yaw_deg,
                 "color": [0.18, 0.31, 0.12]}]
    for y, xs in zip(row_centers, row_xs):
        x = (xs[0] + xs[-1]) / 2
        wx, wy = _rotate(x, y, yaw_deg)
        surfaces.append({"size": [xs[-1] - xs[0] + 3.0 + 2 * jitter,
                                   1.0, 0.02],
                         "position": [wx, wy, -0.005], "yaw_deg": yaw_deg,
                         "color": [0.30, 0.23, 0.15]})
    return surfaces, hi_x - lo_x, hi_y - lo_y


def _structured_layout(layout_rng, rows, slots_per_row, *, dense=False, coherent=False):
    row_bounds = DENSE_ROW_COUNT_BOUNDS if dense else ROW_COUNT_BOUNDS
    slot_bounds = DENSE_SLOT_COUNT_BOUNDS if dense else SLOT_COUNT_BOUNDS
    row_gap_bounds = DENSE_ROW_GAP_BOUNDS_M if dense else ROW_GAP_BOUNDS_M
    along_gap_bounds = DENSE_ALONG_GAP_BOUNDS_M if dense else ALONG_GAP_BOUNDS_M
    missing_probability = DENSE_MISSING_SLOT_PROBABILITY if dense else MISSING_SLOT_PROBABILITY
    if rows is None:
        rows = layout_rng.randint(*row_bounds)
    if coherent and slots_per_row is None:
        # A managed block has a shared planting length. Count varies between
        # blocks, rather than jumping independently from row to row.
        slot_counts = [layout_rng.randint(*slot_bounds)] * rows
    elif slots_per_row is None:
        slot_counts = [layout_rng.randint(*slot_bounds) for _ in range(rows)]
    else:
        slot_counts = [slots_per_row] * rows
    if coherent:
        block_row_pitch = layout_rng.uniform(3.25, 3.65)
        block_along_pitch = layout_rng.uniform(2.65, 3.00)
        row_gaps = [max(row_gap_bounds[0], min(row_gap_bounds[1],
                    block_row_pitch + layout_rng.uniform(-0.10, 0.10)))
                    for _ in range(rows - 1)]
    else:
        row_gaps = [layout_rng.uniform(*row_gap_bounds) for _ in range(rows - 1)]
    row_centers = [0.0]
    for gap in row_gaps:
        row_centers.append(row_centers[-1] + gap)
    midpoint = (row_centers[0] + row_centers[-1]) / 2
    row_centers = [y - midpoint for y in row_centers]
    along_gaps = []
    row_xs = []
    occupied = []
    for count in slot_counts:
        if coherent:
            row_pitch = block_along_pitch + layout_rng.uniform(-0.05, 0.05)
            gaps = [max(along_gap_bounds[0], min(along_gap_bounds[1],
                    row_pitch + layout_rng.uniform(-0.045, 0.045)))
                    for _ in range(count - 1)]
        else:
            gaps = [layout_rng.uniform(*along_gap_bounds) for _ in range(count - 1)]
        xs = [0.0]
        for gap in gaps:
            xs.append(xs[-1] + gap)
        midpoint = (xs[0] + xs[-1]) / 2
        along_gaps.append(gaps)
        row_xs.append([x - midpoint for x in xs])
        planted = ([*range(count)] if coherent else
                   [slot for slot in range(count)
                    if slot == 0 or layout_rng.random() >= missing_probability])
        # A row must remain visibly planted, even with a short explicit override.
        if count > 1 and len(planted) == 1:
            planted.append(count - 1)
        occupied.append(planted)
    return rows, slot_counts, row_gaps, row_centers, along_gaps, row_xs, occupied


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


def _segment_segment_distance(a0, a1, b0, b1):
    """Shortest distance between two closed 3D segments, including skew crossings."""
    u = [q - p for p, q in zip(a0, a1)]
    v = [q - p for p, q in zip(b0, b1)]
    w = [p - q for p, q in zip(a0, b0)]
    dot = lambda x, y: sum(a * b for a, b in zip(x, y))
    aa, bb, cc = dot(u, u), dot(u, v), dot(v, v)
    dd, ee = dot(u, w), dot(v, w)
    if aa <= 1e-20:
        return _point_segment_distance(a0, b0, b1)
    if cc <= 1e-20:
        return _point_segment_distance(b0, a0, a1)
    nearest = min(_point_segment_distance(a0, b0, b1),
                  _point_segment_distance(a1, b0, b1),
                  _point_segment_distance(b0, a0, a1),
                  _point_segment_distance(b1, a0, a1))
    denominator = aa * cc - bb * bb
    if denominator > 1e-20:
        s = (bb * ee - cc * dd) / denominator
        t = (aa * ee - bb * dd) / denominator
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            nearest = min(nearest, math.dist(
                [p + s * d for p, d in zip(a0, u)],
                [p + t * d for p, d in zip(b0, v)]))
    return nearest


def _major_wood_clearance(a, a_asset, b, b_asset):
    """Conservative radial gap or exact capsule gap for nearby structural wood."""
    a_wood, b_wood = a_asset["major_wood"], b_asset["major_wood"]
    if not a_wood or not b_wood:
        return math.inf
    radial_gap = (math.dist(a["position"][:2], b["position"][:2]) -
                  a_asset["major_radius"] - b_asset["major_radius"])
    if radial_gap > MIN_MAJOR_WOOD_GAP_M:
        return radial_gap
    gap = math.inf
    for a0, a1, ar in a_wood:
        wa0, wa1 = _world(a0, a), _world(a1, a)
        for b0, b1, br in b_wood:
            wb0, wb1 = _world(b0, b), _world(b1, b)
            gap = min(gap, _segment_segment_distance(wa0, wa1, wb0, wb1) - ar - br)
    return gap


def _dense_separation(trees, lookup):
    crown_gap, wood_gap = math.inf, math.inf
    for i, a in enumerate(trees):
        for b in trees[i + 1:]:
            gap = math.dist(a["position"][:2], b["position"][:2]) - a["crown_radius_m"] - b["crown_radius_m"]
            crown_gap = min(crown_gap, gap)
            if gap < -MAX_CROWN_OVERLAP_M - 1e-9:
                raise ValueError("Presentation crowns overlap beyond limit")
            structural_gap = _major_wood_clearance(a, lookup[a["instance_id"]],
                                                    b, lookup[b["instance_id"]])
            wood_gap = min(wood_gap, structural_gap)
            if structural_gap < MIN_MAJOR_WOOD_GAP_M - 1e-9:
                raise ValueError("Major wood capsules lack clearance")
    return (crown_gap if math.isfinite(crown_gap) else None,
            wood_gap if math.isfinite(wood_gap) else None)


def _check_dense_candidate(candidate, asset, existing, lookup):
    """Check only new tree pairs; existing trees were checked on insertion."""
    for tree in existing:
        gap = (math.dist(candidate["position"][:2], tree["position"][:2]) -
               candidate["crown_radius_m"] - tree["crown_radius_m"])
        if gap < -MAX_CROWN_OVERLAP_M - 1e-9:
            raise ValueError("Presentation crowns overlap beyond limit")
        if (_major_wood_clearance(candidate, asset, tree,
                                  lookup[tree["instance_id"]]) < MIN_MAJOR_WOOD_GAP_M - 1e-9):
            raise ValueError("Major wood capsules lack clearance")


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


def generate_orchard_spec(pool_dir, seed, rows=None, slots_per_row=None, *,
                          _legacy_layout=False, _structured_v2=False, _structured_v3=False):
    """Return a JSON-serializable orchard with bounded, reproducible candidates.

    A prepared asset's selected fruit alone is eligible, because its USD omits
    exactly that fruit for a dynamic replacement. Reusing an asset is permitted.
    """
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1)
           for value in (rows, slots_per_row)):
        raise ValueError("rows and slots_per_row must be positive integers when specified")
    if _legacy_layout and (rows is None or slots_per_row is None):
        raise ValueError("Legacy layout requires explicit rows and slots_per_row")
    requested_rows, requested_slots_per_row = rows, slots_per_row
    seeds, rng = _streams(seed)
    assets = _pool_assets(pool_dir)
    dense = not (_legacy_layout or _structured_v2)
    coherent = dense and not _structured_v3
    along_min, along_max = DENSE_ALONG_GAP_BOUNDS_M if dense else ALONG_GAP_BOUNDS_M
    row_min, row_max = DENSE_ROW_GAP_BOUNDS_M if dense else ROW_GAP_BOUNDS_M
    if not dense and min(asset["radius"] for asset in assets) * 2 + MIN_CROWN_GAP_M > along_max:
        raise ValueError("No pool tree fits the along-row spacing range")
    if _legacy_layout:
        along_pitch = rng["layout"].uniform(along_min, along_max)
        row_pitch = rng["layout"].uniform(row_min, row_max)
        layout = None
    else:
        layout = _structured_layout(rng["layout"], rows, slots_per_row,
                                    dense=dense, coherent=coherent)
        rows, slot_counts, row_gaps, row_centers, along_gaps, row_xs, occupied = layout
        slots_per_row = max(slot_counts)
        along_pitch = min((gap for gaps in along_gaps for gap in gaps), default=along_min)
        row_pitch = min(row_gaps, default=row_min)
    # Reserve clearance for the smallest possible neighbour and opposing jitter.
    min_radius = min(asset["radius"] for asset in assets)
    if dense:
        selectable = [asset for asset in assets if asset["radius"] + min_radius -
                      MAX_CROWN_OVERLAP_M + 2 * DENSE_PLANTING_JITTER_M <= along_pitch]
    else:
        selectable = [asset for asset in assets if asset["radius"] + min_radius +
                      MIN_CROWN_GAP_M + (0.12 if _legacy_layout else 2 * PLANTING_JITTER_M) <= along_pitch]
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
        slots = range(slots_per_row) if _legacy_layout else occupied[row]
        for slot in slots:
            if _legacy_layout:
                local_x = (slot - (slots_per_row - 1) / 2) * along_pitch
                local_y = (row - (rows - 1) / 2) * row_pitch + rng["layout"].uniform(-0.06, 0.06)
            else:
                jitter = DENSE_PLANTING_JITTER_M if dense else PLANTING_JITTER_M
                local_x = row_xs[row][slot] + rng["layout"].uniform(-jitter, jitter)
                local_y = row_centers[row] + rng["layout"].uniform(-jitter, jitter)
            x, y = _rotate(local_x, local_y, global_yaw)
            instance_id = f"tree_r{row}_s{slot}"
            yaw = global_yaw + rng["layout"].uniform(-5.0, 5.0)
            choices = selectable.copy()
            rng["tree_selection"].shuffle(choices)
            if dense:
                # Retain seeded variation while drawing foliage-rich source
                # trees first when several assets satisfy the same slot.
                choices.sort(key=lambda asset: len(asset["tree"].get("leaves", [])) < 500)
            if coherent and not (row == 0 and slot == 0):
                # Shuffled ties preserve a separate selection stream. Favor
                # unused nearby silhouettes while allowing a geometrically
                # feasible repeat when the small asset pool requires it.
                used = {asset_id: sum(t["asset_id"] == asset_id for t in trees)
                        for asset_id in (asset["id"] for asset in choices)}
                choices.sort(key=lambda asset: (
                    sum(12 for t in trees if t["asset_id"] == asset["id"] and
                        t["row"] == row and t["slot"] == slot - 1) +
                    sum(4 for t in trees if t["asset_id"] == asset["id"] and
                        t["row"] == row - 1 and abs(t["slot"] - slot) <= 1),
                    len(asset["tree"].get("leaves", [])) < 500,
                    used[asset["id"]]))
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
                    if dense:
                        _check_dense_candidate(candidate, asset, trees, lookup)
                    else:
                        _separation(trees + [candidate])
                except ValueError:
                    if len(rejected) < 48:
                        rejected.append({"tree_instance_id": instance_id, "asset_id": asset["id"],
                                         "reason": "wood_or_crown_clearance" if dense else "crown_clearance"})
                    continue
                instance = candidate
                break
            if instance is None:
                raise ValueError(f"No source tree fits slot {instance_id} with native crown bounds")
            trees.append(instance)
            lookup[instance_id] = asset
    minimum_gap, minimum_wood_gap = (_dense_separation(trees, lookup) if dense else
                                     (_separation(trees), None))
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
                stance_margin = 0.02 if dense else 0.0
                if (body_gap <= 0 or basket_gap <= 0 or
                        foliage_gap <= stance_margin or robot_basket_gap <= stance_margin):
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
    if _legacy_layout:
        surfaces = _surfaces(rows, slots_per_row, along_pitch, row_pitch, global_yaw)
        camera_length = (slots_per_row - 1) * along_pitch + 3.0
        camera = {"eye": [0.0, -max(camera_length * 0.9, 10.0),
                          max(camera_length * 0.5, 5.4)], "target": [0.0, 0.0, 1.2]}
        structured_parameters = {}
    else:
        surfaces, footprint_length, footprint_width = _structured_surfaces(
            row_centers, row_xs, global_yaw,
            max(tree["crown_radius_m"] for tree in trees),
            DENSE_PLANTING_JITTER_M if dense else PLANTING_JITTER_M)
        camera_span = max(footprint_length, footprint_width)
        camera = {"eye": [0.0, -max(camera_span * 0.9, 10.0),
                          max(camera_span * 0.5, 5.4)], "target": [0.0, 0.0, 1.2]}
        structured_parameters = {
            "layout_model": ("structured_v4" if coherent else
                             "structured_v3" if dense else "structured_v2"),
            "requested_rows": requested_rows,
            "requested_slots_per_row": requested_slots_per_row,
            "row_count_range": list(DENSE_ROW_COUNT_BOUNDS if dense else ROW_COUNT_BOUNDS),
            "slots_per_row_range": list(DENSE_SLOT_COUNT_BOUNDS if dense else SLOT_COUNT_BOUNDS),
            "row_slot_counts": slot_counts,
            "occupied_slots": occupied,
            "row_gap_m": row_gaps,
            "row_centers_m": row_centers,
            "along_row_gap_m": along_gaps,
            "planting_jitter_m": DENSE_PLANTING_JITTER_M if dense else PLANTING_JITTER_M,
            "missing_slot_probability": (0.0 if coherent else
                                         DENSE_MISSING_SLOT_PROBABILITY if dense else
                                         MISSING_SLOT_PROBABILITY),
        }
        if dense:
            structured_parameters.update({"max_crown_overlap_m": MAX_CROWN_OVERLAP_M,
                                          "major_wood_radius_threshold_m": MAJOR_WOOD_RADIUS_M,
                                          "major_wood_gap_min_m": MIN_MAJOR_WOOD_GAP_M,
                                          "stance_foliage_and_basket_margin_m": 0.02})
    return {"schema": SCHEMA, "seed": seed, "substream_seeds": seeds,
            "parameters": {"rows": rows, "slots_per_row": slots_per_row,
                           "along_row_pitch_m": along_pitch, "row_pitch_m": row_pitch,
                           "global_yaw_deg": global_yaw,
                           "along_row_range_m": [along_min, along_max],
                           "row_range_m": [row_min, row_max],
                           "individual_yaw_range_deg": [-5.0, 5.0],
                           "row_lateral_jitter_m": DENSE_PLANTING_JITTER_M if dense else 0.06,
                           "crown_gap_min_m": MIN_CROWN_GAP_M,
                           "tree_scale": 1.0, "tree_up_axis": "Z",
                           "robot_stance_local_offset_m": list(stance_offset),
                           "robot_stance_candidate_offsets_m": [list(offset) for offset in STANCE_OFFSETS_M],
                           "robot_stance_jitter_m": 0.03,
                           "robot_yaw_relative_deg": 90.0,
                           "robot_yaw_jitter_deg": 3.0,
                           "basket_local_offset_m": list(basket_local),
                           "basket_candidate_offsets_m": [list(offset) for offset in BASKET_OFFSETS_M],
                           **structured_parameters},
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
            "infrastructure": {"surfaces": surfaces},
            "camera": camera,
            "lighting": _lighting(seed, rng["lighting"]),
            "feasibility": {"status": "proxy_only", "full_collision_ik_verified": False,
                            "minimum_crown_gap_m": minimum_gap,
                            **({"minimum_major_wood_gap_m": minimum_wood_gap} if dense else {}),
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
    if parameters.get("layout_model") in ("structured_v2", "structured_v3", "structured_v4"):
        # Reconstruct the original override choice: None means the corresponding
        # dimension was drawn from the layout stream.
        requested_rows = parameters.get("requested_rows")
        requested_slots = parameters.get("requested_slots_per_row")
        expected = generate_orchard_spec(pool_dir, spec["seed"], requested_rows, requested_slots,
                                         _structured_v2=parameters["layout_model"] == "structured_v2",
                                         _structured_v3=parameters["layout_model"] == "structured_v3")
    else:
        expected = generate_orchard_spec(pool_dir, spec["seed"],
                                         parameters["rows"], parameters["slots_per_row"],
                                         _legacy_layout=True)
    # JSON round-trip also rejects NaN and infinity, and normalizes integer keys.
    try:
        actual = json.loads(json.dumps(spec, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Non-JSON or nonfinite orchard specification") from exc
    if actual != expected:
        raise ValueError("Orchard specification differs from deterministic validated assets and parameters")
    return True
