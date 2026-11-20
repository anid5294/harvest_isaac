"""Generated USD assets for the G1 harvesting benchmark (pure Python, no simulator needed).

Everything here is deterministic and written as plain-text USDA into
``harvest_rl/assets/generated/`` the first time it is needed (or with
``scripts/build_g1_harvest_assets.py``):

* **Tree library** (:func:`ensure_tree_library`): variants of the OrchardBench seed-42 tree
  (``tree.json``) with different heights, canopy spreads, mirrored branch arrangements and
  pruned laterals. Each variant is a *kinematic* rigid body (so it can be re-posed at reset)
  with one convex-hull collider per branch, a merged visual wood mesh and merged visual
  leaves, and no apples (the physical apples are spawned by the env). Each variant also gets
  a list of **apple sites**: stem anchors on the underside of thin branches where a hanging
  apple of the largest size clears all wood, spaced so that no two apples overlap.
* **Apple sizes** (:func:`ensure_apple_sizes`): the YCB apple scaled about its centre.
* **G1 with a torso-mounted basket** (:func:`ensure_robot_with_basket`): harvest_rl's G1 USD
  referenced unchanged, plus a ``basket_link`` rigid body (floor + 4 walls) welded to
  ``torso_link`` with a fixed joint inside the articulation. The basket therefore moves with
  the torso under any motion, its mass is part of the robot, and its pose comes from the
  robot's own kinematics.

Each generated prim root carries ``custom`` attributes (``harvest:tree_variant``,
``harvest:apple_scale``) so the env can read back which prototype each env received.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..tree import APPLE_DIR, APPLE_USD, ASSET_DIR, TREE_JSON

GENERATED_DIR = ASSET_DIR / "generated"
GENERATOR_VERSION = 4

APPLE_HALF_EXTENTS = tuple(json.loads((APPLE_DIR / "half_extents.json").read_text())["half_extents_m"])
STEM_LENGTH = 0.025
"""Exposed stem length [m] (tree.json)."""

# ---------------------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------------------


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _add(a, b):
    return [x + y for x, y in zip(a, b)]


def _mul(a, s):
    return [x * s for x in a]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    return _mul(a, 1.0 / n) if n > 1e-12 else [0.0, 0.0, 1.0]


def _quat_rotate(q, v):
    """Rotate v by quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    u = [x, y, z]
    t = _mul(_cross(u, v), 2.0)
    return _add(_add(v, _mul(t, w)), _cross(u, t))


def segment_distance(p, a, b):
    """Distance from point p to segment ab, and the segment parameter."""
    d = _sub(b, a)
    dd = _dot(d, d)
    t = 0.0 if dd < 1e-12 else max(0.0, min(1.0, _dot(_sub(p, a), d) / dd))
    return _norm(_sub(p, _add(a, _mul(d, t)))), t


def _fmt(v):
    return "(" + ", ".join(f"{c:.6g}" for c in v) + ")"


def _fmt_list(vs):
    return "[" + ", ".join(_fmt(v) for v in vs) + "]"


# ---------------------------------------------------------------------------------------
# Tree variants
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeVariantSpec:
    name: str
    seed: int
    height_scale: float = 1.0
    """Scale of all heights (z) about the ground."""
    canopy_scale: float = 1.0
    """Scale of the horizontal spread (x, y) about the trunk."""
    prune_fraction: float = 0.0
    """Fraction of lateral shoots (order >= 3 chains) removed together with their sub-branches."""
    mirror: bool = False
    """Mirror the branch arrangement (x -> -x)."""
    split: str = "train"
    """``"train"`` or ``"heldout"`` (never seen in training; stage-6 evaluation)."""


TREE_LIBRARY: tuple[TreeVariantSpec, ...] = (
    TreeVariantSpec("seed42", seed=0),
    TreeVariantSpec("tall_narrow", seed=1, height_scale=1.12, canopy_scale=0.90, prune_fraction=0.10),
    TreeVariantSpec("short_wide_mirror", seed=2, height_scale=0.90, canopy_scale=1.15, prune_fraction=0.10, mirror=True),
    TreeVariantSpec("pruned_mirror", seed=3, prune_fraction=0.30, mirror=True),
    TreeVariantSpec("wide_dense", seed=4, height_scale=1.05, canopy_scale=1.08),
    TreeVariantSpec("compact_mirror", seed=5, height_scale=0.94, canopy_scale=0.94, prune_fraction=0.18, mirror=True),
    TreeVariantSpec("heldout_tall_sparse", seed=11, height_scale=1.15, canopy_scale=1.10, prune_fraction=0.28, mirror=True, split="heldout"),
    TreeVariantSpec("heldout_low_dense", seed=12, height_scale=0.88, canopy_scale=0.92, prune_fraction=0.05, split="heldout"),
)
"""The tree library, ordered so that stage configs can grow variety by taking a longer prefix."""


@dataclass
class TreeVariant:
    spec: TreeVariantSpec
    branches: list[dict]
    leaves: list[dict]
    sites: list[list[float]] = field(default_factory=list)
    """Apple stem anchors (x, y, z) in the tree's frame [m]."""
    branch_points: list[list[float]] = field(default_factory=list)
    """Points along every branch (x, y, z, radius) for clearance checks [m]."""


def _transform_point(p, spec: TreeVariantSpec):
    x, y, z = p
    if spec.mirror:
        x = -x
    return [x * spec.canopy_scale, y * spec.canopy_scale, z * spec.height_scale]


def make_variant(tree: dict, spec: TreeVariantSpec) -> TreeVariant:
    rng = random.Random(spec.seed)
    branches = {int(b["id"]): b for b in tree["branches"]}
    children: dict[int, list[int]] = {}
    for b in branches.values():
        children.setdefault(int(b["parent_id"]), []).append(int(b["id"]))
    removed: set[int] = set()
    # Lateral shoots: a branch chain of order >= 3 cut at its base (where the order steps up).
    # The leader and scaffold limbs (order <= 2) are never pruned.
    laterals = sorted(
        i for i, b in branches.items()
        if int(b["order"]) >= 3 and int(branches.get(int(b["parent_id"]), b)["order"]) < int(b["order"])
    )
    for i in rng.sample(laterals, int(round(spec.prune_fraction * len(laterals)))):
        stack = [i]
        while stack:
            j = stack.pop()
            if j not in removed:
                removed.add(j)
                stack.extend(children.get(j, []))
    out_branches = []
    for i, b in sorted(branches.items()):
        if i in removed:
            continue
        out_branches.append({
            "id": i, "parent_id": int(b["parent_id"]), "order": int(b["order"]),
            "start": _transform_point(b["start"], spec), "end": _transform_point(b["end"], spec),
            "radius_start": float(b["radius_start"]), "radius_end": float(b["radius_end"]),
        })
    out_leaves = []
    for leaf in tree.get("leaves", []):
        if int(leaf["parent_branch_id"]) in removed:
            continue
        # Leaf blade as a diamond in the source frame, then transformed like the wood.
        q = leaf["orientation_xyzw"]
        base = leaf["position"]
        along = _quat_rotate(q, [0.0, 0.0, 1.0])
        side = _quat_rotate(q, [1.0, 0.0, 0.0])
        length, width = float(leaf["length"]), float(leaf["width"])
        mid = _add(base, _mul(along, 0.45 * length))
        pts = [base, _add(mid, _mul(side, 0.5 * width)), _add(base, _mul(along, length)), _sub(mid, _mul(side, 0.5 * width))]
        out_leaves.append({"points": [_transform_point(p, spec) for p in pts], "class": int(leaf.get("mesh_class", 0))})
    v = TreeVariant(spec, out_branches, out_leaves)
    v.branch_points = _branch_points(out_branches)
    v.sites = _apple_sites(out_branches, rng)
    return v


def _branch_points(branches, step=0.05):
    pts = []
    for b in branches:
        length = _norm(_sub(b["end"], b["start"]))
        n = max(2, int(math.ceil(length / step)) + 1)
        for k in range(n):
            t = k / (n - 1)
            p = _add(b["start"], _mul(_sub(b["end"], b["start"]), t))
            pts.append(p + [b["radius_start"] + t * (b["radius_end"] - b["radius_start"])])
    return pts


MAX_APPLE_SCALE = 1.15
"""Largest apple scale the sites must fit (see ``G1HarvestEnvCfg.apple_scales``)."""


def apple_clearance_radius() -> float:
    return max(APPLE_HALF_EXTENTS) * MAX_APPLE_SCALE


def hanging_center(anchor, scale=MAX_APPLE_SCALE):
    """Centre of an apple of ``scale`` hanging straight down from ``anchor``."""
    return [anchor[0], anchor[1], anchor[2] - STEM_LENGTH - APPLE_HALF_EXTENTS[2] * scale]


def _apple_sites(branches, rng, clearance_margin=0.008, min_height=0.55, max_height=2.3):
    """Stem anchors under thin branches where the largest apple hangs clear of all wood."""
    r_apple = apple_clearance_radius()
    candidates = []
    for b in branches:
        if b["order"] < 1 or b["radius_end"] > 0.03:
            continue
        d = _sub(b["end"], b["start"])
        if _norm(d) < 0.06:
            continue
        for t in (0.25, 0.45, 0.65, 0.85):
            t = min(0.95, max(0.05, t + rng.uniform(-0.08, 0.08)))
            p = _add(b["start"], _mul(d, t))
            r = b["radius_start"] + t * (b["radius_end"] - b["radius_start"])
            anchor = [p[0], p[1], p[2] - r]
            if not (min_height < anchor[2] < max_height):
                continue
            c = hanging_center(anchor)
            stem_mid = [anchor[0], anchor[1], anchor[2] - 0.5 * STEM_LENGTH]
            ok = True
            for o in branches:
                dist, _ = segment_distance(c, o["start"], o["end"])
                if dist - max(o["radius_start"], o["radius_end"]) < r_apple + clearance_margin:
                    ok = False
                    break
                if o is not b:
                    dist, _ = segment_distance(stem_mid, o["start"], o["end"])
                    if dist - max(o["radius_start"], o["radius_end"]) < 0.004:
                        ok = False
                        break
            if ok:
                candidates.append(anchor)
    rng.shuffle(candidates)
    sites = []
    min_gap = 2.0 * r_apple + 0.01
    for a in candidates:
        if all(_norm(_sub(a, s)) > min_gap for s in sites):
            sites.append(a)
    return sites


# ---------------------------------------------------------------------------------------
# USDA writers
# ---------------------------------------------------------------------------------------


def _frustum(start, end, r0, r1, sides):
    axis = _unit(_sub(end, start))
    ref = [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = _unit(_cross(axis, ref))
    w = _cross(axis, u)
    pts = []
    for c, r in ((start, r0), (end, r1)):
        for k in range(sides):
            a = 2.0 * math.pi * k / sides
            pts.append(_add(c, _add(_mul(u, r * math.cos(a)), _mul(w, r * math.sin(a)))))
    faces = []
    for k in range(sides):
        k1 = (k + 1) % sides
        faces += [k, k1, sides + k1, k, sides + k1, sides + k]
    for k in range(1, sides - 1):  # caps
        faces += [0, k + 1, k, sides, sides + k, sides + k + 1]
    return pts, faces


_MATERIALS = '''    def "Materials"
    {{
        def Material "Bark"
        {{
            token outputs:surface.connect = </Tree/Materials/Bark/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {bark}
                float inputs:roughness = 0.85
                token outputs:surface
            }}
        }}
        def Material "Leaf"
        {{
            token outputs:surface.connect = </Tree/Materials/Leaf/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {leaf}
                float inputs:roughness = 0.8
                token outputs:surface
            }}
        }}
    }}
'''


def write_tree_usda(variant: TreeVariant, index: int, path: Path) -> None:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Tree"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "Tree" (',
        '    prepend apiSchemas = ["PhysicsRigidBodyAPI"]',
        ")",
        "{",
        "    bool physics:kinematicEnabled = 1",
        f"    custom int harvest:tree_variant = {index}",
        f'    custom string harvest:tree_name = "{variant.spec.name}"',
        _MATERIALS.format(bark="(0.26, 0.13, 0.055)", leaf="(0.12, 0.32, 0.05)"),
    ]
    # Visual wood: one merged smooth mesh.
    pts, idx = [], []
    for b in variant.branches:
        p, f = _frustum(b["start"], b["end"], b["radius_start"], b["radius_end"], 10)
        idx += [i + len(pts) for i in f]
        pts += p
    lines += [
        '    def Mesh "WoodVisual" (',
        '        prepend apiSchemas = ["MaterialBindingAPI"]',
        "    )",
        "    {",
        "        uniform bool doubleSided = 1",
        f"        int[] faceVertexCounts = [{', '.join(['3'] * (len(idx) // 3))}]",
        f"        int[] faceVertexIndices = [{', '.join(map(str, idx))}]",
        f"        point3f[] points = {_fmt_list(pts)}",
        "        rel material:binding = </Tree/Materials/Bark>",
        '        uniform token subdivisionScheme = "none"',
        "    }",
    ]
    # Leaves: one merged visual mesh, no collision.
    pts, idx = [], []
    for leaf in variant.leaves:
        n = len(pts)
        pts += leaf["points"]
        idx += [n, n + 1, n + 2, n, n + 2, n + 3]
    if pts:
        lines += [
            '    def Mesh "Leaves" (',
            '        prepend apiSchemas = ["MaterialBindingAPI"]',
            "    )",
            "    {",
            "        uniform bool doubleSided = 1",
            f"        int[] faceVertexCounts = [{', '.join(['3'] * (len(idx) // 3))}]",
            f"        int[] faceVertexIndices = [{', '.join(map(str, idx))}]",
            f"        point3f[] points = {_fmt_list(pts)}",
            "        rel material:binding = </Tree/Materials/Leaf>",
            '        uniform token subdivisionScheme = "none"',
            "    }",
        ]
    # Collision: one invisible convex hull per branch (the wood the robot can hit).
    lines += ['    def Scope "WoodColliders"', "    {"]
    for b in variant.branches:
        p, f = _frustum(b["start"], b["end"], b["radius_start"], b["radius_end"], 8)
        lines += [
            f'        def Mesh "C{b["id"]}" (',
            '            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]',
            "        )",
            "        {",
            f"            int[] faceVertexCounts = [{', '.join(['3'] * (len(f) // 3))}]",
            f"            int[] faceVertexIndices = [{', '.join(map(str, f))}]",
            f"            point3f[] points = {_fmt_list(p)}",
            '            uniform token physics:approximation = "convexHull"',
            '            token visibility = "invisible"',
            "        }",
        ]
    lines += ["    }", "}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _library_key() -> str:
    h = hashlib.sha256()
    h.update(TREE_JSON.read_bytes())
    h.update(json.dumps([asdict(s) for s in TREE_LIBRARY], sort_keys=True).encode())
    h.update(f"{GENERATOR_VERSION}:{MAX_APPLE_SCALE}:{APPLE_HALF_EXTENTS}".encode())
    return h.hexdigest()[:16]


def ensure_tree_library(out_dir: Path | None = None) -> dict:
    """Build (if needed) and return the tree library index.

    Returns ``{"key", "variants": [{"name", "split", "usd", "sites", "branch_points", ...}]}``
    with sites/points in each tree's own frame (trunk base at the origin, metres, Z up).
    """
    out_dir = Path(out_dir or GENERATED_DIR / "tree_library")
    index_path = out_dir / "library.json"
    key = _library_key()
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index.get("key") == key and all(Path(v["usd"]).exists() for v in index["variants"]):
            return index
    tree = json.loads(TREE_JSON.read_text())
    variants = []
    for i, spec in enumerate(TREE_LIBRARY):
        v = make_variant(tree, spec)
        usd = out_dir / spec.name / "tree.usda"
        write_tree_usda(v, i, usd)
        heights = [p[2] for p in v.branch_points]
        variants.append({
            "index": i, "name": spec.name, "split": spec.split, "spec": asdict(spec), "usd": str(usd),
            "sites": v.sites, "branch_points": v.branch_points,
            "num_branches": len(v.branches), "num_leaves": len(v.leaves), "height": max(heights),
        })
    index = {"key": key, "generator_version": GENERATOR_VERSION, "variants": variants}
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index


# ---------------------------------------------------------------------------------------
# Apple sizes
# ---------------------------------------------------------------------------------------


def ensure_apple_sizes(scales: tuple[float, ...]) -> list[str]:
    """One USDA per apple scale: the YCB physics apple with its mesh scaled about its centre."""
    out = []
    for s in scales:
        path = GENERATED_DIR / "apples" / f"apple_s{int(round(s * 100)):03d}.usda"
        rel = Path("..") / ".." / APPLE_USD.relative_to(ASSET_DIR)
        text = "\n".join([
            "#usda 1.0",
            "(",
            '    defaultPrim = "Apple"',
            "    metersPerUnit = 1",
            '    upAxis = "Z"',
            ")",
            "",
            'def Xform "Apple" (',
            f"    prepend references = @{rel.as_posix()}@",
            ")",
            "{",
            f"    custom float harvest:apple_scale = {s}",
            '    over "Mesh"',
            "    {",
            f"        float3 xformOp:scale = ({s}, {s}, {s})",
            '        uniform token[] xformOpOrder = ["xformOp:scale"]',
            "    }",
            "}",
            "",
        ])
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        out.append(str(path))
    return out


# ---------------------------------------------------------------------------------------
# G1 with a torso-mounted basket
# ---------------------------------------------------------------------------------------


@dataclass
class BasketSpec:
    parent_body: str = "torso_link"
    """Robot link the basket is welded to."""
    offset: tuple[float, float, float] = (0.284, 0.0, -0.264)
    """Basket frame origin (centre of the floor's top surface) in the parent link frame [m].
    Default: in front of the torso on the body midline; back wall ~16 cm in front of the pelvis
    axis (clear of belly and thighs); rim 10 cm below the pelvis, under the resting fingers."""
    inner_size: tuple[float, float] = (0.22, 0.20)
    """Inner floor size along the basket x (robot forward) and y (robot left) [m]."""
    wall_height: float = 0.12
    thickness: float = 0.012
    mass: float = 0.5
    color: tuple[float, float, float] = (0.06, 0.25, 0.65)


def ensure_robot_with_basket(robot_usd: Path, spec: BasketSpec, parent_pos_in_root=(-0.0039635, 0.0, 0.044)) -> str:
    """USDA referencing ``robot_usd`` with ``basket_link`` welded to ``spec.parent_body``.

    ``parent_pos_in_root`` only seeds the basket's authored transform (the articulation
    solver places the link from the joint anyway).
    """
    key = hashlib.sha256(json.dumps([str(robot_usd), asdict(spec), GENERATOR_VERSION]).encode()).hexdigest()[:10]
    path = GENERATED_DIR / "robots" / f"g1_dex3_basket_{key}.usda"
    if path.exists():
        return str(path)
    rel = Path("..") / ".." / Path(robot_usd).resolve().relative_to(ASSET_DIR.resolve())
    hx, hy = spec.inner_size[0] / 2, spec.inner_size[1] / 2
    t, h = spec.thickness, spec.wall_height
    # (name, centre, size) in the basket frame (origin = top of the floor, at its centre).
    parts = [
        ("floor", (0.0, 0.0, -t / 2), (2 * hx + 2 * t, 2 * hy + 2 * t, t)),
        ("wall_back", (-hx - t / 2, 0.0, h / 2), (t, 2 * hy + 2 * t, h)),
        ("wall_front", (hx + t / 2, 0.0, h / 2), (t, 2 * hy + 2 * t, h)),
        ("wall_right", (0.0, -hy - t / 2, h / 2), (2 * hx, t, h)),
        ("wall_left", (0.0, hy + t / 2, h / 2), (2 * hx, t, h)),
    ]
    seed = [p + o for p, o in zip(parent_pos_in_root, spec.offset)]
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Robot"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "Robot" (',
        f"    prepend references = @{rel.as_posix()}@",
        ")",
        "{",
        '    def Xform "basket_link" (',
        '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
        "    )",
        "    {",
        f"        float physics:mass = {spec.mass}",
        f"        double3 xformOp:translate = {_fmt(seed)}",
        '        uniform token[] xformOpOrder = ["xformOp:translate"]',
    ]
    for name, c, size in parts:
        lines += [
            f'        def Cube "{name}" (',
            '            prepend apiSchemas = ["PhysicsCollisionAPI"]',
            "        )",
            "        {",
            "            double size = 1",
            f"            color3f[] primvars:displayColor = [{_fmt(spec.color)}]",
            f"            double3 xformOp:translate = {_fmt(c)}",
            f"            float3 xformOp:scale = {_fmt(size)}",
            '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
            "        }",
        ]
    lines += [
        "    }",
        "",
        '    def PhysicsFixedJoint "basket_mount_joint"',
        "    {",
        f"        rel physics:body0 = </Robot/{spec.parent_body}>",
        "        rel physics:body1 = </Robot/basket_link>",
        f"        point3f physics:localPos0 = {_fmt(spec.offset)}",
        "        quatf physics:localRot0 = (1, 0, 0, 0)",
        "        point3f physics:localPos1 = (0, 0, 0)",
        "        quatf physics:localRot1 = (1, 0, 0, 0)",
        "    }",
        "}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
