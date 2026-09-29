"""Original, bounded orchard tree visuals. No Isaac or USD Python dependency.

All coordinates are environment-local metres. The generated asset is visual only:
the orchard's existing trunk, three scaffold branches, and stems own collision.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random


Vec3 = tuple[float, float, float]
PARAMS = {
    "version": 1,
    "trunk_height": 1.60,
    "trunk_base_radius": 0.073,
    "branch_sides": 7,
    "secondary_levels": 3,
    "leaf_count_target": 320,
    "target_clearance_radius": 0.20,
}


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _mul(a, scale):
    return tuple(x * scale for x in a)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _unit(a):
    length = math.sqrt(_dot(a, a))
    if length < 1e-9:
        raise ValueError("zero length direction")
    return _mul(a, 1 / length)


@dataclass
class Mesh:
    points: list[Vec3] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)

    def triangle(self, a, b, c):
        self.faces.append((a, b, c))


@dataclass
class TreeGeometry:
    wood: Mesh
    leaves: tuple[Mesh, Mesh, Mesh]
    target: Vec3
    secondary_segments: tuple[tuple[Vec3, Vec3], ...]
    petiole_segments: tuple[tuple[Vec3, Vec3], ...]
    leaf_centers: tuple[Vec3, ...]


def _tube(mesh, start, end, r0, r1, rng, sides=7, roughness=0.035):
    """Append a closed tapered twig, with radial perturbation and outward winding."""
    axis = _unit(_sub(end, start))
    u = _unit(_cross(axis, (0.0, 1.0, 0.0))
              if abs(axis[1]) < 0.85 else _cross(axis, (1.0, 0.0, 0.0)))
    v = _cross(axis, u)
    base = len(mesh.points)
    phase = rng.uniform(-0.1, 0.1)
    for center, radius in ((start, r0), (end, r1)):
        for i in range(sides):
            angle = phase + 2 * math.pi * i / sides
            radial = _add(_mul(u, math.cos(angle)), _mul(v, math.sin(angle)))
            mesh.points.append(_add(center, _mul(radial, radius * rng.uniform(1-roughness, 1+roughness))))
    for i in range(sides):
        j = (i + 1) % sides
        mesh.triangle(base+i, base+j, base+sides+j)
        mesh.triangle(base+i, base+sides+j, base+sides+i)
    lower_center, upper_center = len(mesh.points), len(mesh.points)+1
    mesh.points.extend((start, end))
    for i in range(sides):
        j = (i + 1) % sides
        mesh.triangle(lower_center, base+j, base+i)
        mesh.triangle(upper_center, base+sides+i, base+sides+j)


def _leaf(mesh, center, direction, rng):
    """A small lanceolate six-vertex leaf with asymmetric edges and a bent tip."""
    length = rng.uniform(0.045, 0.083)
    width = length * rng.uniform(0.26, 0.40)
    along = _unit(direction)
    side = _unit(_cross(along, (0.0, 0.0, 1.0))
                 if abs(along[2]) < 0.9 else _cross(along, (0.0, 1.0, 0.0)))
    normal = _unit(_cross(along, side))
    bend = rng.uniform(-0.14, 0.14) * length
    base = len(mesh.points)
    # Along the midrib: heel, two shoulders, midrib nodes, and pointed tip.
    offsets = ((-0.50, 0.00, 0.00), (-0.08, -0.44, 0.02),
               (0.01, 0.60, -0.02), (0.12, 0.00, 0.09),
               (0.34, -0.31, 0.04), (0.38, 0.34, -0.01),
               (0.56, 0.02, 0.12))
    for t, lateral, curl in offsets:
        p = _add(center, _add(_mul(along, t*length),
                  _add(_mul(side, lateral*width + bend*t*t),
                       _mul(normal, curl*length))))
        mesh.points.append(p)
    for face in ((0, 1, 3), (0, 3, 2), (1, 4, 3),
                 (3, 5, 2), (3, 4, 6), (3, 6, 5)):
        mesh.triangle(*(base+i for i in face))


def _outside_target(point, target, radius):
    return math.dist(point, target) >= radius


def _segment_distance(start, end, point):
    delta = _sub(end, start)
    t = max(0.0, min(1.0, _dot(_sub(point, start), delta) / _dot(delta, delta)))
    return math.dist(_add(start, _mul(delta, t)), point)


def generate_tree(layout, seed=42):
    """Generate deterministic visual meshes around an OrchardLayout.

    The trunk/scaffold occupy the same locations as the environment's physical
    cylinders. Fine wood and leaves are omitted inside a 20 cm target sphere.
    """
    rng = random.Random(seed)
    wood = Mesh()
    leaves = (Mesh(), Mesh(), Mesh())
    target = tuple(layout.target)
    root = (layout.trunk[0], layout.trunk[1], 0.0)
    segments = []
    petioles = []
    tips = []
    # A subtly wandering trunk encloses the existing 5.5 cm collider.
    for i in range(8):
        z0, z1 = i*0.20, (i+1)*0.20
        def trunk_point(z):
            shift = 0.0015 * (z / 1.6)**2
            return (root[0] + shift*math.sin(z*4),
                    root[1] + shift*math.cos(z*3), z)
        _tube(wood, trunk_point(z0), trunk_point(z1),
              PARAMS["trunk_base_radius"]-0.008*z0/1.6,
              PARAMS["trunk_base_radius"]-0.008*z1/1.6, rng)
    # Existing physical scaffold goes exactly to each fruit stem endpoint.
    for apple in layout.apples:
        endpoint = (apple[0], apple[1], apple[2]+0.13)
        origin = (root[0], root[1], endpoint[2])
        _tube(wood, origin, endpoint, 0.026, 0.021, rng, roughness=0.015)
    # Secondary levels depart from upper trunk and the two non-target scaffold
    # branches. Each child narrows and lifts, producing an open irregular crown.
    parents = []
    for i, apple in enumerate(layout.apples):
        if i == layout.target_index:
            continue
        endpoint = (apple[0], apple[1], apple[2]+0.13)
        for fraction in (0.44, 0.76):
            origin = (root[0]+fraction*(endpoint[0]-root[0]),
                      root[1]+fraction*(endpoint[1]-root[1]), endpoint[2])
            parents.append((origin, 0.018, 0))
    parents.extend([((root[0], root[1], z), 0.020, 0)
                    for z in (1.24, 1.43, 1.58)])
    for level in range(PARAMS["secondary_levels"]):
        children = []
        for origin, radius, _ in parents:
            for _child in range(2 if level < 2 else 1):
                azimuth = rng.uniform(-math.pi, math.pi)
                distance = rng.uniform(0.14, 0.27) * (0.79**level)
                lift = rng.uniform(0.07, 0.18) * (0.80**level)
                end = (origin[0]+distance*math.cos(azimuth),
                       origin[1]+distance*math.sin(azimuth), origin[2]+lift)
                if not (-0.53 < end[0] < 0.69 and -0.39 < end[1] < 0.62
                        and 1.12 < end[2] < 2.05):
                    continue
                if _segment_distance(origin, end, target) < (
                    PARAMS["target_clearance_radius"] + radius
                ):
                    continue
                next_radius = max(0.003, radius*rng.uniform(0.57, 0.70))
                _tube(wood, origin, end, radius, next_radius, rng)
                segments.append((origin, end))
                children.append((end, next_radius, level+1))
                if level >= 1:
                    tips.append(end)
        parents = children
    tips.extend(p[0] for p in parents)
    # A bounded set of individual leaves, concentrated at twigs and sparse
    # toward the lower/robot-facing part of the tree.
    centers = []
    attempts = 0
    while len(centers) < PARAMS["leaf_count_target"] and attempts < 5000:
        attempts += 1
        if not tips:
            break
        tip = tips[rng.randrange(len(tips))]
        center = (tip[0]+rng.uniform(-0.08, 0.08),
                  tip[1]+rng.uniform(-0.08, 0.08),
                  tip[2]+rng.uniform(-0.06, 0.09))
        if not (-0.58 < center[0] < 0.74 and -0.45 < center[1] < 0.68
                and 1.18 < center[2] < 2.15):
            continue
        if not _outside_target(center, target, PARAMS["target_clearance_radius"]+0.06):
            continue
        direction = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.3, 0.8))
        mesh = leaves[rng.randrange(len(leaves))]
        before = len(mesh.points)
        _leaf(mesh, center, direction, rng)
        heel = mesh.points[before]
        if (all(_outside_target(p, target, PARAMS["target_clearance_radius"])
                for p in mesh.points[before:])
                and _segment_distance(tip, heel, target) >=
                PARAMS["target_clearance_radius"] + 0.002):
            _tube(wood, tip, heel, 0.002, 0.001, rng, sides=5, roughness=0)
            petioles.append((tip, heel))
            centers.append(center)
        else:
            del mesh.points[before:]
            del mesh.faces[-6:]
    return TreeGeometry(wood, leaves, target, tuple(segments),
                        tuple(petioles), tuple(centers))


def _usd_mesh(name, mesh, material):
    points = ", ".join(f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})" for p in mesh.points)
    indices = ", ".join(str(i) for face in mesh.faces for i in face)
    counts = ", ".join("3" for _ in mesh.faces)
    return (f'    def Mesh "{name}" (\n'
            '        prepend apiSchemas = ["MaterialBindingAPI"]\n'
            '    )\n    {\n'
            f'        point3f[] points = [{points}]\n'
            f'        int[] faceVertexCounts = [{counts}]\n'
            f'        int[] faceVertexIndices = [{indices}]\n'
            '        uniform token subdivisionScheme = "none"\n'
            '        bool doubleSided = true\n'
            f'        rel material:binding = </Tree/Looks/{material}>\n'
            '    }\n')


def _usd_material(name, color, roughness):
    return (f'        def Material "{name}"\n        {{\n'
            f'            token outputs:surface.connect = </Tree/Looks/{name}/Surface.outputs:surface>\n'
            '            def Shader "Surface"\n            {\n'
            '                uniform token info:id = "UsdPreviewSurface"\n'
            f'                color3f inputs:diffuseColor = ({color[0]}, {color[1]}, {color[2]})\n'
            f'                float inputs:roughness = {roughness}\n'
            '                token outputs:surface\n            }\n        }\n')


def write_tree_asset(output_dir, layout, seed=42):
    """Write tree.usda and manifest.json into an explicit output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    geometry = generate_tree(layout, seed)
    usda = ('#usda 1.0\n(\n    defaultPrim = "Tree"\n'
            '    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
            'def Xform "Tree"\n{\n'
            + _usd_mesh("Wood", geometry.wood, "Bark")
            + ''.join(_usd_mesh(f"Leaves{i}", mesh, f"Leaf{i}")
                      for i, mesh in enumerate(geometry.leaves))
            + '    def Scope "Looks"\n    {\n'
            + _usd_material("Bark", (0.25, 0.15, 0.08), 0.88)
            + _usd_material("Leaf0", (0.15, 0.29, 0.07), 0.76)
            + _usd_material("Leaf1", (0.21, 0.36, 0.08), 0.73)
            + _usd_material("Leaf2", (0.11, 0.25, 0.06), 0.80)
            + '    }\n}\n')
    asset_path = output_dir / "tree.usda"
    asset_path.write_text(usda, encoding="utf-8")
    manifest = {
        "generator": "vla_isaaclab.envs.orchard_pick.tree_geometry",
        "provenance": "original procedural geometry; no third-party tree asset",
        "layout_seed": layout.seed,
        "layout": json.loads(json.dumps(layout.metadata())),
        "tree_seed": seed,
        "parameters": PARAMS,
        "asset": asset_path.name,
        "tree_file": asset_path.name,
        "sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
        "collision_status": "visual_only_existing_scaffold",
        "wood_triangles": len(geometry.wood.faces),
        "leaf_triangles": sum(len(mesh.faces) for mesh in geometry.leaves),
        "leaf_count": len(geometry.leaf_centers),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest
