"""Seeded, narrow tall-spindle apple tree in environment-local metres.

The branch graph is also the intended collision skeleton. Foliage is visual
only, and every fruit hangs from a connected wood spur. The row runs along Y.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random


Vec3 = tuple[float, float, float]
TRUNK_XY = (0.12, 0.16)
FRUIT_RADIUS = 0.038
STEM_LENGTH = 0.025
TREE_HEIGHT = 2.85


@dataclass(frozen=True)
class Branch:
    start: Vec3
    end: Vec3
    radius_start: float
    radius_end: float
    parent: int
    kind: str


@dataclass(frozen=True)
class Leaf:
    anchor: Vec3
    center: Vec3
    direction: Vec3
    branch_index: int


@dataclass(frozen=True)
class Fruit:
    anchor: Vec3
    center: Vec3
    radius: float
    branch_index: int
    is_target: bool = False


@dataclass(frozen=True)
class TrellisSegment:
    start: Vec3
    end: Vec3
    radius: float
    kind: str


@dataclass(frozen=True)
class CommercialTree:
    seed: int
    branches: tuple[Branch, ...]
    leaves: tuple[Leaf, ...]
    fruits: tuple[Fruit, ...]
    trellis: tuple[TrellisSegment, ...]

    def metadata(self):
        return asdict(self)


def _lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return tuple(a[i] + t * (b[i] - a[i]) for i in range(3))


def _add(a: Vec3, b: Vec3) -> Vec3:
    return tuple(a[i] + b[i] for i in range(3))


def _unit(v: Vec3) -> Vec3:
    magnitude = math.sqrt(sum(x*x for x in v))
    return tuple(x/magnitude for x in v)


def generate_commercial_tree(seed: int = 42) -> CommercialTree:
    """Create one trained tree, including a stable robot-side target fruit.

    Leader diameter tapers from 8 to 5 cm. Short laterals incline slightly
    downwards; fine twig sprays carry the leaves. This is one tree with its own
    stake and wire span, suitable for repeating at orchard row spacing.
    """
    if type(seed) is not int:
        raise ValueError("Tree seed must be an integer")
    rng = random.Random(seed)
    branches: list[Branch] = []
    leaves: list[Leaf] = []
    fruits: list[Fruit] = []
    leader_at_height = {}

    def branch(start, end, r0, r1, parent, kind):
        if math.dist(start, end) < 1e-5 or not (0 < r1 <= r0):
            raise ValueError("Degenerate or expanding branch")
        if parent >= 0 and branches[parent].end != start:
            raise ValueError("Branch graph is disconnected")
        branches.append(Branch(start, end, r0, r1, parent, kind))
        return len(branches) - 1

    # A finely segmented leader permits exact lateral parent attachment.
    for i in range(19):
        z0, z1 = round(i*.15, 6), round((i+1)*.15, 6)
        start = (TRUNK_XY[0], TRUNK_XY[1], z0)
        end = (TRUNK_XY[0], TRUNK_XY[1], z1)
        index = branch(start, end, .040-.015*z0/TREE_HEIGHT,
                       .040-.015*z1/TREE_HEIGHT, i-1, "leader")
        leader_at_height[round(z1, 2)] = index

    def add_lateral(origin_z, first, end, fruit_center, target=False):
        parent = leader_at_height[round(origin_z, 2)]
        origin = branches[parent].end
        first_index = branch(origin, first, .014, .010, parent, "lateral")
        second_index = branch(first, end, .010, .006, first_index, "lateral")
        anchor = (fruit_center[0], fruit_center[1],
                  fruit_center[2] + FRUIT_RADIUS + STEM_LENGTH)
        spur_index = branch(end, anchor, .0055, .003, second_index, "fruit_spur")
        fruits.append(Fruit(anchor, fruit_center, FRUIT_RADIUS, spur_index, target))

        # Two short shoots from the outer lateral, each carrying attached leaves.
        for side in (-1, 1):
            shoot_start = second_index
            shoot_origin = end
            shoot_end = (end[0] + rng.uniform(-.025, .025),
                         end[1] + side*rng.uniform(.075, .115),
                         end[2] + rng.uniform(.025, .065))
            shoot = branch(shoot_origin, shoot_end, .0042, .002, shoot_start, "twig")
            for j in range(7):
                t = .16 + .12*j
                anchor_leaf = _lerp(shoot_origin, shoot_end, t)
                azimuth = rng.uniform(-math.pi, math.pi)
                displacement = (rng.uniform(.025, .055)*math.cos(azimuth),
                                rng.uniform(.025, .055)*math.sin(azimuth),
                                rng.uniform(-.025, .035))
                center = _add(anchor_leaf, displacement)
                # The target fruit stays visible to the approach camera.
                if math.dist(center, fruits[0].center) < .115:
                    continue
                direction = _unit((displacement[0], displacement[1],
                                   displacement[2] + .035))
                leaves.append(Leaf(anchor_leaf, center, direction, shoot))

    # Target route descends from the leader toward the G1 at (0,-.72,.8).
    target = (-.10 + rng.uniform(-.018, .018),
              -.29 + rng.uniform(-.015, .015),
              1.00 + rng.uniform(-.018, .018))
    target_anchor = (target[0], target[1], target[2] + FRUIT_RADIUS + STEM_LENGTH)
    add_lateral(1.20, (.035, -.035, 1.115),
                (target[0] + .014, target[1] + .026, target_anchor[2] + .008),
                target, True)

    # Spindle tiers on both sides of the leader. Along-row spread stays inside
    # the 0.9144 m tree spacing and transverse depth stays narrow.
    for tier, z in enumerate((1.05, 1.35, 1.65, 1.95, 2.25, 2.55)):
        for side in (-1, 1):
            along = rng.uniform(-.24, .24)
            reach = rng.uniform(.26, .35)
            end = (TRUNK_XY[0] + side*reach,
                   TRUNK_XY[1] + along,
                   z - rng.uniform(.06, .12))
            first = (TRUNK_XY[0] + side*reach*.48,
                     TRUNK_XY[1] + along*.44,
                     z - .035)
            # End of the lateral is near, but distinct from, the short spur.
            center = (end[0] + side*rng.uniform(.015, .035),
                      end[1] + rng.uniform(-.018, .018),
                      end[2] - FRUIT_RADIUS - STEM_LENGTH - .012)
            add_lateral(z, first, end, center)

    # A local support stake and four wire spans; global row posts can be added
    # independently at their commercial spacing rather than at every tree.
    trellis = [TrellisSegment((.17, .16, 0), (.17, .16, 2.9), .012, "stake")]
    for height in (.75, 1.50, 2.25, 2.75):
        trellis.append(TrellisSegment((.12, -.30, height),
                                      (.12, .62, height), .0018, "wire"))
    return CommercialTree(seed, tuple(branches), tuple(leaves),
                          tuple(fruits), tuple(trellis))


def write_foliage_usda(tree: CommercialTree, path) -> Path:
    """Write a lightweight, visual-only leaf mesh to an explicit USDA path."""
    path = Path(path)
    points: list[Vec3] = []
    faces: list[tuple[int, int, int]] = []
    for leaf in tree.leaves:
        along = leaf.direction
        side = _unit((-along[1], along[0], 0.0))
        heel = leaf.anchor
        center = leaf.center
        tip = _add(center, tuple(.035*x for x in along))
        left = _add(center, tuple(.012*x for x in side))
        right = _add(center, tuple(-.012*x for x in side))
        offset = len(points)
        points.extend((heel, left, tip, right))
        faces.extend(((offset, offset+1, offset+2),
                      (offset, offset+2, offset+3)))
    def fmt(p):
        return f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})"
    content = ('#usda 1.0\n(\n    defaultPrim = "Foliage"\n'
               '    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
               'def Xform "Foliage"\n{\n'
               '    def Mesh "Leaves" (\n'
               '        prepend apiSchemas = ["MaterialBindingAPI"]\n'
               '    )\n    {\n'
               f'        point3f[] points = [{", ".join(map(fmt, points))}]\n'
               f'        int[] faceVertexCounts = [{", ".join("3" for _ in faces)}]\n'
               f'        int[] faceVertexIndices = [{", ".join(str(i) for f in faces for i in f)}]\n'
               '        uniform token subdivisionScheme = "none"\n'
               '        bool doubleSided = true\n'
               '        rel material:binding = </Foliage/Looks/Leaf>\n'
               '    }\n'
               '    def Scope "Looks"\n    {\n'
               '        def Material "Leaf"\n        {\n'
               '            token outputs:surface.connect = </Foliage/Looks/Leaf/Surface.outputs:surface>\n'
               '            def Shader "Surface"\n            {\n'
               '                uniform token info:id = "UsdPreviewSurface"\n'
               '                color3f inputs:diffuseColor = (0.18, 0.34, 0.09)\n'
               '                float inputs:roughness = 0.8\n'
               '                token outputs:surface\n'
               '            }\n        }\n    }\n}\n')
    path.write_text(content, encoding="utf-8")
    return path
