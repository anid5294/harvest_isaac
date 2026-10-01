"""Small seeded harvesting workspace; coordinates are in the environment frame."""

from dataclasses import asdict, dataclass
import math
import random


@dataclass(frozen=True)
class OrchardLayout:
    seed: int
    apples: tuple
    target_index: int
    basket: tuple
    trunk: tuple = (0.12, 0.16, 0.80)
    radius: float = 0.038
    mass: float = 0.080
    stem_break_force: float = 6.0
    stem_anchors: tuple = ()
    tree_model: str = "legacy"

    @property
    def target(self):
        return self.apples[self.target_index]

    def metadata(self):
        return asdict(self)


def choose_target(apples):
    """Reject unreachable fruit and prefer the exposed fruit near the left palm."""
    reference = (-0.10, -0.30, 1.00)
    candidates = [
        (math.dist(point, reference), i) for i, point in enumerate(apples)
        if -0.23 <= point[0] <= 0.03 and -0.37 <= point[1] <= -0.22
        and 0.92 <= point[2] <= 1.10
    ]
    if not candidates:
        raise ValueError("No apple in the calibrated candidate workspace")
    return min(candidates)[1]


def make_layout(seed=42, tree_model="legacy"):
    if tree_model == "commercial":
        from .commercial_tree import generate_commercial_tree
        tree = generate_commercial_tree(seed)
        return OrchardLayout(
            seed, tuple(fruit.center for fruit in tree.fruits), 0,
            (-0.36, -0.30, 0.78),
            stem_anchors=tuple(fruit.anchor for fruit in tree.fruits),
            tree_model="commercial",
        )
    if tree_model != "legacy":
        raise ValueError(f"Unknown orchard tree model: {tree_model}")
    rng = random.Random(seed)
    apples = (
        (-0.10 + rng.uniform(-0.025, 0.025),
         -0.29 + rng.uniform(-0.02, 0.02), 1.00 + rng.uniform(-0.025, 0.025)),
        (0.32 + rng.uniform(-0.02, 0.02), 0.15, 1.26),
        (-0.24, 0.18, 1.43 + rng.uniform(-0.02, 0.02)),
    )
    basket = (-0.36 + rng.uniform(-0.015, 0.015), -0.30, 0.78)
    return OrchardLayout(seed, apples, choose_target(apples), basket)


def stem_geometry(layout, index):
    """One geometric source for stem collision and joint local frames.

    Fruit default orientation is identity. The joint at the fruit surface uses
    the stem's local +Z frame on both bodies, allowing non-vertical stems too.
    """
    center = layout.apples[index]
    anchor = (layout.stem_anchors[index] if layout.stem_anchors else
              (center[0], center[1], center[2]+0.13))
    delta = tuple(b-a for a,b in zip(center, anchor))
    distance = math.sqrt(sum(v*v for v in delta))
    if not math.isfinite(distance) or distance <= layout.radius+1e-6:
        raise ValueError("Stem attachment must lie outside the fruit")
    direction = tuple(v/distance for v in delta)
    fruit_offset = tuple(layout.radius*v for v in direction)
    bottom = tuple(a+b for a,b in zip(center, fruit_offset))
    if direction[2] < -1+1e-9:
        quaternion = (0.0, 1.0, 0.0, 0.0)
    else:
        raw = (1+direction[2], -direction[1], direction[0], 0.0)
        norm = math.sqrt(sum(v*v for v in raw))
        quaternion = tuple(v/norm for v in raw)
    return {"bottom": bottom, "anchor": anchor,
            "stem_joint_position": (0.0, 0.0, -(distance-layout.radius)/2),
            "fruit_joint_position": fruit_offset, "fruit_joint_rotation": quaternion}


def settled_in_basket(position, speed, basket, radius=0.038):
    """Require the entire sphere to be inside the walls and resting on the base."""
    return (
        all(math.isfinite(v) for v in (*position, speed))
        and abs(position[0] - basket[0]) < 0.15 - radius
        and abs(position[1] - basket[1]) < 0.13 - radius
        and abs(position[2] - (basket[2] + 0.01 + radius)) < 0.012
        and speed < 0.035
    )
