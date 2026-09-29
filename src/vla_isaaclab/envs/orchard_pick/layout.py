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


def make_layout(seed=42):
    rng = random.Random(seed)
    apples = (
        (-0.10 + rng.uniform(-0.025, 0.025),
         -0.29 + rng.uniform(-0.02, 0.02), 1.00 + rng.uniform(-0.025, 0.025)),
        (0.32 + rng.uniform(-0.02, 0.02), 0.15, 1.26),
        (-0.24, 0.18, 1.43 + rng.uniform(-0.02, 0.02)),
    )
    basket = (-0.36 + rng.uniform(-0.015, 0.015), -0.30, 0.78)
    return OrchardLayout(seed, apples, choose_target(apples), basket)


def settled_in_basket(position, speed, basket, radius=0.038):
    """Require the entire sphere to be inside the walls and resting on the base."""
    return (
        all(math.isfinite(v) for v in (*position, speed))
        and abs(position[0] - basket[0]) < 0.15 - radius
        and abs(position[1] - basket[1]) < 0.13 - radius
        and abs(position[2] - (basket[2] + 0.01 + radius)) < 0.012
        and speed < 0.035
    )
