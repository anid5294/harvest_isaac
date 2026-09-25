"""Deterministic geometry for the first Isaac orchard scene gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True)
class Branch:
    center: Vec3
    length: float
    radius: float
    orientation_wxyz: Quat


@dataclass(frozen=True)
class Apple:
    position: Vec3
    radius: float
    color: Vec3


@dataclass(frozen=True)
class OrchardLayout:
    branches: tuple[Branch, ...]
    apples: tuple[Apple, ...]
    foliage: tuple[tuple[Vec3, float], ...]


def _normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude <= 1.0e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return tuple(value / magnitude for value in values)


def cylinder_between(start: Vec3, end: Vec3, radius: float) -> Branch:
    """Return a Z-axis cylinder transform spanning two endpoints."""
    delta = tuple(end[index] - start[index] for index in range(3))
    length = math.sqrt(sum(value * value for value in delta))
    if length <= 1.0e-6:
        raise ValueError("A branch must have two distinct endpoints")
    direction = tuple(value / length for value in delta)
    dot = direction[2]
    if dot < -1.0 + 1.0e-9:
        orientation = (0.0, 1.0, 0.0, 0.0)
    else:
        # Shortest rotation from local +Z to the branch direction.
        orientation = _normalize((1.0 + dot, -direction[1], direction[0], 0.0))
    center = tuple((start[index] + end[index]) * 0.5 for index in range(3))
    return Branch(center=center, length=length, radius=radius, orientation_wxyz=orientation)


def make_orchard_layout(seed: int = 42) -> OrchardLayout:
    """Build a compact tree around one reachable, front-facing apple."""
    branch_endpoints = (
        ((0.00, 0.28, 0.58), (-0.42, 0.13, 1.05), 0.055),
        ((0.00, 0.28, 0.82), (0.46, 0.16, 1.23), 0.050),
        ((0.00, 0.28, 1.08), (-0.36, 0.14, 1.49), 0.043),
        ((0.00, 0.28, 1.31), (0.34, 0.22, 1.72), 0.036),
        ((0.00, 0.28, 1.34), (0.05, 0.30, 1.98), 0.040),
        ((-0.22, 0.19, 1.14), (-0.10, -0.10, 1.29), 0.025),
    )
    branches = tuple(
        cylinder_between(start, end, radius)
        for start, end, radius in branch_endpoints
    )

    rng = random.Random(seed)
    nominal_apples = (
        (-0.10, -0.13, 1.27),  # index 0: future detachable task apple
        (-0.39, 0.10, 1.00),
        (0.43, 0.13, 1.20),
        (-0.34, 0.11, 1.47),
        (0.31, 0.19, 1.69),
        (0.04, 0.27, 1.91),
    )
    colors = ((0.78, 0.03, 0.02), (0.92, 0.22, 0.02), (0.78, 0.12, 0.01))
    apples = []
    for index, nominal in enumerate(nominal_apples):
        jitter = (
            (0.0, 0.0, 0.0)
            if index == 0
            else tuple(rng.uniform(-0.025, 0.025) for _ in range(3))
        )
        apples.append(
            Apple(
                position=tuple(nominal[axis] + jitter[axis] for axis in range(3)),
                radius=0.042,
                color=colors[index % len(colors)],
            )
        )

    foliage = (
        ((-0.34, 0.16, 1.10), 0.24),
        ((0.37, 0.20, 1.30), 0.25),
        ((-0.29, 0.17, 1.52), 0.22),
        ((0.23, 0.25, 1.70), 0.22),
        ((0.04, 0.30, 1.92), 0.20),
    )
    return OrchardLayout(branches=branches, apples=tuple(apples), foliage=foliage)
