"""Read the OrchardBench tree description (tree.json) without starting the simulator.

The tree geometry and the fruit/stem parameters in ``tree.json`` were exported by the
teammate's harvest_isaac pipeline (OrchardBench seed 42). Her ``tree.usda`` contains the wood
colliders, leaves and 39 visual-only apples; it deliberately leaves out fruit 28, which this
environment spawns as a physical apple hanging on a breakable stem.

Apples are the YCB ``013_apple`` scan at its real size (about 7.5 cm across), built by
``scripts/build_ycb_apple.py``: ``tree_ycb.usda`` is her tree with the 39 visual apples swapped
for it, and the physical apple uses ``ycb_013_apple/apple_physics.usda``. Each apple hangs from
its tree.json stem anchor, so its centre is lower than tree.json's (smaller) sphere centre.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "assets"
TREE_DIR = ASSET_DIR / "tree_orchardbench_seed42"
TREE_USD = TREE_DIR / "isaac" / "tree_ycb.usda"
"""Her tree with YCB visual apples. Her original sphere-apple version is ``tree.usda``."""
TREE_JSON = TREE_DIR / "tree.json"
TREE_MANIFEST = TREE_DIR / "isaac" / "manifest.json"
APPLE_DIR = ASSET_DIR / "ycb_013_apple"
APPLE_USD = APPLE_DIR / "apple_physics.usda"


@dataclass(frozen=True)
class Fruit:
    """The physical apple, in the tree's own frame (metres, Z up)."""

    id: int
    center: tuple[float, float, float]
    """Apple (bounding-box) centre at rest, hanging from its stem [m]."""
    anchor: tuple[float, float, float]
    """Point where the stem joins the branch [m]."""
    half_extents: tuple[float, float, float]
    """Half size of the YCB apple along x, y, z [m]; z is the stem axis."""
    radius: float
    """Effective radius (mean half extent) used by the basket/settle checks [m]."""
    mass: float
    """Apple mass [kg], from tree.json."""
    stem_length: float
    """Exposed stem length [m]."""
    stem_break_force: float
    """Pull force at which the stem snaps [N]."""


def physical_fruit_id() -> int:
    """The fruit that tree.usda leaves out so it can be simulated as a rigid body."""
    return int(json.loads(TREE_MANIFEST.read_text())["selected_fruit_id"])


def load_fruit(fruit_id: int | None = None) -> Fruit:
    tree = json.loads(TREE_JSON.read_text())
    if tree.get("units") != "m" or tree.get("up_axis") != "Z":
        raise ValueError("tree.json must be in metres with Z up")
    if fruit_id is None:
        fruit_id = physical_fruit_id()
    elif fruit_id != physical_fruit_id():
        # Every other fruit already exists in the tree USD as a visual apple, and most of them
        # intersect branches. Picking another one needs a re-exported tree.usda.
        raise ValueError(
            f"tree.usda only leaves out fruit {physical_fruit_id()}; fruit {fruit_id} would be doubled."
        )
    fruit = next(f for f in tree["fruits"] if int(f["id"]) == fruit_id)
    stem = fruit["stem"]
    half = tuple(json.loads((APPLE_DIR / "half_extents.json").read_text())["half_extents_m"])
    # Hang the YCB apple so its top touches the end of the stem.
    anchor = fruit["anchor"]
    down = [c - a for a, c in zip(anchor, fruit["center"])]
    norm = math.sqrt(sum(v * v for v in down))
    center = tuple(a + v / norm * (stem["length"] + half[2]) for a, v in zip(anchor, down))
    return Fruit(
        id=fruit_id,
        center=center,
        anchor=tuple(anchor),
        half_extents=half,
        radius=sum(half) / 3.0,
        mass=float(stem["mass_kg"]),
        stem_length=float(stem["length"]),
        stem_break_force=float(stem["break_force_n"]),
    )
