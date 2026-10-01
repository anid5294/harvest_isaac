"""Physical scaffold and visual foliage for the generated commercial tree."""

from pathlib import Path
import tempfile

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from ..orchard_preview.env_cfg import _branch
from ..orchard_preview.layout import cylinder_between
from .commercial_tree import generate_commercial_tree, write_foliage_usda


def _interpolate(start, end, fraction):
    return tuple(a + (b - a) * fraction for a, b in zip(start, end))


def add_commercial_tree(scene, seed, add_asset):
    """Add collidable tapered wood/support approximations and visual-only leaves.

    ``add_asset`` registers a scene config and tracks its name for reconfiguration.
    The tree generator is deterministic for a seed, as is the corresponding layout.
    """
    tree = generate_commercial_tree(seed)
    for index, branch in enumerate(tree.branches):
        middle = _interpolate(branch.start, branch.end, 0.5)
        for part, (start, end, radius) in enumerate((
            (branch.start, middle, (3 * branch.radius_start + branch.radius_end) / 4),
            (middle, branch.end, (branch.radius_start + 3 * branch.radius_end) / 4),
        )):
            add_asset(
                f"commercial_branch_{index}_{part}",
                _branch(f"CommercialBranch_{index}_{part}",
                        cylinder_between(start, end, radius)),
            )

    for index, support in enumerate(tree.trellis):
        add_asset(
            f"commercial_trellis_{index}",
            _branch(f"CommercialTrellis_{index}",
                    cylinder_between(support.start, support.end, support.radius)),
        )

    foliage_dir = Path(tempfile.mkdtemp(prefix="orchard_foliage_"))
    foliage_path = foliage_dir / "foliage.usda"
    write_foliage_usda(tree, foliage_path)
    add_asset(
        "commercial_foliage",
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CommercialFoliage",
            spawn=sim_utils.UsdFileCfg(usd_path=str(foliage_path)),
        ),
    )
    return tree
