"""Config for the grip-pluck-hold task (no basket placing). Kept apart from the env class so
the trainer can read it before the simulator starts."""

from isaaclab.utils import configclass

from ..orchard_env_cfg import OrchardSceneCfg
from ..random_apple.env_cfg import RandomAppleEnvCfg


@configclass
class PluckEnvCfg(RandomAppleEnvCfg):
    """Reach, grip with all three fingers, pluck, hold, bring toward the chest.

    Random apple position (+-8 cm xy, +-5 cm z), random arm start (+-0.1 rad), random stem
    strength (12-25 N). Episodes end on harvest_isaac's failures (dropped / knocked off without
    a grip) or the 15 s timeout.
    """

    scene: OrchardSceneCfg = OrchardSceneCfg(num_envs=1024, env_spacing=5.0, replicate_physics=True)
    arm_reset_noise: float = 0.1
    stem_break_force_range: tuple[float, float] | None = (12.0, 25.0)
    basket_offset_xy: float = 0.0
    w_release_shaping: float = 0.0

    touch_threshold: float = 0.05
    """Contact force [N] for a finger to count as touching the apple."""
    w_reach: float = 10.0
    """x (previous - current fingertip-to-apple distance [m]), every step until the first grasp."""
    w_grasp: float = 5.0
    """Once: first step thumb, index and middle all touch the apple while it's on the stem."""
    w_pluck: float = 10.0
    """Once: stem snaps while all three fingers touch (within the last 3 steps)."""
    w_hold: float = 0.1
    """Every step after the pluck while all three fingers touch."""
    w_chest: float = 10.0
    """x (previous - current apple-to-chest distance [m]), every step while all three fingers touch."""
