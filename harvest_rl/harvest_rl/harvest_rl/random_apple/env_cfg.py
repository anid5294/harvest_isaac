"""Configs for the randomized pick-and-place task (kept apart from the env class so the trainer
can read them before the simulator starts)."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from ..baseline.pick_place_env_cfg import PickPlaceEnvCfg
from ..orchard_env_cfg import OrchardSceneCfg

BASKET_PARTS = ("basket_floor", "basket_wall_0", "basket_wall_1", "basket_wall_2", "basket_wall_3", "pedestal")


@configclass
class RandomAppleEnvCfg(PickPlaceEnvCfg):
    """Apple (and optionally basket) at a random spot each episode. Defaults: evaluation."""

    apple_offset_xy: float = 0.08
    """Max apple shift in x and y [m] (uniform in +-this)."""
    apple_offset_z: float = 0.05
    """Max apple shift in z [m]."""
    offset_grid: int = 9
    """Candidate apple offsets per axis (x, y); z uses about half as many."""
    basket_offset_xy: float = 0.0
    """Max basket shift in x and y [m]."""
    arm_reset_noise: float = 0.0
    """Uniform noise [rad] on the controlled (arm + hand) joints' start pose."""
    w_release_shaping: float = 0.0
    """Reward for progress in opening the hand / moving the palm away once the apple is low
    inside the basket walls (0 disables)."""

    def __post_init__(self):
        super().__post_init__()
        # Stem and basket parts must move each episode, so make them (kinematic) rigid objects.
        for name in ("stem",) + BASKET_PARTS:
            part = getattr(self.scene, name)
            setattr(self.scene, name, RigidObjectCfg(
                prim_path=part.prim_path,
                spawn=part.spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=part.init_state.pos),
            ))


@configclass
class RandomizedTrainEnvCfg(RandomAppleEnvCfg):
    """Training: random apple and basket, release shaping, random arm start, random stem."""

    scene: OrchardSceneCfg = OrchardSceneCfg(num_envs=1024, env_spacing=5.0, replicate_physics=True)
    basket_offset_xy: float = 0.06
    arm_reset_noise: float = 0.1
    stem_break_force_range: tuple[float, float] | None = (12.0, 25.0)
    w_release_shaping: float = 10.0
