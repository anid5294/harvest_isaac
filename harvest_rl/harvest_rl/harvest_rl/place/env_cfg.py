"""Config for the pluck-and-place task. Kept apart from the env class so the trainer can read it
before the simulator starts."""

from isaaclab.utils import configclass

from ..pluck.env_cfg import PluckEnvCfg


@configclass
class PlaceEnvCfg(PluckEnvCfg):
    """Pluck the apple, carry it to a randomly placed basket, release it there.

    Same randomization as the pluck task (apple +-8 cm xy / +-5 cm z, arm start +-0.1 rad,
    stem 12-25 N) plus a random basket position (+-6 cm xy). Episodes end on harvest_isaac's
    success or failures (dropped / knocked off without a grip) or the time limit.
    """

    basket_offset_xy: float = 0.06
    episode_length_s: float = 20.0
    """Longer than the pluck task (15 s) to leave time to carry and release."""

    # Kept from the pluck task: w_reach (x10, until grasp), w_grasp (+5), w_pluck (+10).
    w_hold: float = 0.0
    """Removed: a per-step reward for holding teaches the policy never to let go."""
    w_chest: float = 0.0
    """Removed."""
    w_basket: float = 10.0
    """x (previous - current apple-to-basket-resting-spot distance [m]), every step while all
    three fingers touch."""
    w_release_shaping: float = 10.0
    """x progress in a 0..1 potential once the plucked apple is low inside the basket walls:
    half for opening the hand, half for moving the palm away (once the hand is open)."""
    w_placed: float = 10.0
    """Once: the plucked apple rests settled in the basket with the floor carrying it."""
    w_success: float = 50.0
    """harvest_isaac's success (carried, placed, released, held 30 steps); the episode then ends."""
