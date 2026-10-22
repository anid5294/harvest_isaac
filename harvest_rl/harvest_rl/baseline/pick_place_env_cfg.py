"""Config for the pick-and-place baseline. Kept apart from the env class so the trainer can
read it before the simulator starts (importing the env class first breaks Isaac Sim's startup)."""

from isaaclab.utils import configclass

from ..orchard_env_cfg import OrchardRLEnvCfg
from ..robot import G1_DEX3


@configclass
class PickPlaceEnvCfg(OrchardRLEnvCfg):
    robot = G1_DEX3.replace(
        controlled_joints=G1_DEX3.controlled_joints[15:22] + G1_DEX3.controlled_joints[29:36]
    )  # left arm (shoulder x3, elbow, wrist x3) + left hand (7); waist and the rest hold home
    # Small target nudges (max 0.05 rad per 1/30 s step): random exploration with large offsets
    # swats the apple off the stem before the policy can learn to grasp.
    action_mode: str = "delta"
    action_scale: float = 0.05
    episode_length_s: float = 15.0
    # Pulling (17.3 N) is the harvest mode here; flailing contacts that spin the apple past 90°
    # would otherwise snap the stem.
    stem_twist_break_angle: float | None = None

    # Reward v2. Progress terms pay for *changes* in a 0..1 potential (closer, pulling harder,
    # higher, nearer the basket), so hovering earns nothing; milestones pay once per episode;
    # every step costs w_time. |w_failure| > 450 steps * |w_time| so failing early never beats
    # running out the clock.
    w_reach: float = 5.0
    w_pull: float = 5.0
    w_lift: float = 5.0
    w_transport: float = 10.0
    w_first_grasp: float = 5.0
    w_detach: float = 10.0
    w_carried: float = 10.0
    w_place: float = 10.0
    w_release: float = 5.0
    w_success: float = 50.0
    w_time: float = -0.02
    w_failure: float = -20.0
    w_action_rate: float = -0.01
    w_joint_vel: float = -1e-4
