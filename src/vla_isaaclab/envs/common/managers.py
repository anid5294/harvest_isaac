"""Manager configurations shared by the concrete environments."""

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions import JointPositionToLimitsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .g1 import ACTION_JOINT_NAMES, LOWER_BODY_JOINT_NAMES


ACTION_TERM_NAME = "joint_positions"


def hold_default_joint_targets(env, env_ids, asset_cfg: SceneEntityCfg):
    """Initialize lower-body position targets before the first 43-D action."""
    robot = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    targets = robot.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    robot.set_joint_position_target(targets, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


@configclass
class EventsCfg:
    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    hold_lower_body = EventTerm(
        func=hold_default_joint_targets,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(LOWER_BODY_JOINT_NAMES), preserve_order=True
            )
        },
    )


@configclass
class JointLimitActionsCfg:
    """Normalized joint positions mapped directly by Isaac Lab to soft limits."""

    joint_positions = JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=list(ACTION_JOINT_NAMES),
        scale=1.0,
        rescale_to_limits=True,
    )


@configclass
class PreviewPolicyCfg(ObsGroup):
    joint_position = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTION_JOINT_NAMES))},
    )
    joint_velocity = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTION_JOINT_NAMES))},
    )
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PreviewObservationsCfg:
    policy: PreviewPolicyCfg = PreviewPolicyCfg()


@configclass
class PreviewRewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.0)


@configclass
class PreviewTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
