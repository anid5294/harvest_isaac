"""Robot plug-in point. Swap your humanoid in here; the rest of the environment adapts.

Everything the environment needs to know about the robot lives in :data:`ROBOT`:

* ``articulation``: how to spawn it (USD path, actuators/PD gains, home joint pose).
* ``controlled_joints``: which joints the policy commands. The action vector has one entry
  per joint, in exactly the order listed here.
* ``tracked_bodies``: links whose world pose/velocity is reported in the state (palms, etc.).
* ``contact_bodies``: links whose contact force against the apple is measured.
* ``grasp_*``: the hand that harvests. Success needs an opposing grip: some
  ``grasp_thumb_bodies`` link and some ``grasp_finger_bodies`` link both pressing on the apple.
  The hand counts as open when every ``grasp_hand_joints`` angle is below 0.30 rad.
* ``wrist_cameras``: cameras rigidly attached to robot links.
* ``stance_offset``/``stance_yaw_deg``: where the robot stands relative to the target apple.

The default is the teammate's robot from harvest_isaac: a Unitree G1 (29 DoF) with Dex3
three-finger hands and a fixed pelvis, using her 43-joint ``g1_29body_dex3_43d_v1`` action
order. To use your own robot:

1. Put the USD in ``harvest_rl/assets/`` (or point ``usd_path`` anywhere).
   URDF/MJCF can be used directly via ``sim_utils.UrdfFileCfg`` / ``sim_utils.MjcfFileCfg``
   instead of ``UsdFileCfg``.
2. Replace the actuator groups and ``joint_pos`` with your robot's joint names.
3. Update the joint/body name lists below.

Quaternions in Isaac Lab 3.0 are (x, y, z, w).
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from .tree import ASSET_DIR


@configclass
class WristCameraSpec:
    name: str = MISSING
    """Camera name; images come back under this key."""
    parent_body: str = MISSING
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Offset from the parent link [m], in the link frame."""
    rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Orientation in the link frame, "world" camera convention (forward +X, up +Z)."""
    focal_length: float = 12.0


@configclass
class RobotSpec:
    articulation: ArticulationCfg = MISSING
    controlled_joints: list[str] = MISSING
    """Joint names the policy controls, in action order."""
    tracked_bodies: list[str] = MISSING
    """Body names whose pose/velocity goes into the state dict."""
    contact_bodies: list[str] = []
    """Body names whose contact force with the apple is measured."""
    grasp_palm_body: str = MISSING
    grasp_thumb_bodies: list[str] = MISSING
    """Must be a subset of contact_bodies."""
    grasp_finger_bodies: list[str] = MISSING
    """Must be a subset of contact_bodies."""
    grasp_hand_joints: list[str] = MISSING
    wrist_cameras: list[WristCameraSpec] = []
    stance_offset: tuple[float, float, float] = (0.0, -0.5, 0.0)
    """Robot root position relative to the target apple, x/y [m]; z is the absolute root height [m]."""
    stance_yaw_deg: float = 90.0
    """Robot heading about world Z [deg]. 90 faces +Y (towards the tree for the default apple)."""

    def root_rot_xyzw(self) -> tuple[float, float, float, float]:
        half = math.radians(self.stance_yaw_deg) / 2.0
        return (0.0, 0.0, math.sin(half), math.cos(half))


def _pitch_xyzw(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


# ---------------------------------------------------------------------------------------
# Unitree G1 29-DoF + Dex3 hands, fixed pelvis (from harvest_isaac)
# ---------------------------------------------------------------------------------------

G1_USD = ASSET_DIR / "robot_g1_placeholder" / "g1_29dof_with_dex3_base_fix.usd"

# harvest_isaac's g1_29body_dex3_43d_v1 contract order (envs/common/g1.py ACTION_JOINT_NAMES).
_G1_LEGS = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]
_G1_WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
_G1_ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
_G1_LEFT_ARM = [f"left_{j}_joint" for j in _G1_ARM]
_G1_RIGHT_ARM = [f"right_{j}_joint" for j in _G1_ARM]
_G1_LEFT_HAND = [f"left_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]
# The contract swaps index and middle for the right hand (matches Unitree's DDS order).
_G1_RIGHT_HAND = [f"right_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
G1_CONTRACT_JOINTS = _G1_LEGS + _G1_WAIST + _G1_LEFT_ARM + _G1_RIGHT_ARM + _G1_LEFT_HAND + _G1_RIGHT_HAND

_G1_LEFT_HAND_LINKS = [
    "left_hand_thumb_0_link", "left_hand_thumb_1_link", "left_hand_thumb_2_link",
    "left_hand_middle_0_link", "left_hand_middle_1_link",
    "left_hand_index_0_link", "left_hand_index_1_link",
]
_G1_RIGHT_HAND_LINKS = [name.replace("left_", "right_", 1) for name in _G1_LEFT_HAND_LINKS]

# Home pose from harvest_isaac (left arm raised slightly toward the tree, legs slightly bent).
# Joints not listed start at 0. Patterns must not overlap (no catch-all ".*").
_G1_HOME_JOINT_POS = {
    "left_hip_pitch_joint": -0.05,
    "right_hip_pitch_joint": -0.05,
    "left_knee_joint": 0.2,
    "right_knee_joint": 0.2,
    "left_ankle_pitch_joint": -0.15,
    "right_ankle_pitch_joint": -0.15,
    "left_shoulder_pitch_joint": 0.255715,
    "left_shoulder_roll_joint": 0.431127,
    "left_shoulder_yaw_joint": -0.368285,
    "left_elbow_joint": -0.04442,
    "left_wrist_roll_joint": -0.333947,
}

G1_DEX3 = RobotSpec(
    articulation=ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(G1_USD),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=_G1_HOME_JOINT_POS, joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=0.9,
        # PD gains from harvest_isaac.
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"],
                joint_effort_limit=300.0,
                joint_velocity_limit=100.0,
                stiffness=1000.0,
                damping=100.0,
                armature=0.01,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*_joint"],
                joint_effort_limit=1000.0,
                stiffness=10000.0,
                damping=10000.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
                joint_effort_limit=300.0,
                joint_velocity_limit=100.0,
                stiffness={".*_shoulder_.*_joint": 300.0, ".*_elbow_joint": 400.0, ".*_wrist_.*_joint": 400.0},
                damping={".*_shoulder_.*_joint": 3.0, ".*_elbow_joint": 2.5, ".*_wrist_.*_joint": 2.5},
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_(thumb|middle|index)_.*_joint"],
                joint_effort_limit=300.0,
                joint_velocity_limit=100.0,
                stiffness=100.0,
                damping=10.0,
                armature=0.1,
            ),
        },
    ),
    controlled_joints=G1_CONTRACT_JOINTS,
    tracked_bodies=["left_hand_palm_link", "right_hand_palm_link"],
    contact_bodies=_G1_LEFT_HAND_LINKS + _G1_RIGHT_HAND_LINKS,
    # harvest_isaac harvests with the left hand.
    grasp_palm_body="left_hand_palm_link",
    grasp_thumb_bodies=_G1_LEFT_HAND_LINKS[:3],
    grasp_finger_bodies=_G1_LEFT_HAND_LINKS[3:],
    grasp_hand_joints=_G1_LEFT_HAND,
    # harvest_isaac's wrist cameras (envs/common/scene.py and orchard_pick/env_cfg.py).
    wrist_cameras=[
        WristCameraSpec(name="cam_left_wrist", parent_body="left_hand_palm_link",
                        pos=(-0.015, -0.040, 0.045), rot_xyzw=_pitch_xyzw(24.0), focal_length=12.0),
        WristCameraSpec(name="cam_right_wrist", parent_body="right_hand_palm_link",
                        pos=(-0.015, 0.040, 0.045), rot_xyzw=_pitch_xyzw(24.0), focal_length=12.0),
    ],
    # harvest_isaac's candidate stance: 0.43 m in front of the apple, apple slightly to the left.
    # Reachability was not validated there; tune it for your robot.
    stance_offset=(0.10, -0.43, 0.80),
    stance_yaw_deg=90.0,
)

ROBOT: RobotSpec = G1_DEX3
"""The robot the environment uses. Point this at your own :class:`RobotSpec`."""
