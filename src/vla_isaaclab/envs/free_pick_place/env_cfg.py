"""Deterministic CPU physics scene with one freely resting apple and basket."""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ..common import (EventsCfg, JointLimitActionsCfg, PreviewObservationsCfg,
                      PreviewRewardsCfg, VLAEnvCfg, camera_cfg,
                      g1_left_wrist_camera_cfg, ground_cfg, light_cfgs,
                      make_g1_cfg, robot_rgb_camera_cfg)
from ..orchard_pick.contact import FINGER_LINKS
from . import mdp

APPLE_RADIUS = 0.030
APPLE_MASS = 0.160
APPLE_START = (-0.10, -0.29, 0.850)
SUPPORT_TOP = APPLE_START[2] - APPLE_RADIUS
BASKET = (-0.36, -0.30, 0.78)
CAMERA_EYE = (0.85, 2.1, 1.95)
CAMERA_TARGET = (-0.22, -0.30, 0.82)
CAMERA_PROFILE = "free_pick_place_three_view_v1"


def _box(path, size, position, color):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{path}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


@configclass
class FreePickPlaceSceneCfg(InteractiveSceneCfg):
    ground = ground_cfg()
    dome_light, key_light = light_cfgs()
    robot = make_g1_cfg((0.0, -0.72, 0.80))
    cam_left_high = camera_cfg(CAMERA_EYE, CAMERA_TARGET)
    cam_left_wrist = g1_left_wrist_camera_cfg()
    cam_right_wrist = robot_rgb_camera_cfg(
        "right_hand_palm_link", (-0.015, 0.040, 0.045),
        (0.9781476, 0.0, 0.2079117, 0.0), 12.0,
    )
    # A narrow physical fixture supports the free apple; no joint or stem exists.
    apple_support = _box("AppleSupport", (0.10, 0.10, 0.08),
                         (APPLE_START[0], APPLE_START[1], SUPPORT_TOP - 0.04),
                         (0.45, 0.35, 0.23))
    support_pedestal = _box("ApplePedestal", (0.06, 0.06, SUPPORT_TOP - 0.08),
                            (APPLE_START[0], APPLE_START[1], (SUPPORT_TOP - 0.08) / 2),
                            (0.36, 0.30, 0.25))
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple_00",
        spawn=sim_utils.SphereCfg(
            radius=APPLE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=APPLE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.4, dynamic_friction=1.1, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.035, 0.02)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=APPLE_START),
    )
    basket_floor = _box("BasketFloor", (0.32, 0.28, 0.02), BASKET, (0.06, 0.25, 0.65))
    basket_pedestal = _box("BasketPedestal", (0.09, 0.09, BASKET[2] - 0.01),
                           (BASKET[0], BASKET[1], (BASKET[2] - 0.01) / 2),
                           (0.3, 0.3, 0.3))
    basket_wall_0 = _box("BasketWall_0", (0.02, 0.28, 0.10),
                         (BASKET[0]-0.16, BASKET[1], BASKET[2]+0.06), (0.06, 0.25, 0.65))
    basket_wall_1 = _box("BasketWall_1", (0.02, 0.28, 0.10),
                         (BASKET[0]+0.16, BASKET[1], BASKET[2]+0.06), (0.06, 0.25, 0.65))
    basket_wall_2 = _box("BasketWall_2", (0.32, 0.02, 0.10),
                         (BASKET[0], BASKET[1]-0.14, BASKET[2]+0.06), (0.06, 0.25, 0.65))
    basket_wall_3 = _box("BasketWall_3", (0.32, 0.02, 0.10),
                         (BASKET[0], BASKET[1]+0.14, BASKET[2]+0.06), (0.06, 0.25, 0.65))
    apple_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Apple_00", update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/BasketFloor", "{ENV_REGEX_NS}/AppleSupport"]
        + [f"{{ENV_REGEX_NS}}/Robot/{name}" for name in FINGER_LINKS],
    )


@configclass
class FreePickPlaceEventsCfg(EventsCfg):
    reset_free_pick_place = EventTermCfg(func=mdp.reset_free_pick_place, mode="reset")


@configclass
class FreePickPlaceTerminationsCfg:
    success = TerminationTermCfg(func=mdp.success)
    failed = TerminationTermCfg(func=mdp.failed)
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)


@configclass
class FreePickPlaceEnvCfg(VLAEnvCfg):
    scene: FreePickPlaceSceneCfg = FreePickPlaceSceneCfg(num_envs=1, env_spacing=4.0)
    actions: JointLimitActionsCfg = JointLimitActionsCfg()
    observations: PreviewObservationsCfg = PreviewObservationsCfg()
    rewards: PreviewRewardsCfg = PreviewRewardsCfg()
    events: FreePickPlaceEventsCfg = FreePickPlaceEventsCfg()
    terminations: FreePickPlaceTerminationsCfg = FreePickPlaceTerminationsCfg()
    episode_length_s: float = 90.0
    task_instruction: str = "Pick up the freely resting apple and place it in the basket."
    camera_eye: tuple = CAMERA_EYE
    camera_target: tuple = CAMERA_TARGET
    camera_profile: str = CAMERA_PROFILE
    free_pick_place: bool = True
    apple_start: tuple = APPLE_START
    apple_radius: float = APPLE_RADIUS
    apple_mass: float = APPLE_MASS
    basket: tuple = BASKET
    grasp_offset: tuple = (-0.065, -0.085, 0.035)

    def __post_init__(self):
        super().__post_init__()
        self.sim.device = "cpu"
