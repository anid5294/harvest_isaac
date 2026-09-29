"""Physical fruit, breakable stems, supported basket, and three RGB views."""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ..common import (
    EventsCfg, JointLimitActionsCfg, PreviewObservationsCfg, PreviewRewardsCfg,
    VLAEnvCfg, camera_cfg, g1_left_wrist_camera_cfg,
    ground_cfg, light_cfgs, make_g1_cfg, robot_rgb_camera_cfg,
)
from ..orchard_preview.env_cfg import _branch, _static_box, _foliage
from ..orchard_preview.layout import cylinder_between
from .layout import make_layout
from . import mdp
from .contact import FINGER_LINKS

# Fixed front/top view. Calibrate its framing on Songkhla before collection.
EYE = (0.9, 3.6, 2.7)
LOOKAT = (-0.10, -0.15, 0.95)
CAMERA_PROFILE = "orchard_fixed_front_top_three_view_v1"


@configclass
class OrchardSceneCfg(InteractiveSceneCfg):
    ground = ground_cfg()
    dome_light, key_light = light_cfgs()
    robot = make_g1_cfg((0.0, -0.72, 0.80))
    # Keep the contract's required feature key, but identify this sensor as a
    # world-fixed external camera in the exported calibration metadata.
    cam_left_high = camera_cfg(EYE, LOOKAT)
    cam_left_wrist = g1_left_wrist_camera_cfg()
    cam_right_wrist = robot_rgb_camera_cfg(
        "right_hand_palm_link", (-0.015, 0.040, 0.045),
        (0.9781476, 0.0, 0.2079117, 0.0), 12.0,
    )
    apple_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Apple_00", update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/BasketFloor"] + [
            f"{{ENV_REGEX_NS}}/Robot/{name}" for name in FINGER_LINKS
        ],
    )


@configclass
class OrchardEventsCfg(EventsCfg):
    reset_orchard = EventTermCfg(func=mdp.reset_orchard, mode="reset")


@configclass
class OrchardTerminationsCfg:
    success = TerminationTermCfg(func=mdp.success)
    failed = TerminationTermCfg(func=mdp.failed)
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)


@configclass
class OrchardPickEnvCfg(VLAEnvCfg):
    scene: OrchardSceneCfg = OrchardSceneCfg(num_envs=1, env_spacing=4.0)
    actions: JointLimitActionsCfg = JointLimitActionsCfg()
    observations: PreviewObservationsCfg = PreviewObservationsCfg()
    rewards: PreviewRewardsCfg = PreviewRewardsCfg()
    events: OrchardEventsCfg = OrchardEventsCfg()
    terminations: OrchardTerminationsCfg = OrchardTerminationsCfg()
    episode_length_s: float = 90.0
    task_instruction: str = "Pick an apple from the tree and place it in the basket."
    camera_eye: tuple = EYE
    camera_target: tuple = LOOKAT
    camera_profile: str = CAMERA_PROFILE
    orchard_seed: int = 42
    grasp_offset: tuple = (-0.065, -0.085, 0.035)

    def __post_init__(self):
        super().__post_init__()
        self.configure_layout(self.orchard_seed)

    def configure_layout(self, seed):
        self.orchard_seed = seed
        layout = make_layout(seed)
        self.orchard_layout = layout
        scene = self.scene
        scene.trunk = _branch("Trunk", cylinder_between(
            (layout.trunk[0], layout.trunk[1], 0.0),
            (layout.trunk[0], layout.trunk[1], 1.6), 0.055))
        for i, position in enumerate(layout.apples):
            endpoint = (position[0], position[1], position[2] + 0.13)
            setattr(scene, f"branch_{i}", _branch(f"Branch_{i}", cylinder_between(
                (layout.trunk[0], layout.trunk[1], endpoint[2]), endpoint, 0.018)))
            setattr(scene, f"stem_{i}", _branch(f"Stem_{i}", cylinder_between(
                (position[0], position[1], position[2] + layout.radius), endpoint, 0.003)))
            apple = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Apple_{i:02d}",
                spawn=sim_utils.SphereCfg(
                    radius=layout.radius,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=layout.mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    activate_contact_sensors=True,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.4, dynamic_friction=1.1, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.035, 0.02)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
            )
            setattr(scene, "object" if i == layout.target_index else f"apple_{i}", apple)
        scene.apple_contact.prim_path = f"{{ENV_REGEX_NS}}/Apple_{layout.target_index:02d}"
        scene.foliage = _foliage("Foliage", (0.12, 0.20, 1.55), 0.26)
        x, y, z = layout.basket
        blue = (0.06, 0.25, 0.65)
        scene.basket_floor = _static_box("BasketFloor", (0.32, 0.28, 0.02), (x, y, z), blue)
        scene.pedestal = _static_box("Pedestal", (0.09, 0.09, z-0.01),
                                     (x, y, (z-0.01)/2), (0.3, 0.3, 0.3))
        for i, (size, position) in enumerate((
            ((0.02, 0.28, 0.10), (x-0.16, y, z+0.06)),
            ((0.02, 0.28, 0.10), (x+0.16, y, z+0.06)),
            ((0.32, 0.02, 0.10), (x, y-0.14, z+0.06)),
            ((0.32, 0.02, 0.10), (x, y+0.14, z+0.06)),
        )):
            setattr(scene, f"wall_{i}", _static_box(f"BasketWall_{i}", size, position, blue))
