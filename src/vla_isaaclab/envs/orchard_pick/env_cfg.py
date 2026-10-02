"""Physical fruit, breakable stems, supported basket, and three RGB views."""

from pathlib import Path

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
from .layout import make_layout, stem_geometry
from . import mdp
from .contact import FINGER_LINKS

# Fixed front/top view. Calibrate its framing on Songkhla before collection.
EYE = (0.9, 3.6, 2.7)
LOOKAT = (-0.10, -0.15, 0.95)
CAMERA_PROFILE = "orchard_fixed_front_top_three_view_v1"
COMMERCIAL_EYE = (1.5, 4.5, 3.3)
COMMERCIAL_LOOKAT = (0.1, 0.0, 1.4)
COMMERCIAL_CAMERA_PROFILE = "orchard_commercial_full_tree_three_view_v1"


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
    tree_model: str = "orchardbench"
    external_tree_asset: str | None = str(
        Path(__file__).resolve().parents[4] / "assets/orchard/orchardbench_seed42/isaac")
    commercial_tree_summary: dict | None = None
    grasp_offset: tuple = (-0.065, -0.085, 0.035)

    def __post_init__(self):
        super().__post_init__()
        self.configure_layout(self.orchard_seed)

    def configure_layout(self, seed):
        self.orchard_seed = seed
        if self.tree_model == "orchardbench":
            from .external_tree import load_asset
            if not self.external_tree_asset:
                raise ValueError("OrchardBench requires a prepared --orchardbench-asset")
            layout, external_usd, external_manifest = load_asset(self.external_tree_asset, seed)
        else:
            layout = make_layout(seed, tree_model=self.tree_model)
        self.orchard_layout = layout
        scene = self.scene
        scene.robot = make_g1_cfg((0.0, -0.72, 0.80))
        for name in getattr(self, "_orchard_scene_names", ()):
            if hasattr(scene, name):
                delattr(scene, name)
        self._orchard_scene_names = []

        def add_asset(name, asset):
            setattr(scene, name, asset)
            self._orchard_scene_names.append(name)

        if self.tree_model == "orchardbench":
            from isaaclab.assets import AssetBaseCfg
            add_asset("external_tree", AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/OrchardBenchTree",
                spawn=sim_utils.UsdFileCfg(usd_path=str(external_usd)),
            ))
            # Move the robot, never the morphology. This is only a candidate
            # stance: reachability and body/branch clearance need lab review.
            x, y, _ = layout.target
            scene.robot = make_g1_cfg((x+0.10, y-0.43, 0.80))
            self.commercial_tree_summary = external_manifest
            self.camera_eye = (3.4, 4.0, 3.2)
            self.camera_target = (0.0, -0.15, 1.2)
            self.camera_profile = "orchardbench_single_tree_three_view_v1"
        elif self.tree_model == "commercial":
            from .commercial_scene import add_commercial_tree

            tree = add_commercial_tree(scene, seed, add_asset)
            self.commercial_tree_summary = {
                "training_system": "tall_spindle_trellis",
                "seed": seed,
                "branch_count": len(tree.branches),
                "fruit_count": len(tree.fruits),
                "leaf_count": len(tree.leaves),
                "provenance": "original_procedural",
                "collision": "branch_and_trellis_cylinders; foliage_visual_only",
            }
            self.camera_eye = COMMERCIAL_EYE
            self.camera_target = COMMERCIAL_LOOKAT
            self.camera_profile = COMMERCIAL_CAMERA_PROFILE
        elif self.tree_model == "legacy":
            self.commercial_tree_summary = None
            add_asset("trunk", _branch("Trunk", cylinder_between(
                (layout.trunk[0], layout.trunk[1], 0.0),
                (layout.trunk[0], layout.trunk[1], 1.6), 0.055)))
            for i, position in enumerate(layout.apples):
                endpoint = stem_geometry(layout, i)["anchor"]
                add_asset(f"branch_{i}", _branch(f"Branch_{i}", cylinder_between(
                    (layout.trunk[0], layout.trunk[1], endpoint[2]), endpoint, 0.018)))
            add_asset("foliage", _foliage("Foliage", (0.12, 0.20, 1.55), 0.26))
            self.camera_eye = EYE
            self.camera_target = LOOKAT
            self.camera_profile = CAMERA_PROFILE
        else:
            raise ValueError(f"Unknown orchard tree model: {self.tree_model}")
        scene.cam_left_high = camera_cfg(self.camera_eye, self.camera_target)
        for i, position in enumerate(layout.apples):
            stem = stem_geometry(layout, i)
            add_asset(f"stem_{i}", _branch(f"Stem_{i}", cylinder_between(
                stem["bottom"], stem["anchor"], 0.003)))
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
            add_asset("object" if i == layout.target_index else f"apple_{i}", apple)
        scene.apple_contact.prim_path = f"{{ENV_REGEX_NS}}/Apple_{layout.target_index:02d}"
        x, y, z = layout.basket
        blue = (0.06, 0.25, 0.65)
        add_asset("basket_floor", _static_box("BasketFloor", (0.32, 0.28, 0.02), (x, y, z), blue))
        add_asset("pedestal", _static_box("Pedestal", (0.09, 0.09, z-0.01),
                                         (x, y, (z-0.01)/2), (0.3, 0.3, 0.3)))
        for i, (size, position) in enumerate((
            ((0.02, 0.28, 0.10), (x-0.16, y, z+0.06)),
            ((0.02, 0.28, 0.10), (x+0.16, y, z+0.06)),
            ((0.32, 0.02, 0.10), (x, y-0.14, z+0.06)),
            ((0.32, 0.02, 0.10), (x, y+0.14, z+0.06)),
        )):
            add_asset(f"wall_{i}", _static_box(f"BasketWall_{i}", size, position, blue))
