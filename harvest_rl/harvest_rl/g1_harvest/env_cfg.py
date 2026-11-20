"""Configs for the G1 harvesting benchmark (kept apart from the env class so trainers can read
them before the simulator starts).

Stage presets follow the training sequence of the instructions (section 6); every one is a
registered Gym ID (see ``harvest_rl/g1_harvest/__init__.py``). Nothing here trains anything.

=====  ======================================  =================================================
Stage  Config                                  What varies / what counts as success
=====  ======================================  =================================================
1      ``G1HarvestValidateEnvCfg``             seed-42 tree at its nominal pose, one apple,
                                               nominal physics. Used by the validation script.
2      ``G1HarvestReachGraspEnvCfg``           random apple sites; stems cannot break; success
                                               = sustained opposing grasp on the target.
3      ``G1HarvestPluckEnvCfg``                + random masses, sizes, stem thresholds; success
                                               = target detached under grip and carried.
4      ``G1HarvestPlaceEnvCfg``                full chain; success = target resting in the
                                               G1-mounted basket after release.
5      ``G1HarvestFullEnvCfg``                 + tree geometry library (train split), tree pose,
                                               1-3 apples; curriculum grows the pose variation.
6      ``G1HarvestHeldOutEnvCfg``              held-out trees, out-of-range masses/thresholds;
                                               evaluation only (no curriculum).
=====  ======================================  =================================================
"""

from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from ..orchard_env_cfg import FRUIT
from ..robot import G1_DEX3, G1_USD
from ..terrain import orchard_terrain_cfg
from .assets_gen import TREE_LIBRARY, BasketSpec, ensure_apple_sizes, ensure_robot_with_basket, ensure_tree_library
from .stem import StemParams

ROBOT_POS = (FRUIT.center[0] + 0.10, FRUIT.center[1] - 0.43, 0.80)
"""G1 pelvis position, env-local [m] (harvest_isaac's stance in front of the seed-42 apple)."""
ROBOT_YAW = math.pi / 2
"""G1 heading: facing +y (towards the tree)."""

LEFT_ARM = G1_DEX3.controlled_joints[15:22]
LEFT_HAND = G1_DEX3.controlled_joints[29:36]
LEFT_HAND_LINKS = G1_DEX3.grasp_thumb_bodies + G1_DEX3.grasp_finger_bodies
LEFT_FINGERTIPS = ["left_hand_thumb_2_link", "left_hand_index_1_link", "left_hand_middle_1_link"]
LEFT_ARM_LINKS = ["left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
                  "left_wrist_pitch_link", "left_wrist_yaw_link", "left_hand_palm_link"] + LEFT_HAND_LINKS


@configclass
class G1HarvestSceneCfg(InteractiveSceneCfg):
    """Rough orchard tile, one tree (kinematic, library prototype), K apples, G1 with basket."""

    terrain: TerrainImporterCfg = MISSING
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=1100.0, color=(0.85, 0.88, 1.0)))
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2200.0, color=(1.0, 0.95, 0.88), angle=0.8),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3827, 0.0, 0.0, 0.9239)),
    )
    tree: RigidObjectCfg = MISSING
    robot: ArticulationCfg = MISSING
    # Apple_0..K-1 and Stem_0..K-1 are added in G1HarvestEnvCfg.__post_init__.
    apple_contact: ContactSensorCfg = MISSING
    arm_contact: ContactSensorCfg = MISSING


@configclass
class G1HarvestEnvCfg(DirectRLEnvCfg):
    """Full benchmark config; the stage presets below override parts of it."""

    # -- timing (as harvest_rl): 120 Hz physics, 30 Hz control
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        physics=PhysxCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8, dynamic_friction=0.65, restitution=0.02, friction_combine_mode="max"
        ),
        default_visualizer_cfg=VisualizerCfg(eye=(3.0, -4.2, 2.4), lookat=(0.5, -1.0, 0.9)),
    )
    decimation: int = 4
    episode_length_s: float = 20.0
    scene: G1HarvestSceneCfg = G1HarvestSceneCfg(num_envs=64, env_spacing=5.0, replicate_physics=True)
    action_space: int = 0
    observation_space: int = 0
    state_space: int = 0

    # -- tree library (harvest_rl/g1_harvest/assets_gen.py)
    tree_pool: tuple[str, ...] = tuple(s.name for s in TREE_LIBRARY if s.split == "train")
    """Tree variants spawned (env i gets one of them). Stage presets grow this list."""
    tree_pose_variation: float = 1.0
    """Initial curriculum level c in [0, 1]: scales the yaw range and the trunk translation range."""
    tree_yaw_max: float = math.pi
    """Max |tree yaw| at c = 1 [rad]."""
    tree_shift_range: tuple[float, float] = (0.35, 0.80)
    """Max trunk shift from the tile centre at c = 0 and c = 1 [m] (the solver needs some room even
    at c = 0 to put a site of a non-nominal tree into reach)."""
    pose_candidates: int = 64
    """Candidate tree poses tried per reset; the first collision-free one wins."""
    target_reach_box: tuple[tuple[float, float], ...] = ((0.28, 0.52), (-0.08, 0.28), (0.10, 0.42))
    """Where the target apple's centre may be, in the G1 pelvis frame (forward, left, up) [m]."""
    robot_clearance_box: tuple[float, float] = (0.09, 0.20)
    """Wood must stay outside this torso box (half depth forward/back, half width sideways, pelvis
    frame) below ``robot_clearance_height`` [m]."""
    robot_clearance_height: float = 1.45
    basket_clearance_radius: float = 0.20

    # -- apples
    num_apples: int = 3
    """Physical apples per env (K). Unused ones are parked under the ground on their stems."""
    num_apples_range: tuple[int, int] = (1, 3)
    apple_scales: tuple[float, ...] = (0.85, 1.0, 1.15)
    """Apple size prototypes (scale of the 7.5 cm YCB apple); fixed per env at spawn."""
    apple_density_range: tuple[float, float] = (700.0, 950.0)
    """[kg/m^3]; mass = density x volume(size), clipped to ``apple_mass_range``."""
    apple_mass_range: tuple[float, float] = (0.10, 0.30)
    """Engineering range from the instructions [kg]."""

    # -- stems (harvest_rl/g1_harvest/stem.py); thresholds sampled per apple per episode
    stem: StemParams = StemParams()
    stem_tensile_range: tuple[float, float] = (12.0, 25.0)
    """Straight-pull detachment force [N] (harvest_rl's 12-25 N; tree.json: 17.3 N)."""
    stem_bending_range: tuple[float, float] = (0.25, 0.60)
    """Bending-moment limit [N m]; with the ~6 cm lever this is ~4-10 N sideways."""
    stem_twist_range: tuple[float, float] = (0.03, 0.06)
    """Torsion limit [N m]; ~1-2 rad of twist with the default twist stiffness."""
    stems_unbreakable: bool = False

    # -- robot and basket
    basket: BasketSpec = BasketSpec()
    self_collisions: bool = True
    """Self-collisions on, so the hand/arm cannot pass through the mounted basket."""
    controlled_joints: list[str] = LEFT_ARM + LEFT_HAND
    """Arm and hand joints first (base and waist held, as the instructions ask)."""
    action_scale: float = 0.05
    """Max change of a joint-position target per control step [rad] (bounded relative targets)."""
    arm_reset_noise: float = 0.1
    """Uniform noise on the controlled joints' start pose [rad]."""

    # -- task outcome
    success_mode: str = "place"
    """``"grasp"`` (stage 2), ``"carry"`` (stage 3) or ``"place"`` (stage 4+)."""
    grasp_success_steps: int = 15
    terminate_on_success: bool = True
    terminate_on_failure: bool = True

    # -- safety / damage proxies
    finger_force_safe: float = 10.0
    """Contact force per finger link above which the apple is assumed bruised [N]."""
    impact_speed_safe: float = 0.6
    """Apple speed when it first touches the basket above which it counts as a damaging drop [m/s]."""
    wood_contact_threshold: float = 1.0
    """Arm/hand force on the wood counted as an unsafe collision [N]."""
    basket_contact_threshold: float = 5.0
    """Arm/hand force on the basket counted as a self-collision [N]."""

    # -- curriculum (stage 5)
    curriculum: bool = False
    curriculum_window: int = 256
    curriculum_up: float = 0.6
    curriculum_down: float = 0.2
    curriculum_step: float = 0.1

    # -- rewards (potential-based progress + one-time milestones + penalties)
    w_reach: float = 5.0
    w_grasp: float = 5.0
    w_hold_grasp: float = 0.05
    w_detach: float = 10.0
    w_lift: float = 5.0
    w_transport: float = 10.0
    w_place: float = 10.0
    w_release: float = 5.0
    w_success: float = 50.0
    w_time: float = -0.01
    w_failure: float = -20.0
    w_finger_force: float = -0.05
    w_impact: float = -2.0
    w_wood_collision: float = -0.1
    w_basket_collision: float = -0.1
    w_knock_off: float = -5.0
    w_action_rate: float = -0.01
    w_joint_vel: float = -1e-4

    def __post_init__(self):
        lib = ensure_tree_library()
        by_name = {v["name"]: v for v in lib["variants"]}
        unknown = [n for n in self.tree_pool if n not in by_name]
        if unknown:
            raise ValueError(f"Unknown tree variants {unknown}; library has {list(by_name)}")
        s = self.scene
        # Flat pads: the trunk may move up to tree_shift_range[1] from the tile centre.
        s.terrain = orchard_terrain_cfg(
            [(0.0, 0.0, self.tree_shift_range[1] + 0.35), (ROBOT_POS[0], ROBOT_POS[1], 0.45)],
            num_envs=s.num_envs, env_spacing=s.env_spacing,
        )
        s.tree = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tree",
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=[by_name[n]["usd"] for n in self.tree_pool],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        robot_usd = ensure_robot_with_basket(G1_USD, self.basket)
        art = G1_DEX3.articulation
        s.robot = art.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=art.spawn.replace(
                usd_path=robot_usd,
                articulation_props=art.spawn.articulation_props.replace(enabled_self_collisions=self.self_collisions),
            ),
            init_state=art.init_state.replace(
                pos=ROBOT_POS, rot=(0.0, 0.0, math.sin(ROBOT_YAW / 2), math.cos(ROBOT_YAW / 2))
            ),
        )
        apple_usds = ensure_apple_sizes(self.apple_scales)
        for k in range(self.num_apples):
            setattr(s, f"apple_{k}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Apple_{k}",
                spawn=sim_utils.MultiUsdFileCfg(
                    usd_path=apple_usds,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=FRUIT.mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    activate_contact_sensors=True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -3.0 - 0.3 * k)),
            ))
            # Visual stem (kinematic, no collision), moved with its apple's site at reset.
            setattr(s, f"stem_{k}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Stem_{k}",
                spawn=sim_utils.CylinderCfg(
                    radius=0.003, height=0.025, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.11, 0.035), roughness=0.88),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -3.0 - 0.3 * k + 0.05)),
            ))
        # Apple contacts: basket, then every left/right hand link (robot.py's contact bodies).
        s.apple_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Apple_.*",
            update_period=0.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/basket_link"]
            + [f"{{ENV_REGEX_NS}}/Robot/{b}" for b in G1_DEX3.contact_bodies],
        )
        # Arm/hand contacts with the wood and with the mounted basket (unsafe collisions).
        s.arm_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(LEFT_ARM_LINKS) + ")",
            update_period=0.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Tree", "{ENV_REGEX_NS}/Robot/basket_link"],
        )


# ---------------------------------------------------------------------------------------
# Stage presets (section 6 of the instructions)
# ---------------------------------------------------------------------------------------


@configclass
class G1HarvestValidateEnvCfg(G1HarvestEnvCfg):
    """Stage 1: mounted basket + joint control in the existing single-tree scene, nothing random."""

    scene: G1HarvestSceneCfg = G1HarvestSceneCfg(num_envs=4, env_spacing=5.0, replicate_physics=True)
    tree_pool: tuple[str, ...] = ("seed42",)
    tree_pose_variation: float = 0.0
    tree_yaw_max: float = 0.0
    tree_shift_range: tuple[float, float] = (0.0, 0.0)
    num_apples: int = 1
    num_apples_range: tuple[int, int] = (1, 1)
    apple_scales: tuple[float, ...] = (1.0,)
    apple_density_range: tuple[float, float] = (755.0, 755.0)  # 160 g, as tree.json
    stem_tensile_range: tuple[float, float] = (17.34, 17.34)
    stem_bending_range: tuple[float, float] = (0.4, 0.4)
    stem_twist_range: tuple[float, float] = (0.045, 0.045)
    arm_reset_noise: float = 0.0
    terminate_on_success: bool = False
    terminate_on_failure: bool = False


@configclass
class G1HarvestReachGraspEnvCfg(G1HarvestEnvCfg):
    """Stage 2: reach and grasp randomly placed apples (seed-42 tree, stems never break)."""

    scene: G1HarvestSceneCfg = G1HarvestSceneCfg(num_envs=1024, env_spacing=5.0, replicate_physics=True)
    tree_pool: tuple[str, ...] = ("seed42",)
    tree_pose_variation: float = 0.0
    tree_yaw_max: float = 0.35
    num_apples: int = 1
    num_apples_range: tuple[int, int] = (1, 1)
    apple_scales: tuple[float, ...] = (1.0,)
    apple_density_range: tuple[float, float] = (755.0, 755.0)
    stems_unbreakable: bool = True
    success_mode: str = "grasp"
    episode_length_s: float = 10.0


@configclass
class G1HarvestPluckEnvCfg(G1HarvestReachGraspEnvCfg):
    """Stage 3: physically modelled stems, varying masses/sizes/thresholds; controlled plucking."""

    apple_scales: tuple[float, ...] = (0.85, 1.0, 1.15)
    apple_density_range: tuple[float, float] = (700.0, 950.0)
    stems_unbreakable: bool = False
    success_mode: str = "carry"
    episode_length_s: float = 15.0


@configclass
class G1HarvestPlaceEnvCfg(G1HarvestPluckEnvCfg):
    """Stage 4: pluck and place the apple into the G1-mounted basket."""

    success_mode: str = "place"
    episode_length_s: float = 20.0


@configclass
class G1HarvestFullEnvCfg(G1HarvestPlaceEnvCfg):
    """Stage 5: tree geometry (train split) + tree pose + 1-3 apples, curriculum on pose variation."""

    tree_pool: tuple[str, ...] = tuple(s.name for s in TREE_LIBRARY if s.split == "train")
    tree_pose_variation: float = 0.0
    tree_yaw_max: float = math.pi
    num_apples: int = 3
    num_apples_range: tuple[int, int] = (1, 3)
    curriculum: bool = True


@configclass
class G1HarvestHeldOutEnvCfg(G1HarvestFullEnvCfg):
    """Stage 6: evaluation on held-out trees and unseen apple/stem physics (no curriculum)."""

    scene: G1HarvestSceneCfg = G1HarvestSceneCfg(num_envs=256, env_spacing=5.0, replicate_physics=True)
    tree_pool: tuple[str, ...] = tuple(s.name for s in TREE_LIBRARY if s.split == "heldout")
    tree_pose_variation: float = 1.0
    curriculum: bool = False
    apple_density_range: tuple[float, float] = (650.0, 1000.0)
    apple_mass_range: tuple[float, float] = (0.08, 0.34)
    stem_tensile_range: tuple[float, float] = (9.0, 30.0)
    stem_bending_range: tuple[float, float] = (0.18, 0.75)
    stem_twist_range: tuple[float, float] = (0.02, 0.08)
    arm_reset_noise: float = 0.0
