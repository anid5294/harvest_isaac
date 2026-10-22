"""Configuration for the orchard harvesting environment (Isaac Lab 3.0, PhysX).

Two presets:

* :class:`OrchardEnvCfg` (``Harvest-Orchard-v0``): a replica of harvest_isaac's
  ``VLA-OrchardPick-G1-JointPos-v0`` with the OrchardBench tree. Same scene, lights, robot,
  43-D action, 129-D observation, three cameras, success/failure rules and 90 s episodes.
* :class:`OrchardRLEnvCfg` (``Harvest-Orchard-RL-v0``): the same task for RL: no cameras,
  many parallel environments and a state observation that includes the apple.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from .robot import ROBOT, RobotSpec
from .tree import APPLE_USD, TREE_USD, load_fruit

FRUIT = load_fruit()

# Basket placement relative to the apple, from harvest_isaac (to the robot's left).
BASKET_OFFSET_XY = (-0.35, -0.40)
BASKET_FLOOR_HEIGHT = 0.78
BASKET_INNER_SIZE = (0.32, 0.28)
BASKET_WALL_HEIGHT = 0.10
BASKET_COLOR = (0.06, 0.25, 0.65)

# harvest_isaac's fixed external camera for the OrchardBench tree ("cam_left_high").
CAMERA_EYE = (3.4, 4.0, 3.2)
CAMERA_TARGET = (0.0, -0.15, 1.2)
CAMERA_PROFILE = "orchardbench_single_tree_three_view_v1"


def basket_center() -> tuple[float, float, float]:
    return (FRUIT.center[0] + BASKET_OFFSET_XY[0], FRUIT.center[1] + BASKET_OFFSET_XY[1], BASKET_FLOOR_HEIGHT)


def _material(color, roughness=0.72):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness, metallic=0.0)


def _kinematic_box(prim: str, size, pos, color) -> AssetBaseCfg:
    """harvest_isaac's ``_static_box``: an immovable (kinematic) collidable box."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=_material(color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _stem() -> AssetBaseCfg:
    """harvest_isaac's stem: a 3 mm kinematic collidable cylinder from the apple surface up to
    the branch (vertical for this apple)."""
    bottom_z, top_z = FRUIT.anchor[2] - FRUIT.stem_length, FRUIT.anchor[2]
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stem",
        spawn=sim_utils.CylinderCfg(
            radius=0.003, height=top_z - bottom_z, axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=_material((0.28, 0.11, 0.035), roughness=0.88),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(FRUIT.anchor[0], FRUIT.anchor[1], (bottom_z + top_z) / 2)),
    )


def _look_at_xyzw(eye, target):
    """OpenGL camera orientation looking from ``eye`` at ``target`` (Z up), as (x, y, z, w)."""
    def norm(v):
        n = math.sqrt(sum(c * c for c in v))
        return tuple(c / n for c in v)

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    back = norm(tuple(e - t for e, t in zip(eye, target)))
    right = norm(cross((0.0, 0.0, 1.0), back))
    up = cross(back, right)
    m = [[right[i], up[i], back[i]] for i in range(3)]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s)
    elif m[0][0] >= m[1][1] and m[0][0] >= m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = (0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    elif m[1][1] >= m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = ((m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s)
    return norm(q)


def external_camera_cfg() -> CameraCfg:
    """harvest_isaac's ``camera_cfg``: 640x480 RGB + depth at 30 Hz from a fixed viewpoint."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=1.0 / 30.0,
        width=640,
        height=480,
        data_types=["rgb", "distance_to_image_plane"],
        depth_clipping_behavior="max",
        offset=CameraCfg.OffsetCfg(pos=CAMERA_EYE, rot=_look_at_xyzw(CAMERA_EYE, CAMERA_TARGET), convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0, focus_distance=2.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)
        ),
    )


def wrist_camera_cfg(parent_body, pos, rot_xyzw, focal_length) -> CameraCfg:
    """harvest_isaac's ``robot_rgb_camera_cfg``: 640x480 RGB at 30 Hz on a robot link."""
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{parent_body}/Camera",
        update_period=1.0 / 30.0,
        width=640,
        height=480,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot_xyzw, convention="world"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length, focus_distance=0.7, horizontal_aperture=20.955, clipping_range=(0.03, 5.0)
        ),
    )


@configclass
class OrchardSceneCfg(InteractiveSceneCfg):
    """Ground, lights, the OrchardBench tree, one hanging apple, a basket and the robot.

    Positions are in each environment's local frame, which is the tree's own frame.
    """

    # harvest_isaac's ground and lights (envs/common/scene.py).
    ground = AssetBaseCfg(
        prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg(color=(0.25, 0.25, 0.25)), collision_group=-1
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=1100.0, color=(0.85, 0.88, 1.0))
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2200.0, color=(1.0, 0.95, 0.88), angle=0.8),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3827, 0.0, 0.0, 0.9239)),
    )

    # 168 static wood colliders, visual leaves, 39 visual-only apples (fruit 28 is left out).
    tree = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/OrchardBenchTree", spawn=sim_utils.UsdFileCfg(usd_path=str(TREE_USD)))

    # The one physical apple: YCB 013_apple at real size (convex-hull collider, friction
    # 1.4/1.1 authored in the USD), held on the branch by the stem model in OrchardEnv.
    apple = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple_00",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(APPLE_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=FRUIT.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=FRUIT.center),
    )
    stem = _stem()

    # Basket: floor + four walls on a pedestal, all kinematic (as in harvest_isaac).
    basket_floor = _kinematic_box("BasketFloor", (*BASKET_INNER_SIZE, 0.02), basket_center(), BASKET_COLOR)
    basket_wall_0 = _kinematic_box(
        "BasketWall_0", (0.02, BASKET_INNER_SIZE[1], BASKET_WALL_HEIGHT),
        (basket_center()[0] - BASKET_INNER_SIZE[0] / 2, basket_center()[1], BASKET_FLOOR_HEIGHT + 0.06), BASKET_COLOR)
    basket_wall_1 = _kinematic_box(
        "BasketWall_1", (0.02, BASKET_INNER_SIZE[1], BASKET_WALL_HEIGHT),
        (basket_center()[0] + BASKET_INNER_SIZE[0] / 2, basket_center()[1], BASKET_FLOOR_HEIGHT + 0.06), BASKET_COLOR)
    basket_wall_2 = _kinematic_box(
        "BasketWall_2", (BASKET_INNER_SIZE[0], 0.02, BASKET_WALL_HEIGHT),
        (basket_center()[0], basket_center()[1] - BASKET_INNER_SIZE[1] / 2, BASKET_FLOOR_HEIGHT + 0.06), BASKET_COLOR)
    basket_wall_3 = _kinematic_box(
        "BasketWall_3", (BASKET_INNER_SIZE[0], 0.02, BASKET_WALL_HEIGHT),
        (basket_center()[0], basket_center()[1] + BASKET_INNER_SIZE[1] / 2, BASKET_FLOOR_HEIGHT + 0.06), BASKET_COLOR)
    pedestal = _kinematic_box(
        "Pedestal", (0.09, 0.09, BASKET_FLOOR_HEIGHT - 0.01),
        (basket_center()[0], basket_center()[1], (BASKET_FLOOR_HEIGHT - 0.01) / 2), (0.3, 0.3, 0.3))

    # Filled in from the RobotSpec in OrchardEnvCfg.__post_init__ (plus cameras when enabled).
    robot: ArticulationCfg = MISSING
    apple_contact: ContactSensorCfg | None = None


@configclass
class OrchardEnvCfg(DirectRLEnvCfg):
    """Replica of harvest_isaac's OrchardBench pick task. Rewards are left at zero."""

    # -- robot (see harvest_rl/robot.py)
    robot: RobotSpec = ROBOT

    # -- timing: 120 Hz physics, 30 Hz control (harvest_isaac's VLAEnvCfg)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        physics=PhysxCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8, dynamic_friction=0.65, restitution=0.02, friction_combine_mode="max"
        ),
        default_visualizer_cfg=VisualizerCfg(eye=CAMERA_EYE, lookat=CAMERA_TARGET),
    )
    decimation: int = 4
    episode_length_s: float = 90.0
    num_rerenders_on_reset: int = 1
    wait_for_textures: bool = True

    # -- scene. harvest_isaac runs a single environment.
    scene: OrchardSceneCfg = OrchardSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=True)

    # -- metadata (harvest_isaac EnvCfg fields)
    task_instruction: str = "Pick an apple from the tree and place it in the basket."
    camera_profile: str = CAMERA_PROFILE

    # -- spaces: computed from the robot at startup
    action_space: int = 0
    observation_space: int = 0
    state_space: int = 0

    # -- sensors
    cameras: bool = True
    """Add harvest_isaac's three cameras (needs ``--enable_cameras`` or a window)."""

    # -- observation
    observation_mode: str = "contract"
    """``"contract"``: harvest_isaac's policy observation: joint position minus home pose, joint
    velocity and last action for the controlled joints (3 x 43 = 129 for the G1).
    ``"state"``: the contract terms plus palm-to-apple vectors, apple-to-basket vector, apple
    velocity and the stem-attached flag."""

    # -- actions
    action_mode: str = "absolute"
    """How actions map to joint position targets for ``robot.controlled_joints``:

    * ``"absolute"``: [-1, 1] maps onto each joint's soft limits (-1 = lower, 0 = middle,
      +1 = upper). harvest_isaac's JointPositionToLimitsActionCfg.
    * ``"offset"``: target = home pose + ``action_scale`` * action  [rad].
    * ``"delta"``: target = previous target + ``action_scale`` * action  [rad per step].
    """
    action_scale: float = 0.1
    clip_actions: float = 1.0

    # -- stem model (see OrchardEnv._apply_stem). Defaults come from tree.json.
    stem_break_force: float = FRUIT.stem_break_force
    """Elastic stem force at which the stem snaps [N]. tree.json: 17.34 N."""
    stem_break_force_range: tuple[float, float] | None = None
    """If set, sample a break force per episode uniformly in this range [N] (domain randomization)."""
    stem_axial_stiffness: float = 1000.0
    """Stiffness along the stem (pulling straight away from the branch) [N/m]."""
    stem_lateral_stiffness: float = 300.0
    """Stiffness sideways (bending the stem) [N/m]."""
    stem_damping_ratio: float = 0.7
    stem_twist_stiffness: float = 0.03
    """Torsional stiffness against twisting the apple [N*m/rad]. Kept small for stability with a 160 g apple."""
    stem_twist_break_angle: float | None = 1.57
    """Twisting the apple this far [rad] also snaps the stem (twist-off picking). None disables it."""

    # -- episode ends (harvest_isaac: success, failed, time_out terminations)
    terminate_on_success: bool = True
    terminate_on_failure: bool = True
    reset_joint_noise: float = 0.0
    """Uniform noise [rad] added to the robot's home joint positions on reset (0 in harvest_isaac)."""

    def __post_init__(self):
        spec = self.robot
        x, y, z = spec.stance_offset
        self.scene.robot = spec.articulation.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=spec.articulation.init_state.replace(
                pos=(FRUIT.center[0] + x, FRUIT.center[1] + y, z),
                rot=spec.root_rot_xyzw(),
            ),
        )
        # Contact forces between the apple and the basket floor, then each robot contact body.
        self.scene.apple_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Apple_00",
            update_period=0.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/BasketFloor"]
            + [f"{{ENV_REGEX_NS}}/Robot/{name}" for name in spec.contact_bodies],
        )
        if self.cameras:
            # Keeps harvest_isaac's feature key "cam_left_high" for the fixed external camera.
            self.scene.cam_left_high = external_camera_cfg()
            for cam in spec.wrist_cameras:
                setattr(self.scene, cam.name,
                        wrist_camera_cfg(cam.parent_body, cam.pos, cam.rot_xyzw, cam.focal_length))


@configclass
class OrchardRLEnvCfg(OrchardEnvCfg):
    """Same task, set up for RL: no cameras, many environments, state observation."""

    scene: OrchardSceneCfg = OrchardSceneCfg(num_envs=64, env_spacing=5.0, replicate_physics=True)
    cameras: bool = False
    observation_mode: str = "state"
    num_rerenders_on_reset: int = 0
