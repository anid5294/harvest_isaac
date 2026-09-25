"""G1 orchard scene preview for visual and physics integration checks."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from ..common import (
    JointLimitActionsCfg,
    PreviewObservationsCfg,
    PreviewRewardsCfg,
    PreviewTerminationsCfg,
    VLAEnvCfg,
    camera_cfg,
    g1_head_camera_cfg,
    g1_left_wrist_camera_cfg,
    ground_cfg,
    light_cfgs,
    make_g1_cfg,
)
from .layout import Apple, Branch, make_orchard_layout


CAMERA_EYE = (2.45, -2.75, 2.05)
CAMERA_TARGET = (-0.03, 0.02, 1.05)
ORCHARD_PREVIEW_SEED = 42
_LAYOUT = make_orchard_layout(ORCHARD_PREVIEW_SEED)


def _material(color, roughness=0.72):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness, metallic=0.0)


def _static_box(name: str, size, position, color) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=_material(color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


def _branch(name: str, branch: Branch) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CylinderCfg(
            radius=branch.radius,
            height=branch.length,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=_material((0.28, 0.11, 0.035), roughness=0.88),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=branch.center,
            rot=branch.orientation_wxyz,
        ),
    )


def _apple(name: str, apple: Apple) -> AssetBaseCfg:
    # Preview apples are visual markers. The harvest task will replace apple_00
    # with a rigid fruit and an explicit breakable stem joint.
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.SphereCfg(
            radius=apple.radius,
            collision_props=None,
            visual_material=_material(apple.color, roughness=0.42),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=apple.position),
    )


def _foliage(name: str, position, radius: float) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.SphereCfg(
            radius=radius,
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.31, 0.055), roughness=0.95
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


_DOME_LIGHT, _KEY_LIGHT = light_cfgs()


@configclass
class OrchardPreviewSceneCfg(InteractiveSceneCfg):
    ground = ground_cfg()
    dome_light = _DOME_LIGHT
    key_light = _KEY_LIGHT
    robot = make_g1_cfg((0.0, -0.78, 0.80))
    cam_side = camera_cfg(CAMERA_EYE, CAMERA_TARGET)
    cam_left_high = g1_head_camera_cfg()
    cam_left_wrist = g1_left_wrist_camera_cfg()

    tree_trunk = _branch(
        "TreeTrunk",
        Branch(
            center=(0.0, 0.28, 0.85),
            length=1.70,
            radius=0.09,
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    branch_00 = _branch("Branch_00", _LAYOUT.branches[0])
    branch_01 = _branch("Branch_01", _LAYOUT.branches[1])
    branch_02 = _branch("Branch_02", _LAYOUT.branches[2])
    branch_03 = _branch("Branch_03", _LAYOUT.branches[3])
    branch_04 = _branch("Branch_04", _LAYOUT.branches[4])
    branch_05 = _branch("Branch_05", _LAYOUT.branches[5])

    foliage_00 = _foliage("Foliage_00", *_LAYOUT.foliage[0])
    foliage_01 = _foliage("Foliage_01", *_LAYOUT.foliage[1])
    foliage_02 = _foliage("Foliage_02", *_LAYOUT.foliage[2])
    foliage_03 = _foliage("Foliage_03", *_LAYOUT.foliage[3])
    foliage_04 = _foliage("Foliage_04", *_LAYOUT.foliage[4])

    apple_00 = _apple("Apple_00", _LAYOUT.apples[0])
    apple_01 = _apple("Apple_01", _LAYOUT.apples[1])
    apple_02 = _apple("Apple_02", _LAYOUT.apples[2])
    apple_03 = _apple("Apple_03", _LAYOUT.apples[3])
    apple_04 = _apple("Apple_04", _LAYOUT.apples[4])
    apple_05 = _apple("Apple_05", _LAYOUT.apples[5])

    # The tray is visibly supported, eliminating the floating-basket geometry
    # from the earlier MuJoCo prototype.
    tray_pedestal = _static_box(
        "TrayPedestal", (0.10, 0.10, 0.76), (-0.47, -0.15, 0.38), (0.22, 0.24, 0.27)
    )
    tray_floor = _static_box(
        "TrayFloor", (0.38, 0.28, 0.025), (-0.47, -0.15, 0.77), (0.05, 0.28, 0.68)
    )
    tray_wall_front = _static_box(
        "TrayWallFront", (0.38, 0.025, 0.13), (-0.47, -0.285, 0.83), (0.05, 0.28, 0.68)
    )
    tray_wall_back = _static_box(
        "TrayWallBack", (0.38, 0.025, 0.13), (-0.47, -0.015, 0.83), (0.05, 0.28, 0.68)
    )
    tray_wall_left = _static_box(
        "TrayWallLeft", (0.025, 0.28, 0.13), (-0.6475, -0.15, 0.83), (0.05, 0.28, 0.68)
    )
    tray_wall_right = _static_box(
        "TrayWallRight", (0.025, 0.28, 0.13), (-0.2925, -0.15, 0.83), (0.05, 0.28, 0.68)
    )


@configclass
class OrchardPreviewEnvCfg(VLAEnvCfg):
    scene: OrchardPreviewSceneCfg = OrchardPreviewSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=True,
    )
    actions: JointLimitActionsCfg = JointLimitActionsCfg()
    observations: PreviewObservationsCfg = PreviewObservationsCfg()
    rewards: PreviewRewardsCfg = PreviewRewardsCfg()
    terminations: PreviewTerminationsCfg = PreviewTerminationsCfg()
    task_instruction: str = (
        "Observe the G1, supported tray, apple tree, and reachable target apple."
    )
    camera_eye: tuple[float, float, float] = CAMERA_EYE
    camera_target: tuple[float, float, float] = CAMERA_TARGET
