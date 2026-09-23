"""Concrete G1 scene-preview environment configurations."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
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
    table_cfgs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
YCB_ROOT = PROJECT_ROOT / "assets/YCB/Axis_Aligned_Physics"
DINNERWARE_ROOT = PROJECT_ROOT / "assets/YCB/dinnerware"
MICROWAVE_USD = PROJECT_ROOT / "assets/furniture_sim/generated/microwave.usd"
TABLETOP_CAMERA_EYE = (0.35, 1.90, 1.85)
TABLETOP_CAMERA_TARGET = (-0.15, -0.30, 0.72)
MICROWAVE_CAMERA_EYE = (2.6, -3.2, 2.2)
MICROWAVE_CAMERA_TARGET = (0.0, -0.30, 0.90)


def _ycb_cfg(asset: str, position, half_height: float, rotation=None) -> RigidObjectCfg:
    path = YCB_ROOT / f"{asset}.usd"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached YCB asset: {path}")
    return RigidObjectCfg(
        prim_path="",
        spawn=sim_utils.UsdFileCfg(usd_path=str(path)),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(position[0], position[1], position[2] + half_height),
            rot=rotation or (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
        ),
    )


def _dinnerware_cfg(path: Path, position) -> RigidObjectCfg:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing converted YCB dinnerware asset: {path}. Run scripts/prepare_ycb_dinnerware.sh first."
        )
    return RigidObjectCfg(
        prim_path="",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def _common_scene_assets(camera_eye, camera_target, robot_position):
    dome, key = light_cfgs()
    surface, leg_0, leg_1, leg_2, leg_3 = table_cfgs()
    return {
        "ground": ground_cfg(),
        "dome_light": dome,
        "key_light": key,
        "support_surface": surface,
        "support_leg_0": leg_0,
        "support_leg_1": leg_1,
        "support_leg_2": leg_2,
        "support_leg_3": leg_3,
        "robot": make_g1_cfg(robot_position),
        "camera": camera_cfg(camera_eye, camera_target),
    }


_TABLETOP = _common_scene_assets(TABLETOP_CAMERA_EYE, TABLETOP_CAMERA_TARGET, (0.0, -0.64, 0.80))


@configclass
class YCBSceneCfg(InteractiveSceneCfg):
    ground = _TABLETOP["ground"]
    dome_light = _TABLETOP["dome_light"]
    key_light = _TABLETOP["key_light"]
    support_surface = _TABLETOP["support_surface"]
    support_leg_0 = _TABLETOP["support_leg_0"]
    support_leg_1 = _TABLETOP["support_leg_1"]
    support_leg_2 = _TABLETOP["support_leg_2"]
    support_leg_3 = _TABLETOP["support_leg_3"]
    robot = _TABLETOP["robot"]
    camera = _TABLETOP["camera"]
    cam_left_high = g1_head_camera_cfg()
    cam_left_wrist = g1_left_wrist_camera_cfg()

    object = _ycb_cfg("003_cracker_box", (-0.18, -0.28, 0.70), 0.107)
    object.prim_path = "{ENV_REGEX_NS}/Object"
    goal = _ycb_cfg("005_tomato_soup_can", (0.12, -0.24, 0.70), 0.051)
    goal.prim_path = "{ENV_REGEX_NS}/Goal"
    distractor_sugar = _ycb_cfg("004_sugar_box", (-0.02, 0.18, 0.70), 0.088)
    distractor_sugar.prim_path = "{ENV_REGEX_NS}/DistractorSugar"
    distractor_mustard = _ycb_cfg(
        "006_mustard_bottle",
        (0.02, -0.12, 0.70),
        0.095,
        rotation=(math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0),
    )
    distractor_mustard.prim_path = "{ENV_REGEX_NS}/DistractorMustard"


@configclass
class DinnerwareSceneCfg(InteractiveSceneCfg):
    ground = _TABLETOP["ground"].copy()
    dome_light = _TABLETOP["dome_light"].copy()
    key_light = _TABLETOP["key_light"].copy()
    support_surface = _TABLETOP["support_surface"].copy()
    support_leg_0 = _TABLETOP["support_leg_0"].copy()
    support_leg_1 = _TABLETOP["support_leg_1"].copy()
    support_leg_2 = _TABLETOP["support_leg_2"].copy()
    support_leg_3 = _TABLETOP["support_leg_3"].copy()
    robot = _TABLETOP["robot"].copy()
    camera = _TABLETOP["camera"].copy()
    cam_left_high = g1_head_camera_cfg()
    cam_left_wrist = g1_left_wrist_camera_cfg()

    object = _dinnerware_cfg(DINNERWARE_ROOT / "024_bowl/024_bowl_physics.usd", (-0.18, -0.28, 0.70))
    object.prim_path = "{ENV_REGEX_NS}/Object"
    goal = _dinnerware_cfg(DINNERWARE_ROOT / "029_plate/029_plate_physics.usd", (0.12, -0.24, 0.70))
    goal.prim_path = "{ENV_REGEX_NS}/Goal"


_MICROWAVE = _common_scene_assets(
    MICROWAVE_CAMERA_EYE, MICROWAVE_CAMERA_TARGET, (0.0, -0.95, 0.74)
)


def _microwave_cfg() -> ArticulationCfg:
    if not MICROWAVE_USD.is_file():
        raise FileNotFoundError(
            f"Missing converted microwave asset: {MICROWAVE_USD}. Run scripts/prepare_microwave_asset.sh first."
        )
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Microwave",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(MICROWAVE_USD),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.62),
            joint_pos={"micro0joint": 0.0},
            joint_vel={"micro0joint": 0.0},
        ),
        actuators={
            "door": ImplicitActuatorCfg(
                joint_names_expr=["micro0joint"],
                effort_limit_sim=30.0,
                velocity_limit_sim=1.2,
                stiffness=30.0,
                damping=4.0,
            )
        },
    )


@configclass
class MicrowaveSceneCfg(InteractiveSceneCfg):
    ground = _MICROWAVE["ground"]
    dome_light = _MICROWAVE["dome_light"]
    key_light = _MICROWAVE["key_light"]
    support_surface = _MICROWAVE["support_surface"]
    support_leg_0 = _MICROWAVE["support_leg_0"]
    support_leg_1 = _MICROWAVE["support_leg_1"]
    support_leg_2 = _MICROWAVE["support_leg_2"]
    support_leg_3 = _MICROWAVE["support_leg_3"]
    robot = _MICROWAVE["robot"]
    camera = _MICROWAVE["camera"]
    cam_left_high = g1_head_camera_cfg()
    cam_left_wrist = g1_left_wrist_camera_cfg()
    microwave = _microwave_cfg()


@configclass
class ScenePreviewEnvCfg(VLAEnvCfg):
    actions: JointLimitActionsCfg = JointLimitActionsCfg()
    observations: PreviewObservationsCfg = PreviewObservationsCfg()
    rewards: PreviewRewardsCfg = PreviewRewardsCfg()
    terminations: PreviewTerminationsCfg = PreviewTerminationsCfg()
    task_instruction: str = "Observe the scene while the robot remains standing."


@configclass
class YCBScenePreviewEnvCfg(ScenePreviewEnvCfg):
    scene: YCBSceneCfg = YCBSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
    camera_eye: tuple[float, float, float] = TABLETOP_CAMERA_EYE
    camera_target: tuple[float, float, float] = TABLETOP_CAMERA_TARGET


@configclass
class DinnerwareScenePreviewEnvCfg(ScenePreviewEnvCfg):
    scene: DinnerwareSceneCfg = DinnerwareSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
    camera_eye: tuple[float, float, float] = TABLETOP_CAMERA_EYE
    camera_target: tuple[float, float, float] = TABLETOP_CAMERA_TARGET


@configclass
class MicrowaveScenePreviewEnvCfg(ScenePreviewEnvCfg):
    scene: MicrowaveSceneCfg = MicrowaveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
    camera_eye: tuple[float, float, float] = MICROWAVE_CAMERA_EYE
    camera_target: tuple[float, float, float] = MICROWAVE_CAMERA_TARGET
