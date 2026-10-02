"""CPU attached-apple trial: measured grasp, native stem break, physical placement."""

import math

import torch
from isaaclab.utils.math import quat_error_magnitude, quat_slerp

from vla_isaaclab.envs.common.g1 import LEFT_END_EFFECTOR
from .cpu_harvest_fsm import HarvestFSM, PHASES
from .ycb_sugar_box import YCBSugarBoxScriptedPolicy
from .ycb_sugar_box_strategy import EndEffectorTarget, SugarBoxPhaseStrategy


class CPUHarvestStrategy:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.apple = env.scene["object"]
        self.palm_id = self.robot.find_bodies([LEFT_END_EFFECTOR], preserve_order=True)[0][0]
        self.orientation = torch.tensor(
            SugarBoxPhaseStrategy.GRASP_QUAT_WXYZ, device=env.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        self.reset()

    def reset(self):
        if self.env.num_envs != 1 or str(self.env.device) != "cpu":
            raise ValueError("CPU harvest supports one CPU environment")
        self.fsm = HarvestFSM()
        self.phase = self.fsm.phase
        self.start = self.robot.data.body_pos_w[:, self.palm_id].clone()
        self.target = self.start.clone()
        self.start_quat = self.robot.data.body_quat_w[:, self.palm_id].clone()
        self.target_quat = self.start_quat.clone()
        self.grip_start = 0.0
        self.grip_target = 0.0
        self.palm_to_apple = None
        self.pull_origin = None
        self.pull_direction = None
        self.pull_distance_m = 0.0
        self._set_geometry()
        self.env.orchard_phase = self.phase

    @property
    def failed(self):
        return self.fsm.failed

    def _tensor(self, xyz):
        return self.apple.data.root_pos_w.new_tensor([xyz])

    def _set_geometry(self):
        grasp = self.apple.data.root_pos_w + self._tensor(self.env.cfg.grasp_offset)
        self.waypoints = {
            "pregrasp": grasp + self._tensor((0.0, -0.12, 0.08)),
            "approach": grasp,
            "close": grasp,
            "verify_grasp": grasp,
        }

    def _pose_ready(self, position_tolerance=0.035, angle_tolerance=math.radians(15)):
        palm = self.robot.data.body_pos_w[:, self.palm_id]
        quat = self.robot.data.body_quat_w[:, self.palm_id]
        return (float(torch.linalg.vector_norm(palm - self.target)) < position_tolerance
                and float(quat_error_magnitude(quat, self.target_quat)[0]) < angle_tolerance)

    def _basket_target(self, height_above_floor):
        basket = self.env.cfg.orchard_layout.basket
        center = self._tensor((basket[0], basket[1], basket[2] + height_above_floor))
        return center + self.palm_to_apple

    def _set_phase(self, phase):
        palm = self.robot.data.body_pos_w[:, self.palm_id].clone()
        self.phase = phase
        self.start = palm
        self.start_quat = self.robot.data.body_quat_w[:, self.palm_id].clone()
        self.target_quat = self.orientation.clone()
        self.grip_start = self.grip_target
        if phase in self.waypoints:
            self.target = self.waypoints[phase].clone()
            self.grip_target = 1.0 if phase in ("close", "verify_grasp") else 0.0
        elif phase == "pull":
            self.env.orchard_pull_started = True
            self.palm_to_apple = (palm - self.apple.data.root_pos_w).clone()
            self.pull_origin = palm.clone()
            layout = self.env.cfg.orchard_layout
            anchor = layout.stem_anchors[layout.target_index] if layout.stem_anchors else (
                layout.target[0], layout.target[1], layout.target[2] + 0.13)
            direction = self._tensor(tuple(a - b for a, b in zip(layout.target, anchor)))
            direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-6)
            self.pull_direction = direction
            self.pull_distance_m = 0.0
            self.target = palm.clone()
            self.grip_target = 1.0
        elif phase == "lift":
            # Stop the pull immediately at the first observed control-step break.
            self.target = palm + self._tensor((0.0, 0.0, 0.20))
            self.grip_target = 1.0
        elif phase == "transport":
            self.palm_to_apple = (palm - self.apple.data.root_pos_w).clone()
            self.target = self._basket_target(0.30)
            self.grip_target = 1.0
        elif phase == "lower":
            self.target = self._basket_target(0.17)
            self.grip_target = 1.0
        elif phase == "release":
            self.target = palm
            self.grip_target = 0.0
        elif phase == "retreat":
            self.target = palm + self._tensor((0.0, -0.20, 0.20))
            self.grip_target = 0.0
        else:
            self.target = palm
            self.target_quat = self.start_quat.clone()
        self.env.orchard_phase = phase
        self.env.policy_phase.fill_(PHASES.index(phase))

    def compute(self, step=0):
        metrics = dict(getattr(self.env, "orchard_metrics", {}))
        position = metrics.get("position", (float("inf"),) * 3)
        basket = self.env.cfg.orchard_layout.basket
        metrics["in_release_xy"] = (abs(position[0] - basket[0]) < 0.08
                                    and abs(position[1] - basket[1]) < 0.07)
        old = self.phase
        self.fsm.update(step=int(self.env.episode_length_buf[0]), metrics=metrics,
                        ready=self._pose_ready())
        if self.fsm.phase != old:
            self._set_phase(self.fsm.phase)
        if self.failed:
            self.env.orchard_failure = self.fsm.failure
        if self.phase == "pull" and not metrics.get("detached"):
            # Two millimeters per reached control step, bounded to 12 cm.
            # The native joint alone determines whether/when the fruit detaches.
            palm = self.robot.data.body_pos_w[:, self.palm_id]
            if float(torch.linalg.vector_norm(palm - self.target)) < 0.025:
                self.pull_distance_m = min(self.pull_distance_m + 0.002, 0.12)
                self.target = self.pull_origin + self.pull_distance_m * self.pull_direction
        blend = min(self.fsm.phase_steps / 45.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        if self.phase == "pull":
            blend = 1.0
        grip = self.grip_start + min(self.fsm.phase_steps / 45.0, 1.0) * (
            self.grip_target - self.grip_start)
        return EndEffectorTarget(
            position=self.start + blend * (self.target - self.start),
            orientation=torch.stack([
                quat_slerp(self.start_quat[i], self.target_quat[i].clone(), blend)
                for i in range(self.env.num_envs)
            ]),
            gripper_closed_fraction=torch.tensor([[grip]], device=self.env.device),
        )

    def diagnostics(self):
        return {
            "phase": self.phase, "failed": self.failed, "failure": self.fsm.failure,
            "transitions": self.fsm.transitions,
            "grasp_observed_control_step": self.fsm.grasp_step,
            "native_break_observed_control_step": self.fsm.break_step,
            "event_timing_resolution": "30 Hz control-step observation; internal PhysX break time unknown",
            "pull_distance_commanded_m": self.pull_distance_m,
            "physics_device": str(self.env.device),
            "metrics": getattr(self.env, "orchard_last_metrics", {}),
            "grasp_offset_world_m": list(self.env.cfg.grasp_offset),
            "grasp_parameters": "provisional; requires CPU physical rollout and measured contact review",
        }


class CPUHarvestScriptedPolicy(YCBSugarBoxScriptedPolicy):
    """Use the existing bounded DLS and exact normalized 43-D joint mapping."""

    def __init__(self, env):
        super().__init__(env, strategy=CPUHarvestStrategy(env))
