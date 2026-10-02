"""Measured finite-state free-apple controller using the existing bounded G1 IK."""

import math

import torch
from isaaclab.utils.math import quat_error_magnitude, quat_slerp

from vla_isaaclab.envs.common.g1 import LEFT_END_EFFECTOR
from .ycb_sugar_box import YCBSugarBoxScriptedPolicy
from .ycb_sugar_box_strategy import EndEffectorTarget, SugarBoxPhaseStrategy


PHASES = (
    "settle", "pregrasp", "approach", "close", "verify_grasp", "lift",
    "transport", "lower", "release", "retreat", "verify_placement", "done", "failed",
)
TIMEOUT_STEPS = {
    "settle": 45, "pregrasp": 360, "approach": 300, "close": 150,
    "verify_grasp": 90, "lift": 300, "transport": 420, "lower": 360,
    "release": 150, "retreat": 300, "verify_placement": 300,
}


class FreePickPlaceStrategy:
    """Left-hand waypoints; all manipulation transitions inspect measured state."""

    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.apple = env.scene["object"]
        self.palm_id = self.robot.find_bodies([LEFT_END_EFFECTOR], preserve_order=True)[0][0]
        self.orientation = torch.tensor(
            SugarBoxPhaseStrategy.GRASP_QUAT_WXYZ,
            device=env.device, dtype=torch.float32,
        ).unsqueeze(0)
        self.reset()

    def reset(self):
        if self.env.num_envs != 1:
            raise ValueError("Scripted free pick-and-place supports one environment")
        self.phase = "settle"
        self.phase_steps = 0
        self.transitions = []
        self.failure = None
        self.contact_loss_steps = 0
        self.start = self.robot.data.body_pos_w[:, self.palm_id].clone()
        self.target = self.start.clone()
        self.start_quat = self.robot.data.body_quat_w[:, self.palm_id].clone()
        self.target_quat = self.start_quat.clone()
        self.grip_start = 0.0
        self.grip_target = 0.0
        self.palm_to_apple = None
        self._set_geometry()
        self.env.free_pick_place_phase = self.phase
        self.env.policy_phase = torch.zeros(1, dtype=torch.long, device=self.env.device)

    @property
    def failed(self):
        return self.failure is not None

    def _tensor(self, xyz):
        return self.apple.data.root_pos_w.new_tensor([xyz])

    def _set_geometry(self):
        apple = self.apple.data.root_pos_w.clone()
        grasp = apple + self._tensor(self.env.cfg.grasp_offset)
        self.waypoints = {
            "pregrasp": grasp + self._tensor((0.0, -0.12, 0.08)),
            "approach": grasp,
            "close": grasp,
            "verify_grasp": grasp,
            "lift": grasp + self._tensor((0.0, 0.0, 0.20)),
        }

    def _set_phase(self, phase, position, grip, orientation=None):
        self.transitions.append({"from": self.phase, "to": phase,
                                 "step": int(self.env.episode_length_buf[0])})
        self.phase = phase
        self.phase_steps = 0
        self.start = self.robot.data.body_pos_w[:, self.palm_id].clone()
        self.target = position.clone()
        self.start_quat = self.robot.data.body_quat_w[:, self.palm_id].clone()
        self.target_quat = self.orientation if orientation is None else orientation.clone()
        self.grip_start = self.grip_target
        self.grip_target = grip
        self.env.free_pick_place_phase = phase
        self.env.policy_phase.fill_(PHASES.index(phase))

    def _fail(self, reason):
        self.failure = reason
        self.env.free_pick_place_failure = reason
        palm = self.robot.data.body_pos_w[:, self.palm_id]
        self._set_phase("failed", palm, self.grip_target,
                        self.robot.data.body_quat_w[:, self.palm_id])

    def _pose_ready(self, position_tolerance=0.035, angle_tolerance=math.radians(15)):
        palm = self.robot.data.body_pos_w[:, self.palm_id]
        quat = self.robot.data.body_quat_w[:, self.palm_id]
        return (float(torch.linalg.vector_norm(palm - self.target)) < position_tolerance
                and float(quat_error_magnitude(quat, self.target_quat)[0]) < angle_tolerance)

    def _basket_target(self, center_height):
        center = self._tensor((self.env.cfg.basket[0], self.env.cfg.basket[1], center_height))
        return center + self.palm_to_apple

    def _advance(self, metrics):
        if self.phase in ("done", "failed"):
            return
        if metrics.get("failure"):
            self._fail(metrics["failure"])
            return
        self.phase_steps += 1
        if self.phase_steps > TIMEOUT_STEPS[self.phase]:
            self._fail(f"{self.phase}_timeout")
            return
        if self.phase in ("lift", "transport", "lower"):
            self.contact_loss_steps = (0 if metrics.get("grasp_contact")
                                       else self.contact_loss_steps + 1)
            if self.contact_loss_steps > 15:
                self._fail("grasp_contact_lost")
                return
        if self.phase == "settle":
            if self.phase_steps >= 20 and metrics.get("initial_support_force_n", 0.0) > 0.1:
                self._set_geometry()
                self._set_phase("pregrasp", self.waypoints["pregrasp"], 0.0)
        elif self.phase == "pregrasp" and self._pose_ready():
            self._set_phase("approach", self.waypoints["approach"], 0.0)
        elif self.phase == "approach" and self._pose_ready(0.025, math.radians(12)):
            if metrics.get("hand_distance", 1.0) < 0.16:
                self._set_phase("close", self.waypoints["close"], 1.0)
        elif self.phase == "close":
            if metrics.get("grasp_contact") and self.phase_steps >= 30:
                self._set_phase("verify_grasp", self.waypoints["verify_grasp"], 1.0)
        elif self.phase == "verify_grasp":
            if metrics.get("grasp_contact") and self.phase_steps >= 10:
                self._set_phase("lift", self.waypoints["lift"], 1.0)
        elif self.phase == "lift":
            if metrics.get("lifted") and self._pose_ready(0.055):
                self.palm_to_apple = (self.robot.data.body_pos_w[:, self.palm_id]
                                      - self.apple.data.root_pos_w).clone()
                self._set_phase("transport", self._basket_target(self.env.cfg.basket[2] + 0.30), 1.0)
        elif self.phase == "transport":
            if metrics.get("carried") and self._pose_ready(0.05):
                self._set_phase("lower", self._basket_target(self.env.cfg.basket[2] + 0.17), 1.0)
        elif self.phase == "lower":
            pos = metrics.get("position", (float("inf"), float("inf"), float("inf")))
            basket = self.env.cfg.basket
            in_release_xy = abs(pos[0] - basket[0]) < 0.08 and abs(pos[1] - basket[1]) < 0.07
            if in_release_xy and self._pose_ready(0.05):
                self._set_phase("release", self.target, 0.0)
        elif self.phase == "release":
            if self.phase_steps >= 35 and metrics.get("finger_open"):
                self._set_phase("retreat", self.target + self._tensor((0.0, -0.20, 0.20)), 0.0)
        elif self.phase == "retreat" and self._pose_ready(0.06):
            self._set_phase("verify_placement", self.target, 0.0)
        elif self.phase == "verify_placement" and metrics.get("success"):
            self._set_phase("done", self.target, 0.0)

    def compute(self, step=0):
        metrics = getattr(self.env, "free_pick_place_metrics", {})
        self._advance(metrics)
        # Ease each waypoint in pose space. Joint increments are separately bounded by IK.
        blend = min(self.phase_steps / 45.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        position = self.start + blend * (self.target - self.start)
        grip = self.grip_start + min(self.phase_steps / 45.0, 1.0) * (
            self.grip_target - self.grip_start)
        return EndEffectorTarget(
            position=position,
            orientation=torch.stack([
                quat_slerp(self.start_quat[index], self.target_quat[index].clone(), blend)
                for index in range(self.env.num_envs)
            ]),
            gripper_closed_fraction=torch.tensor([[grip]], device=self.env.device),
        )

    def diagnostics(self):
        return {
            "phase": self.phase, "failed": self.failed, "failure": self.failure,
            "transitions": self.transitions,
            "metrics": getattr(self.env, "free_pick_place_last_metrics", {}),
            "grasp_offset_world_m": list(self.env.cfg.grasp_offset),
            "grasp_parameters": "provisional sphere grasp; requires CPU rollout and contact inspection",
        }


class FreePickPlaceScriptedPolicy(YCBSugarBoxScriptedPolicy):
    """Reuse bounded DLS and the exact soft-limit inverse for the 43-D action."""

    def __init__(self, env):
        super().__init__(env, strategy=FreePickPlaceStrategy(env))
