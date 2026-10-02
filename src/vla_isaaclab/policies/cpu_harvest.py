"""CPU attached-apple trial: measured grasp, native stem break, physical placement."""

import math

import torch
from isaaclab.utils.math import quat_error_magnitude, quat_slerp

from vla_isaaclab.envs.common.g1 import LEFT_END_EFFECTOR
from vla_isaaclab.envs.common.managers import ACTION_TERM_NAME
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
        result = EndEffectorTarget(
            position=self.start + blend * (self.target - self.start),
            orientation=torch.stack([
                quat_slerp(self.start_quat[i], self.target_quat[i].clone(), blend)
                for i in range(self.env.num_envs)
            ]),
            gripper_closed_fraction=torch.tensor([[grip]], device=self.env.device),
        )
        self.last_ee_target = result
        return result

    def diagnostics(self):
        metrics = getattr(self.env, "orchard_last_metrics", {})
        failure = self.fsm.failure or metrics.get("failure") or getattr(self.env, "orchard_failure", None)
        break_step = self.fsm.break_step
        if break_step is None and metrics.get("detached"):
            # Termination may follow env.step() without another policy.compute().
            break_step = int(self.env.episode_length_buf[0])
        return {
            "phase": self.phase, "failed": bool(failure), "failure": failure,
            "transitions": self.fsm.transitions,
            "grasp_observed_control_step": self.fsm.grasp_step,
            "native_break_observed_control_step": break_step,
            "native_break_observation_source": (
                "policy_fsm" if self.fsm.break_step is not None else
                "terminal_environment_metrics" if break_step is not None else None
            ),
            "event_timing_resolution": "30 Hz control-step observation; internal PhysX break time unknown",
            "pull_distance_commanded_m": self.pull_distance_m,
            "physics_device": str(self.env.device),
            "metrics": metrics,
            "grasp_offset_world_m": list(self.env.cfg.grasp_offset),
            "grasp_parameters": "provisional; requires CPU physical rollout and measured contact review",
        }


class CPUHarvestScriptedPolicy(YCBSugarBoxScriptedPolicy):
    """Use the existing bounded DLS and exact normalized 43-D joint mapping."""

    def __init__(self, env):
        super().__init__(env, strategy=CPUHarvestStrategy(env))
        self.orientation_weight = float(getattr(env.cfg, "cpu_harvest_orientation_weight", 0.20))
        if not 0.0 <= self.orientation_weight <= 1.0 or not math.isfinite(self.orientation_weight):
            raise ValueError("cpu_harvest_orientation_weight must be finite and in [0, 1]")
        self.control_trace = []

    def reset(self):
        super().reset()
        self.control_trace = []

    def compute(self, step=0):
        # Capture the target presented to the actuator before issuing this step's action.
        previous_target = self.robot.data.joint_pos_target[0, self.arm_joint_ids].detach().cpu().tolist()
        previous_command = self.last_joint_targets[0].clone()
        action = super().compute(step)
        target = self.strategy.last_ee_target
        palm_pos = self.robot.data.body_pos_w[0, self.palm_body_id]
        palm_quat = self.robot.data.body_quat_w[0, self.palm_body_id]
        current = self.robot.data.joint_pos[0, self.arm_joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[0, self.arm_joint_ids]
        normalized = action[0, self.arm_action_indices]
        mapped = limits[:, 0] + 0.5 * (normalized + 1.0) * (limits[:, 1] - limits[:, 0])
        requested = self.last_joint_targets[0]
        lower, upper = limits[:, 0].clone(), limits[:, 1].clone()
        waist_default = self.robot.data.default_joint_pos[0, self.arm_joint_ids[0]]
        lower[0] = torch.maximum(lower[0], waist_default - self.max_waist_yaw_deviation)
        upper[0] = torch.minimum(upper[0], waist_default + self.max_waist_yaw_deviation)
        reference = torch.clamp(previous_command, current - self.max_tracking_error,
                                current + self.max_tracking_error)
        reference = torch.clamp(reference, lower, upper)
        self.control_trace.append({
            "step": int(self.env.episode_length_buf[0]),
            "phase": self.phase,
            "ee_position_m": palm_pos.detach().cpu().tolist(),
            "ee_target_m": target.position[0].detach().cpu().tolist(),
            "ee_quaternion_wxyz": palm_quat.detach().cpu().tolist(),
            "ee_target_quaternion_wxyz": target.orientation[0].detach().cpu().tolist(),
            "ee_body": LEFT_END_EFFECTOR,
            "ee_body_index": self.palm_body_id,
            "jacobian_body_index": self.palm_jacobian_id,
            "ee_position_error_m": float(torch.linalg.vector_norm(target.position[0] - palm_pos)),
            "ee_orientation_error_rad": float(quat_error_magnitude(palm_quat.unsqueeze(0), target.orientation)[0]),
            "orientation_weight": self.orientation_weight,
            "arm_joint_names": self.arm_joint_names,
            "arm_joint_position_rad": current.detach().cpu().tolist(),
            "arm_previous_actuator_target_rad": previous_target,
            "arm_requested_target_rad": requested.detach().cpu().tolist(),
            "arm_previous_command_rad": previous_command.detach().cpu().tolist(),
            "arm_ik_reference_rad": reference.detach().cpu().tolist(),
            "arm_ik_correction_rad": (requested - reference).detach().cpu().tolist(),
            "arm_ik_correction_at_bound": ((requested - reference).abs() >= self.max_joint_delta - 1e-5).detach().cpu().tolist(),
            "arm_effective_lower_rad": lower.detach().cpu().tolist(),
            "arm_effective_upper_rad": upper.detach().cpu().tolist(),
            "arm_requested_at_effective_limit": ((requested - lower).abs().minimum((requested - upper).abs()) < 1e-5).detach().cpu().tolist(),
            "arm_action_normalized": normalized.detach().cpu().tolist(),
            "arm_action_remapped_target_rad": mapped.detach().cpu().tolist(),
            "arm_soft_lower_rad": limits[:, 0].detach().cpu().tolist(),
            "arm_soft_upper_rad": limits[:, 1].detach().cpu().tolist(),
            "arm_action_saturated": (normalized.abs() >= 1.0 - 1.0e-5).detach().cpu().tolist(),
        })
        return action

    def observe_after_step(self):
        """Append what the action term and articulation actually did on the issued step."""
        if not self.control_trace:
            return
        sample = self.control_trace[-1]
        term = self.env.action_manager.get_term(ACTION_TERM_NAME)
        processed = term.processed_actions[0, self.arm_action_indices]
        actual = self.robot.data.joint_pos[0, self.arm_joint_ids]
        actuator_target = self.robot.data.joint_pos_target[0, self.arm_joint_ids]
        requested = self.last_joint_targets[0]
        palm_pos = self.robot.data.body_pos_w[0, self.palm_body_id]
        palm_quat = self.robot.data.body_quat_w[0, self.palm_body_id]
        ee_target = self.strategy.last_ee_target
        sample.update({
            "post_step": int(self.env.episode_length_buf[0]),
            "ee_post_step_position_m": palm_pos.detach().cpu().tolist(),
            "ee_post_step_quaternion_wxyz": palm_quat.detach().cpu().tolist(),
            "ee_post_step_position_error_m": float(torch.linalg.vector_norm(ee_target.position[0] - palm_pos)),
            "ee_post_step_orientation_error_rad": float(quat_error_magnitude(palm_quat.unsqueeze(0), ee_target.orientation)[0]),
            "arm_processed_action_target_rad": processed.detach().cpu().tolist(),
            "arm_actuator_target_rad": actuator_target.detach().cpu().tolist(),
            "arm_post_step_position_rad": actual.detach().cpu().tolist(),
            "arm_post_step_target_error_rad": (actuator_target - actual).detach().cpu().tolist(),
            "arm_processed_vs_requested_error_rad": (processed - requested).detach().cpu().tolist(),
            "arm_actuator_vs_processed_error_rad": (actuator_target - processed).detach().cpu().tolist(),
        })
        metrics = getattr(self.env, "orchard_last_metrics", {})
        sample["terminal_failure"] = metrics.get("failure") or getattr(self.env, "orchard_failure", None)
        sample["detached"] = bool(metrics.get("detached"))
        sample["grasp_contact"] = bool(metrics.get("grasp_contact"))

    def diagnostics(self):
        result = super().diagnostics()
        result["control"]["orientation_weight"] = self.orientation_weight
        result["control"]["trace_steps"] = len(self.control_trace)
        return result
