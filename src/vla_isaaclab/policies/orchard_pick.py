"""Orchard phases using the existing bounded G1 IK and normalized joint action."""

import torch

from vla_isaaclab.envs.common.g1 import LEFT_END_EFFECTOR
from vla_isaaclab.envs.orchard_pick.progress import Progress
from .ycb_sugar_box import YCBSugarBoxScriptedPolicy
from .ycb_sugar_box_strategy import EndEffectorTarget, SugarBoxPhaseStrategy


class OrchardStrategy:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.palm = self.robot.find_bodies([LEFT_END_EFFECTOR], preserve_order=True)[0][0]
        self.reset()

    def reset(self):
        self.progress = Progress()
        self.origin = self.env.scene.env_origins[0]
        layout = self.env.cfg.orchard_layout
        tensor = lambda value: torch.tensor(value, device=self.env.device, dtype=torch.float32)
        apple = tensor(layout.target) + self.origin
        basket = tensor(layout.basket) + self.origin
        # Provisional sphere-grasp offset, to be calibrated against lab rollouts.
        offset = tensor(self.env.cfg.grasp_offset)
        grasp = apple + offset
        self.targets = {
            "settle": self.robot.data.body_pos_w[0, self.palm].clone(),
            "pregrasp": grasp + tensor((0, -0.12, 0.07)),
            "approach": grasp,
            "close": grasp,
            "pull": grasp + tensor((0, -0.12, 0.10)),
            "transfer": basket + offset + tensor((0, 0, 0.25)),
            "lower": basket + offset + tensor((0, 0, 0.17)),
            "release": basket + offset + tensor((0, 0, 0.17)),
            "retreat": basket + offset + tensor((0, -0.18, 0.32)),
            "hold": basket + offset + tensor((0, -0.18, 0.32)),
        }
        self.orientation = tensor(SugarBoxPhaseStrategy.GRASP_QUAT_WXYZ).unsqueeze(0)

    @property
    def phase(self):
        return self.progress.phase

    @property
    def failed(self):
        return self.progress.failure is not None

    def compute(self, step=0):
        target = self.targets[self.phase]
        reached = float(torch.linalg.vector_norm(
            target - self.robot.data.body_pos_w[0, self.palm])) < 0.025
        metrics = self.env.orchard_metrics
        self.progress.update(reached=reached, detached=metrics.get("detached", False),
                             carried=metrics.get("carried", False),
                             grasp_contact=metrics.get("grasp_contact", False))
        if self.failed:
            self.env.orchard_failure = self.progress.failure
        phase = self.phase
        self.env.orchard_phase = phase
        self.env.policy_phase.fill_(self.progress.index)
        if phase == "pull":
            self.env.orchard_pull_started = True
        if phase == "release":
            self.env.orchard_released = True
        closure = float(phase in ("pull", "transfer", "lower"))
        if phase == "close":
            closure = min(self.progress.steps / 60, 1.0)
        elif phase == "release":
            closure = max(1.0 - self.progress.steps / 60, 0.0)
        return EndEffectorTarget(
            position=self.targets[phase].unsqueeze(0), orientation=self.orientation,
            gripper_closed_fraction=torch.tensor([[closure]], device=self.env.device),
        )

    def diagnostics(self):
        return dict(phase=self.phase, failed=self.failed, failure=self.progress.failure,
                    transitions=self.progress.transitions,
                    layout=self.env.cfg.orchard_layout.metadata(),
                    metrics=getattr(self.env, "orchard_last_metrics", {}),
                    grasp_offset_world_m=list(self.env.cfg.grasp_offset),
                    grasp_parameters="provisional; review contact and wrist videos on the lab machine")


class OrchardScriptedPolicy(YCBSugarBoxScriptedPolicy):
    def __init__(self, env):
        super().__init__(env, strategy=OrchardStrategy(env))
