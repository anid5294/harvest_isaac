"""Baseline RL task: pick the apple off the tree and place it in the basket.

A subclass of OrchardEnv with a dense staged reward, a 17-joint action space (waist + left
arm + left hand, as offsets from the home pose) and actor/critic observations. It leaves
OrchardEnv (and its empty _get_rewards) untouched. Success/failure are harvest_isaac's rules.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..orchard_env import OrchardEnv
from ..orchard_env_cfg import FRUIT
from .pick_place_env_cfg import PickPlaceEnvCfg

LEFT_HAND_TIPS = ["left_hand_thumb_2_link", "left_hand_index_1_link", "left_hand_middle_1_link"]


class PickPlaceEnv(OrchardEnv):
    cfg: PickPlaceEnvCfg

    def _configure_gym_env_spaces(self):
        self._init_buffers()
        n = self.num_controlled_joints
        self._tip_ids = self._robot.find_bodies(LEFT_HAND_TIPS, preserve_order=True)[0]
        self.cfg.action_space = n
        self.cfg.observation_space = 3 * n + 3 + 9 + 4 + 3 + 3 + 1 + 7
        self.cfg.state_space = self.cfg.observation_space + 6
        super(OrchardEnv, self)._configure_gym_env_spaces()
        self._prev_actions = torch.zeros_like(self.actions)
        # Previous potentials (reach, pull, lift, transport); invalid right after a reset.
        self._phi_prev = torch.zeros(self.num_envs, 4, device=self.device)
        self._phi_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._paid = {k: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                      for k in ("first_grasp", "detach", "carried", "place", "release")}
        self._terms = {k: torch.zeros(self.num_envs, device=self.device) for k in (
            "reach", "pull", "lift", "transport", "first_grasp", "detach", "carried", "place", "release",
            "success", "time", "failure", "action_rate", "joint_vel")}

    # ------------------------------------------------------------------------------------

    def _pull_toward_robot(self, apple_pos, root_pos):
        """Stem force [N] the hand is applying in the direction of the robot's chest.

        The stem spring's elastic force on the apple is -K * (pos - rest); the hand holds that
        displacement with the opposite force K * (pos - rest). Only its component along the
        apple -> chest direction counts; sideways bending or twisting adds nothing.
        """
        disp = apple_pos - self._stem_rest
        disp_ax = (disp * self._stem_axis).sum(-1, keepdim=True) * self._stem_axis
        hand_force = self._k_ax * disp_ax + self._k_lat * (disp - disp_ax)
        chest = root_pos + torch.tensor([0.0, 0.0, 0.25], device=self.device)
        toward = chest - self._apple_center
        toward = toward / torch.linalg.norm(toward, dim=-1, keepdim=True)
        return (hand_force * toward).sum(-1)

    def _grasp_point(self):
        """Mean of the thumb, index and middle fingertip links (env-local)."""
        return self._robot.data.body_pos_w.torch[:, self._tip_ids].mean(1) - self.scene.env_origins

    def _get_observations(self) -> dict:
        s = self.get_state()
        robot = self._robot.data
        ids = self.ctrl_joint_ids
        apple = s["apple_pos"]
        tips = robot.body_pos_w.torch[:, self._tip_ids] - self.scene.env_origins.unsqueeze(1)
        palm = s["body_pos"][:, 0]
        n_fingers = len(self.cfg.robot.grasp_thumb_bodies) + len(self.cfg.robot.grasp_finger_bodies)
        actor = torch.cat((
            robot.joint_pos.torch[:, ids] - robot.default_joint_pos.torch[:, ids],
            0.1 * robot.joint_vel.torch[:, ids],
            self.actions,
            apple - palm,
            (apple.unsqueeze(1) - tips).flatten(1),
            s["body_quat"][:, 0],
            self.basket_pos - apple,
            s["apple_lin_vel"],
            s["apple_attached"].float().unsqueeze(-1),
            torch.clamp(s["finger_contact_force"][:, :n_fingers], 0.0, 5.0) / 5.0,
        ), dim=-1)
        privileged = torch.stack((
            s["stem_tension"] / self.stem_break_force,
            s["grasp_contact"].float(),
            s["carried"].float(),
            s["released"].float(),
            s["apple_settled_in_basket"].float(),
            (apple[:, 2] - self._apple_center[:, 2]) * 10.0,
        ), dim=-1)
        return {"policy": actor, "critic": torch.cat((actor, privileged), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        c, h = self.cfg, self.harvest
        s = self.get_state()
        apple = s["apple_pos"]
        attached = s["apple_attached"]
        ok = s["failure"] == 0

        d_grasp = torch.linalg.norm(self._grasp_point() - apple, dim=-1)
        lift = torch.clamp((apple[:, 2] - self._apple_center[:, 2]) / 0.06, 0.0, 1.0)
        # Where the apple rests on the basket floor (floor top is 1 cm above basket_pos).
        target = self.basket_pos + torch.tensor([0.0, 0.0, 0.01 + FRUIT.half_extents[2]], device=self.device)
        d_basket = torch.linalg.norm(apple - target, dim=-1)

        # Potentials in [0, 1]; only their change is rewarded, so holding still earns nothing.
        gripping = h.grasp_contact.float()
        pull = torch.clamp(self._pull_toward_robot(apple, s["root_pos"]) / self.stem_break_force, 0.0, 1.0)
        phi = torch.stack((
            1.0 - torch.tanh(d_grasp / 0.1),  # reach
            torch.where(attached, gripping * pull, self._phi_prev[:, 1]),  # pull (frozen once snapped)
            torch.where(attached, torch.zeros_like(lift), lift),  # lift
            1.0 - torch.tanh(d_basket / 0.2),  # transport: apple closeness to its resting spot in the basket
        ), dim=-1)
        delta = torch.where(self._phi_valid.unsqueeze(-1), phi - self._phi_prev, torch.zeros_like(phi))
        # Snap bonus scales with how much of the stem force pointed at the robot just before the
        # stem broke (the frozen pull potential), so bending/twisting it off earns nothing.
        pulled_off = torch.clamp(self._phi_prev[:, 1] / 0.5, 0.0, 1.0)
        self._phi_prev[:] = phi
        self._phi_valid[:] = True

        def once(name, condition):
            new = condition & ~self._paid[name]
            self._paid[name] |= condition
            return new.float()

        terms = {
            "reach": c.w_reach * delta[:, 0],
            "pull": c.w_pull * delta[:, 1],
            "lift": c.w_lift * delta[:, 2],
            # Progress toward the basket counts only while the hand grips the apple right now;
            # not gripping = 0. Holding still earns nothing, so releasing costs nothing.
            "transport": c.w_transport * gripping * delta[:, 3],
            "first_grasp": c.w_first_grasp * once("first_grasp", h.grasp_contact & attached),
            "detach": c.w_detach * once("detach", ~attached & ok) * pulled_off,
            "carried": c.w_carried * once("carried", h.carried),
            "place": c.w_place * once("place", h.carried & s["apple_settled_in_basket"]),
            "release": c.w_release * once("release", h.released),
            "success": c.w_success * h.success.float(),
            "time": torch.full_like(lift, c.w_time),
            "failure": c.w_failure * (~ok).float(),
            "action_rate": c.w_action_rate * torch.sum((self.actions - self._prev_actions) ** 2, dim=-1),
            "joint_vel": c.w_joint_vel * torch.sum(s["ctrl_joint_vel"] ** 2, dim=-1),
        }
        reward = torch.zeros(self.num_envs, device=self.device)
        for name, value in terms.items():
            reward += value
            self._terms[name] += value
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        log = self.extras.setdefault("log", {})
        for name, total in self._terms.items():
            log[f"Episode_Reward/{name}"] = total[env_ids].mean().item()
            total[env_ids] = 0.0
        log["Metrics/grasp_contact"] = self.harvest.grasp_contact[env_ids].float().mean().item()
        super()._reset_idx(env_ids)
        self._phi_valid[env_ids] = False
        for paid in self._paid.values():
            paid[env_ids] = False
