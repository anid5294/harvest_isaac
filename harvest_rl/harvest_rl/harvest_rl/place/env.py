"""Pluck-and-place task: the pluck task's reach/grasp/pluck rewards, then carry the apple to a
random basket and let go.

reach    = w_reach * (prev - curr fingertip-to-apple distance), until the first grasp
grasp    = w_grasp once, thumb + index + middle all touching the apple on the stem
pluck    = w_pluck once, stem snaps while all three touch (last 3 steps)
basket   = w_basket * (prev - curr apple-to-basket-resting-spot distance), while all three touch
release  = w_release_shaping * progress(opening the hand, then moving the palm away), once the
           plucked apple is low inside the basket walls
placed   = w_placed once, plucked apple settled in the basket on its floor
success  = w_success, harvest_isaac's success rule (episode ends)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..orchard_env_cfg import BASKET_INNER_SIZE, FRUIT
from ..pluck.env import PluckEnv
from .env_cfg import PlaceEnvCfg

TERMS = ("reach", "grasp", "pluck", "basket", "release", "placed", "success")


class PlaceEnv(PluckEnv):
    cfg: PlaceEnvCfg

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        n, dev = self.num_envs, self.device
        self._terms = {k: torch.zeros(n, device=dev) for k in TERMS}
        self._prev_basket = torch.zeros(n, device=dev)
        self._placed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.place_counts = {"episodes": 0, "plucked": 0, "placed": 0, "success": 0}

    def _get_rewards(self) -> torch.Tensor:
        c, h = self.cfg, self.harvest
        s = self.get_state()
        apple, attached = s["apple_pos"], s["apple_attached"]
        touch = self._all_fingers_touch(s)
        self._touch_age = torch.where(touch, torch.zeros_like(self._touch_age), self._touch_age + 1)
        valid = self._valid.float()

        d_reach = torch.linalg.norm(self._grasp_point() - apple, dim=-1)
        rest_spot = self._basket_pos + torch.tensor([0.0, 0.0, 0.01 + FRUIT.half_extents[2]], device=self.device)
        d_basket = torch.linalg.norm(apple - rest_spot, dim=-1)

        # Release potential: only once the plucked apple is low inside the basket walls.
        rel = apple - self._basket_pos
        r = FRUIT.radius
        in_zone = (
            self._plucked & ~attached
            & (rel[:, 0].abs() < BASKET_INNER_SIZE[0] / 2 - 0.01 - r)
            & (rel[:, 1].abs() < BASKET_INNER_SIZE[1] / 2 - 0.01 - r)
            & (rel[:, 2] < 0.01 + FRUIT.half_extents[2] + 0.08)
        )
        hand = self._robot.data.joint_pos.torch[:, self._hand_ids].abs().max(-1).values
        openness = 1.0 - torch.clamp(hand, 0.0, 1.2) / 1.2
        away = torch.clamp(h.hand_distance / 0.2, 0.0, 1.0) * h.opened.float()
        phi_release = in_zone.float() * (0.5 * openness + 0.5 * away)

        new_grasp = touch & attached & ~self._grasped
        new_pluck = ~attached & ~self._plucked & (self._touch_age <= 2) & (s["failure"] == 0)
        new_placed = self._plucked & h.settled & h.supported & ~self._placed

        terms = {
            "reach": c.w_reach * (self._prev_reach - d_reach) * valid * (~self._grasped).float(),
            "grasp": c.w_grasp * new_grasp.float(),
            "pluck": c.w_pluck * new_pluck.float(),
            "basket": c.w_basket * (self._prev_basket - d_basket) * valid * touch.float(),
            "release": c.w_release_shaping * (phi_release - self._phi_release) * valid,
            "placed": c.w_placed * new_placed.float(),
            "success": c.w_success * h.success.float(),
        }
        self._grasped |= new_grasp
        self._plucked |= new_pluck
        self._placed |= new_placed
        self._prev_reach[:], self._prev_basket[:] = d_reach, d_basket
        self._phi_release[:] = phi_release
        self._valid[:] = True

        reward = torch.zeros(self.num_envs, device=self.device)
        for name, value in terms.items():
            reward += value
            self._terms[name] += value
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ran = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(ran) > 0:
            log = self.extras.setdefault("log", {})
            log["Place/placed"] = self._placed[ran].float().mean().item()
            log["Place/success"] = self.harvest.success[ran].float().mean().item()
            pc = self.place_counts
            pc["episodes"] += len(ran)
            pc["plucked"] += int(self._plucked[ran].sum())
            pc["placed"] += int(self._placed[ran].sum())
            pc["success"] += int(self.harvest.success[ran].sum())
        super()._reset_idx(env_ids)
        self._placed[env_ids] = False
