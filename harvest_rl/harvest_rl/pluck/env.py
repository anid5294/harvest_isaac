"""Grip-pluck-hold task: the reward the user specified, on the randomized apple env.

reach  = w_reach * (prev - curr fingertip-to-apple distance), until the first grasp
grasp  = w_grasp once, when thumb, index and middle all touch the apple on the stem
pluck  = w_pluck once, when the stem snaps while all three touch (last 3 steps)
hold   = w_hold every step after the pluck while all three touch
chest  = w_chest * (prev - curr apple-to-chest distance), every step while all three touch
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..random_apple.env import RandomAppleEnv
from .env_cfg import PluckEnvCfg

TERMS = ("reach", "grasp", "pluck", "hold", "chest")


class PluckEnv(RandomAppleEnv):
    cfg: PluckEnvCfg

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        n, dev = self.num_envs, self.device
        self._terms = {k: torch.zeros(n, device=dev) for k in TERMS}
        self._prev_reach = torch.zeros(n, device=dev)
        self._prev_chest = torch.zeros(n, device=dev)
        self._valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._grasped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._plucked = torch.zeros(n, dtype=torch.bool, device=dev)
        self._touch_age = torch.full((n,), 1000, dtype=torch.long, device=dev)
        self._hold_steps = torch.zeros(n, device=dev)
        self.pluck_counts = {k: 0 for k in ("episodes", "grasped", "plucked", "held_to_end",
                                            "dropped_after_pluck", "knocked_off_without_grip")}
        self.pluck_counts["hold_seconds_when_plucked"] = 0.0

    def _all_fingers_touch(self, s) -> torch.Tensor:
        # contact_bodies order (robot.py): left thumb 0-2, left middle 3-4, left index 5-6.
        f = s["finger_contact_force"] >= self.cfg.touch_threshold
        return f[:, 0:3].any(-1) & f[:, 3:5].any(-1) & f[:, 5:7].any(-1)

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        s = self.get_state()
        apple, attached = s["apple_pos"], s["apple_attached"]
        touch = self._all_fingers_touch(s)
        self._touch_age = torch.where(touch, torch.zeros_like(self._touch_age), self._touch_age + 1)

        d_reach = torch.linalg.norm(self._grasp_point() - apple, dim=-1)
        chest = s["root_pos"] + torch.tensor([0.0, 0.0, 0.25], device=self.device)
        d_chest = torch.linalg.norm(apple - chest, dim=-1)
        valid = self._valid.float()

        new_grasp = touch & attached & ~self._grasped
        new_pluck = ~attached & ~self._plucked & (self._touch_age <= 2) & (s["failure"] == 0)
        holding = self._plucked & touch

        terms = {
            "reach": c.w_reach * (self._prev_reach - d_reach) * valid * (~self._grasped).float(),
            "grasp": c.w_grasp * new_grasp.float(),
            "pluck": c.w_pluck * new_pluck.float(),
            "hold": c.w_hold * holding.float(),
            "chest": c.w_chest * (self._prev_chest - d_chest) * valid * touch.float(),
        }
        self._grasped |= new_grasp
        self._plucked |= new_pluck
        self._hold_steps += holding.float()
        self._prev_reach[:], self._prev_chest[:] = d_reach, d_chest
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
        log = self.extras.setdefault("log", {})
        ran = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(ran) > 0:
            log["Pluck/grasped"] = self._grasped[ran].float().mean().item()
            log["Pluck/plucked_with_grip"] = self._plucked[ran].float().mean().item()
            log["Pluck/hold_seconds"] = (self._hold_steps[ran] * self.step_dt).mean().item()
            # Running totals for evaluation scripts.
            pc = self.pluck_counts
            plucked = self._plucked[ran]
            fail = self.harvest.failure[ran]
            pc["episodes"] += len(ran)
            pc["grasped"] += int(self._grasped[ran].sum())
            pc["plucked"] += int(plucked.sum())
            pc["held_to_end"] += int((plucked & (fail == 0)).sum())
            pc["dropped_after_pluck"] += int((plucked & (fail != 0)).sum())
            pc["knocked_off_without_grip"] += int((~plucked & ~self.apple_attached[ran]).sum())
            pc["hold_seconds_when_plucked"] += float((self._hold_steps[ran][plucked] * self.step_dt).sum())
        super()._reset_idx(env_ids)
        for buf in (self._valid, self._grasped, self._plucked):
            buf[env_ids] = False
        self._touch_age[env_ids] = 1000
        self._hold_steps[env_ids] = 0.0
