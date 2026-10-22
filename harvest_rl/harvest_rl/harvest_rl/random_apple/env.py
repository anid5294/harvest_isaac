"""Pick-and-place with the apple (and optionally the basket) at a random spot every episode.

The apple, its (virtual) stem attachment point and the visual stem are shifted together by a
random offset, so the stem physics and harvest_isaac's success rules work exactly as before.
Offsets that would put the apple inside a branch are rejected (conservative bounding-sphere
check against tree.json's wood). Because the attachment point moves with the apple, a shifted
apple hangs near, not from, a branch. The basket (floor, walls, pedestal) shifts as one piece.

Training extras (see RandomizedTrainEnvCfg): random start pose for the controlled joints only,
and a release-shaping reward. Per-episode outcomes are counted in ``RandomAppleEnv.counts``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence

import torch

from ..baseline.pick_place_env import PickPlaceEnv
from ..orchard_env_cfg import BASKET_INNER_SIZE, FRUIT
from ..task_logic import APPLE_DROPPED, DETACHED_WITHOUT_GRASP, NO_FAILURE
from ..tree import TREE_JSON
from .env_cfg import BASKET_PARTS, RandomAppleEnvCfg


def _clear_of_wood(center, radius, branches):
    """Same test as harvest_isaac's wood_clearance: sphere vs. every tapered branch segment."""
    for b in branches:
        start, end = b["start"], b["end"]
        d = [e - s for s, e in zip(start, end)]
        t = sum((p - s) * di for p, s, di in zip(center, start, d)) / sum(di * di for di in d)
        q = [s + max(0.0, min(1.0, t)) * di for s, di in zip(start, d)]
        if math.dist(center, q) - radius - max(b["radius_start"], b["radius_end"]) <= 0.0:
            return False
    return True


def valid_offsets(cfg: RandomAppleEnvCfg) -> list[tuple[float, float, float]]:
    branches = json.loads(TREE_JSON.read_text())["branches"]
    r = max(FRUIT.half_extents)
    n_xy, n_z = cfg.offset_grid, max(3, cfg.offset_grid // 2 + 1)

    def axis(limit, n):
        return [0.0] if limit == 0.0 else [-limit + 2 * limit * i / (n - 1) for i in range(n)]

    out = []
    for dx in axis(cfg.apple_offset_xy, n_xy):
        for dy in axis(cfg.apple_offset_xy, n_xy):
            for dz in axis(cfg.apple_offset_z, n_z):
                c = [FRUIT.center[0] + dx, FRUIT.center[1] + dy, FRUIT.center[2] + dz]
                if _clear_of_wood(c, r, branches):
                    out.append((dx, dy, dz))
    if not out:
        raise ValueError("No apple offset is clear of the branches")
    return out


class RandomAppleEnv(PickPlaceEnv):
    cfg: RandomAppleEnvCfg

    OUTCOMES = ("episodes", "gripped", "detached", "pulled_off_with_grip", "carried", "placed",
                "released", "success", "apple_dropped", "detached_without_sustained_grasp", "timeout")

    def _setup_scene(self):
        super()._setup_scene()
        self._stem = self.scene["stem"]
        self._basket_parts = [self.scene[name] for name in BASKET_PARTS]

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        self._terms["release_shaping"] = torch.zeros(self.num_envs, device=self.device)
        self._phi_release = torch.zeros(self.num_envs, device=self.device)

    def _init_buffers(self):
        super()._init_buffers()
        self._base_center = self._apple_center.clone()
        self._base_anchor = self._stem_anchor.clone()
        self._base_basket = self._basket_pos.clone()
        self._rest_minus_center = self._stem_rest - self._apple_center
        self._offsets = torch.tensor(valid_offsets(self.cfg), device=self.device)
        # Lift and basket checks follow each episode's own apple height and basket position.
        self._rest_height = self._apple_center[:, 2].clone()
        b = self._basket_pos
        self.harvest.p = dataclasses.replace(
            self.harvest.p, apple_rest_height=self._rest_height, basket_center=(b[:, 0], b[:, 1], b[:, 2])
        )
        self._hand_ids = self._robot.find_joints(self.cfg.robot.grasp_hand_joints, preserve_order=True)[0]
        self.counts = {k: 0 for k in self.OUTCOMES}
        print(f"[RandomApple] {len(self._offsets)} candidate apple offsets clear of the branches")

    # ------------------------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        if self.cfg.w_release_shaping == 0.0:
            return reward
        # Once the carried apple is low inside the basket walls: progress in opening the hand,
        # then in moving the palm away (harvest_isaac's release needs open hand, palm > 18 cm).
        h = self.harvest
        apple = self._apple.data.root_pos_w.torch - self.scene.env_origins
        rel = apple - self._basket_pos
        r = FRUIT.radius
        in_zone = (
            h.carried & ~self.apple_attached
            & (rel[:, 0].abs() < BASKET_INNER_SIZE[0] / 2 - 0.01 - r)
            & (rel[:, 1].abs() < BASKET_INNER_SIZE[1] / 2 - 0.01 - r)
            & (rel[:, 2] < 0.01 + FRUIT.half_extents[2] + 0.08)
        )
        hand = self._robot.data.joint_pos.torch[:, self._hand_ids].abs().max(-1).values
        openness = 1.0 - torch.clamp(hand, 0.0, 1.2) / 1.2
        away = torch.clamp(h.hand_distance / 0.2, 0.0, 1.0) * h.opened.float()
        phi = in_zone.float() * (0.5 * openness + 0.5 * away)
        delta = torch.where(self._phi_valid, phi - self._phi_release, torch.zeros_like(phi))
        self._phi_release[:] = phi
        term = self.cfg.w_release_shaping * delta
        self._terms["release_shaping"] += term
        return reward + term

    # ------------------------------------------------------------------------------------

    def _record(self, env_ids):
        ran = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(ran) == 0:
            return
        h = self.harvest
        fail = h.failure[ran]
        timeout = self.episode_length_buf[ran] >= self.max_episode_length - 1
        counted = {
            "episodes": torch.ones_like(fail, dtype=torch.bool),
            "gripped": self._paid["first_grasp"][ran],
            "detached": ~self.apple_attached[ran],
            "pulled_off_with_grip": ~self.apple_attached[ran] & (fail != DETACHED_WITHOUT_GRASP),
            "carried": h.carried[ran],
            "placed": self._paid["place"][ran],
            "released": self._paid["release"][ran],
            "success": h.success[ran],
            "apple_dropped": fail == APPLE_DROPPED,
            "detached_without_sustained_grasp": fail == DETACHED_WITHOUT_GRASP,
            "timeout": timeout & (fail == NO_FAILURE) & ~h.success[ran],
        }
        for k, v in counted.items():
            self.counts[k] += int(v.sum())

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._record(env_ids)
        k = len(env_ids)
        origins = self.scene.env_origins[env_ids]

        # New apple position (stem anchor and spring rest point move with it).
        offset = self._offsets[torch.randint(len(self._offsets), (k,), device=self.device)]
        self._apple_center[env_ids] = self._base_center[env_ids] + offset
        self._stem_anchor[env_ids] = self._base_anchor[env_ids] + offset
        self._stem_rest[env_ids] = self._apple_center[env_ids] + self._rest_minus_center[env_ids]
        self._rest_height[env_ids] = self._apple_center[env_ids, 2]

        # New basket position (all parts move together, horizontally).
        b_off = torch.zeros(k, 3, device=self.device)
        b_off[:, :2] = (torch.rand(k, 2, device=self.device) * 2.0 - 1.0) * self.cfg.basket_offset_xy
        self._basket_pos[env_ids] = self._base_basket[env_ids] + b_off

        super()._reset_idx(env_ids)
        self._phi_release[env_ids] = 0.0

        for part, shift in [(self._stem, offset)] + [(p, b_off) for p in self._basket_parts]:
            pose = part.data.default_root_pose.torch[env_ids].clone()
            pose[:, :3] += origins + shift
            part.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)

        # Random start pose for the controlled (arm + hand) joints only.
        if self.cfg.arm_reset_noise > 0.0:
            ids = self.ctrl_joint_ids
            q = self._robot.data.default_joint_pos.torch[env_ids][:, ids]
            q = q + (torch.rand_like(q) * 2.0 - 1.0) * self.cfg.arm_reset_noise
            q = torch.clamp(q, self._ctrl_lower[env_ids], self._ctrl_upper[env_ids])
            self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=env_ids)
            self._robot.set_joint_position_target_index(target=q, joint_ids=ids, env_ids=env_ids)
            self._joint_targets[env_ids] = q
