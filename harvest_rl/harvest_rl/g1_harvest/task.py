"""Harvest stages and outcome for the target apple, with a basket that moves with the robot.

Same rules as :mod:`harvest_rl.task_logic` (harvest_isaac's evidence chain), with the basket
checks done in the **basket's own frame** (it is welded to the G1 torso), so the target is
re-evaluated in world coordinates every step. Pure torch.

Per control step:

* ``grasp_contact``: thumb link and an opposing finger link both press on the target apple
  (>= ``contact_threshold``) for ``sustained_contact_steps`` steps.
* failure ``detached_without_sustained_grasp``: the target's stem broke without a recent grasp.
* ``carried``: detached, gripped, lifted ``lift_height`` above where it hung, palm close,
  for ``carry_steps`` steps.
* ``released``: carried, then hand open, no finger contact, palm > ``release_hand_distance`` away.
* ``placed``: the apple rests inside the basket walls on its floor (basket frame), the floor
  carries >= ``support_fraction`` of its weight, and it is slow.
* ``success``: carried + released + placed, held for ``success_hold_steps`` steps.
* failures ``apple_dropped`` (below ``drop_height`` above the local ground) and ``non_finite_state``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FAILURE_NAMES = ("none", "apple_dropped", "non_finite_state", "detached_without_sustained_grasp")
NO_FAILURE, APPLE_DROPPED, NON_FINITE, DETACHED_WITHOUT_GRASP = range(4)


@dataclass
class HarvestRules:
    basket_half_inner: tuple[float, float] = (0.11, 0.10)
    contact_threshold: float = 0.05
    sustained_contact_steps: int = 5
    recent_contact_steps: int = 2
    lift_height: float = 0.06
    lift_hand_distance: float = 0.18
    carry_steps: int = 10
    release_hand_distance: float = 0.15
    hand_open_threshold: float = 0.30
    support_fraction: float = 0.4
    settle_height_tolerance: float = 0.015
    settle_speed: float = 0.05
    success_hold_steps: int = 30
    drop_height: float = 0.40
    gravity: float = 9.81


class TargetHarvestTracker:
    def __init__(self, num_envs: int, rules: HarvestRules, device):
        self.r = rules
        n = num_envs
        b = lambda: torch.zeros(n, dtype=torch.bool, device=device)  # noqa: E731
        i = lambda v=0: torch.full((n,), v, dtype=torch.long, device=device)  # noqa: E731
        self.contact_steps, self.contact_age, self.lift_steps, self.hold_steps = i(), i(1_000_000), i(), i()
        self.detached_seen, self.evidence_failure, self.carried, self.released = b(), b(), b(), b()
        self.success, self.opposing_contact, self.grasp_contact, self.opened = b(), b(), b(), b()
        self.supported, self.settled, self.placed = b(), b(), b()
        self.failure = i()
        self.hand_distance = torch.zeros(n, device=device)

    def reset(self, env_ids):
        for buf in (self.contact_steps, self.lift_steps, self.hold_steps, self.failure):
            buf[env_ids] = 0
        self.contact_age[env_ids] = 1_000_000
        for buf in (self.detached_seen, self.evidence_failure, self.carried, self.released, self.success,
                    self.opposing_contact, self.grasp_contact, self.opened, self.supported, self.settled, self.placed):
            buf[env_ids] = False
        self.hand_distance[env_ids] = 0.0

    def update(
        self,
        *,
        detached: torch.Tensor,  # (n,) target stem broken
        apple_pos: torch.Tensor,  # (n, 3) env-local
        apple_rest_height: torch.Tensor,  # (n,) z where the target hung
        ground_height: torch.Tensor,  # (n,) local ground under the apple (0 on the pads)
        apple_speed: torch.Tensor,  # (n,)
        apple_mass: torch.Tensor,  # (n,)
        apple_half_height: torch.Tensor,  # (n,)
        apple_in_basket: torch.Tensor,  # (n, 3) apple centre in the basket frame (origin = floor top centre)
        basket_normal_force: torch.Tensor,  # (n,) basket's force on the apple along the basket's up axis [N]
        palm_pos: torch.Tensor,  # (n, 3)
        thumb_forces: torch.Tensor,  # (n, a)
        finger_forces: torch.Tensor,  # (n, b)
        hand_joint_pos: torch.Tensor,  # (n, h)
        finite: torch.Tensor,  # (n,)
    ) -> None:
        r = self.r
        hand_distance = torch.linalg.norm(palm_pos - apple_pos, dim=-1)
        contact = (thumb_forces.max(-1).values >= r.contact_threshold) & (finger_forces.max(-1).values >= r.contact_threshold)
        self.contact_steps = torch.where(contact, self.contact_steps + 1, torch.zeros_like(self.contact_steps))
        grasp = self.contact_steps >= r.sustained_contact_steps
        self.contact_age = torch.where(grasp, torch.zeros_like(self.contact_age), self.contact_age + 1)
        recent = self.contact_age <= r.recent_contact_steps
        self.evidence_failure |= detached & ~self.detached_seen & ~recent
        self.detached_seen |= detached
        lifted = detached & (apple_pos[:, 2] > apple_rest_height + r.lift_height) & (hand_distance < r.lift_hand_distance)
        self.lift_steps = torch.where(lifted & recent, self.lift_steps + 1, torch.zeros_like(self.lift_steps))
        self.carried |= (self.lift_steps >= r.carry_steps) & ~self.evidence_failure

        opened = hand_joint_pos.abs().max(-1).values < r.hand_open_threshold
        self.released = self.carried & opened & ~contact & (hand_distance > r.release_hand_distance)
        supported = basket_normal_force > apple_mass * r.gravity * r.support_fraction
        b = apple_in_basket
        settled = (
            torch.isfinite(b).all(-1)
            & (b[:, 0].abs() < r.basket_half_inner[0] - apple_half_height)
            & (b[:, 1].abs() < r.basket_half_inner[1] - apple_half_height)
            & ((b[:, 2] - apple_half_height).abs() < r.settle_height_tolerance)
            & (apple_speed < r.settle_speed)
        )
        placed = detached & supported & settled
        done = finite & placed & self.carried & self.released
        self.hold_steps = torch.where(done, self.hold_steps + 1, torch.zeros_like(self.hold_steps))

        failure = torch.where(self.evidence_failure, torch.full_like(self.failure, DETACHED_WITHOUT_GRASP), self.failure)
        failure = torch.where(apple_pos[:, 2] - ground_height < r.drop_height, torch.full_like(failure, APPLE_DROPPED), failure)
        self.failure = torch.where(~finite, torch.full_like(failure, NON_FINITE), failure)
        self.success = (self.hold_steps >= r.success_hold_steps) & (self.failure == NO_FAILURE)

        self.opposing_contact, self.grasp_contact, self.opened = contact, grasp, opened
        self.supported, self.settled, self.placed, self.hand_distance = supported, settled, placed, hand_distance
