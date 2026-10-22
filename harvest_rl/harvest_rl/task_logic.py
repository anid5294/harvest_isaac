"""harvest_isaac's orchard success/failure logic, vectorized over environments.

A batched port of ``update_metrics`` (orchard_pick/mdp.py), ``HarvestEvidence``
(orchard_pick/interaction.py), ``opposing_contact`` (orchard_pick/contact.py) and
``settled_in_basket`` (orchard_pick/layout.py), for the OrchardBench ("measured") task.
Pure torch, no simulator imports, so it can be tested against the original code
(tests/test_task_logic.py).

Per control step it decides:

* failure ``detached_without_sustained_grasp``: the stem broke while there was no recent
  sustained opposing (thumb vs finger) grip;
* ``carried``: detached, gripped, lifted 6 cm above its tree position with the palm within
  18 cm, for 10 consecutive steps;
* ``released``: carried, then hand open, no finger contact and palm > 18 cm away;
* ``success``: carried + released + basket floor bears >= 40 % of the apple's weight + the
  apple rests inside the walls on the floor, held for 30 consecutive steps;
* failure ``apple_dropped`` (below 0.40 m) and ``non_finite_state``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FAILURE_NAMES = ("none", "apple_dropped", "non_finite_state", "detached_without_sustained_grasp")
NO_FAILURE, APPLE_DROPPED, NON_FINITE, DETACHED_WITHOUT_GRASP = range(4)


@dataclass(frozen=True)
class HarvestParams:
    apple_radius: float
    apple_mass: float
    apple_rest_height: float
    """z of the apple on the tree [m]; lifting is measured from here."""
    basket_center: tuple[float, float, float]
    """Basket floor centre [m]; its top surface is 0.01 m above."""
    basket_half_inner: tuple[float, float] = (0.15, 0.13)
    """Half inner size of the basket in x/y [m] (walls are 2 cm thick)."""
    contact_threshold: float = 0.05
    """Per-link force [N] counted as contact for the opposing-grip test."""
    sustained_contact_steps: int = 5
    recent_contact_steps: int = 2
    lift_height: float = 0.06
    lift_hand_distance: float = 0.18
    carry_steps: int = 10
    release_hand_distance: float = 0.18
    hand_open_threshold: float = 0.30
    """Max |hand joint position| [rad] for the hand to count as open."""
    support_fraction: float = 0.4
    settle_height_tolerance: float = 0.012
    settle_speed: float = 0.035
    success_hold_steps: int = 30
    drop_height: float = 0.40
    gravity: float = 9.81


class HarvestTracker:
    """Batched version of harvest_isaac's per-episode harvest bookkeeping."""

    def __init__(self, num_envs: int, params: HarvestParams, device):
        self.p = params
        n = num_envs
        self.contact_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.contact_age = torch.full((n,), 1_000_000, dtype=torch.long, device=device)
        self.detached_seen = torch.zeros(n, dtype=torch.bool, device=device)
        self.lift_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.carried = torch.zeros(n, dtype=torch.bool, device=device)
        self.released = torch.zeros(n, dtype=torch.bool, device=device)
        self.hold_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.evidence_failure = torch.zeros(n, dtype=torch.bool, device=device)
        self.failure = torch.zeros(n, dtype=torch.long, device=device)
        self.success = torch.zeros(n, dtype=torch.bool, device=device)
        # last-step diagnostics
        self.opposing_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self.grasp_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self.supported = torch.zeros(n, dtype=torch.bool, device=device)
        self.settled = torch.zeros(n, dtype=torch.bool, device=device)
        self.opened = torch.zeros(n, dtype=torch.bool, device=device)
        self.hand_distance = torch.zeros(n, device=device)

    def reset(self, env_ids):
        for buf, value in (
            (self.contact_steps, 0), (self.contact_age, 1_000_000), (self.detached_seen, False),
            (self.lift_steps, 0), (self.carried, False), (self.released, False), (self.hold_steps, 0),
            (self.evidence_failure, False), (self.failure, NO_FAILURE), (self.success, False), (self.opposing_contact, False),
            (self.grasp_contact, False), (self.supported, False), (self.settled, False),
            (self.opened, False), (self.hand_distance, 0.0),
        ):
            buf[env_ids] = value

    def update(
        self,
        *,
        detached: torch.Tensor,  # (n,) bool, stem broken
        apple_pos: torch.Tensor,  # (n, 3) env-local
        apple_speed: torch.Tensor,  # (n,)
        palm_pos: torch.Tensor,  # (n, 3) env-local, grasping palm
        thumb_forces: torch.Tensor,  # (n, k) per thumb link [N]
        finger_forces: torch.Tensor,  # (n, j) per opposing finger link [N]
        basket_normal_force_z: torch.Tensor,  # (n,) basket floor force on apple, world z [N]
        hand_joint_pos: torch.Tensor,  # (n, h) grasping hand joints [rad]
        finite: torch.Tensor,  # (n,) bool, robot + apple state finite
    ) -> None:
        p = self.p
        hand_distance = torch.linalg.norm(palm_pos - apple_pos, dim=-1)
        lifted = detached & (apple_pos[:, 2] > p.apple_rest_height + p.lift_height) & (
            hand_distance < p.lift_hand_distance
        )

        contact = (thumb_forces.max(-1).values >= p.contact_threshold) & (
            finger_forces.max(-1).values >= p.contact_threshold
        )
        self.contact_steps = torch.where(contact, self.contact_steps + 1, torch.zeros_like(self.contact_steps))
        grasp_contact = self.contact_steps >= p.sustained_contact_steps

        # HarvestEvidence.update
        self.contact_age = torch.where(grasp_contact, torch.zeros_like(self.contact_age), self.contact_age + 1)
        recent = self.contact_age <= p.recent_contact_steps
        first_detach = detached & ~self.detached_seen
        self.evidence_failure |= first_detach & ~recent
        self.detached_seen |= detached
        self.lift_steps = torch.where(detached & lifted & recent, self.lift_steps + 1, torch.zeros_like(self.lift_steps))
        self.carried |= (self.lift_steps >= p.carry_steps) & ~self.evidence_failure

        supported = basket_normal_force_z > p.apple_mass * p.gravity * p.support_fraction
        opened = hand_joint_pos.abs().max(-1).values < p.hand_open_threshold
        self.released = self.carried & opened & ~contact & (hand_distance > p.release_hand_distance)

        bx, by, bz = p.basket_center
        r = p.apple_radius
        settled = (
            torch.isfinite(apple_pos).all(-1)
            & torch.isfinite(apple_speed)
            & ((apple_pos[:, 0] - bx).abs() < p.basket_half_inner[0] - r)
            & ((apple_pos[:, 1] - by).abs() < p.basket_half_inner[1] - r)
            & ((apple_pos[:, 2] - (bz + 0.01 + r)).abs() < p.settle_height_tolerance)
            & (apple_speed < p.settle_speed)
        )
        placed = (
            finite & detached & self.carried & self.released & opened
            & (hand_distance > p.release_hand_distance) & supported & settled
        )
        self.hold_steps = torch.where(placed, self.hold_steps + 1, torch.zeros_like(self.hold_steps))

        # Same precedence as update_metrics: the evidence failure is copied in, then a
        # non-finite state or a dropped apple overrides it for this step.
        failure = torch.where(self.evidence_failure, torch.full_like(self.failure, DETACHED_WITHOUT_GRASP), self.failure)
        failure = torch.where(apple_pos[:, 2] < p.drop_height, torch.full_like(failure, APPLE_DROPPED), failure)
        self.failure = torch.where(~finite, torch.full_like(failure, NON_FINITE), failure)
        self.success = (self.hold_steps >= p.success_hold_steps) & (self.failure == NO_FAILURE)

        self.opposing_contact, self.grasp_contact = contact, grasp_contact
        self.supported, self.settled, self.opened, self.hand_distance = supported, settled, opened, hand_distance
