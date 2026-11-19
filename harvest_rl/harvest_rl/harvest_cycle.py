"""Simulator-free contracts for future repeated harvesting; not a controller.

No robot/object writes, joint commands, reward or success overrides. The physical
adapter must supply independently verified evidence. Legacy HybridHarvestPolicy
continues to own the complete left-arm/hand skill, including learned reaching.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math


class Phase(str, Enum):
    SELECT = "select"
    MANIPULATE = "manipulate"
    VERIFY = "verify"
    STOW = "stow"
    NAVIGATE = "navigate"
    FULL = "basket_full"
    REVIEW = "review_required"


class FruitStatus(str, Enum):
    AVAILABLE = "available"
    ACTIVE = "active"
    DEPOSITED = "deposited"
    FAILED = "failed"
    INACCESSIBLE = "inaccessible"
    LOST = "lost_after_deposit"


@dataclass(frozen=True)
class FruitTarget:
    target_id: str
    tree_id: str
    fruit_id: int
    mass_kg: float
    parent_branch_id: int
    geometry_sha256: str

    def __post_init__(self):
        if (not self.target_id or not self.tree_id or type(self.fruit_id) is not int
                or type(self.parent_branch_id) is not int
                or not math.isfinite(self.mass_kg) or self.mass_kg <= 0
                or len(self.geometry_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.geometry_sha256)):
            raise ValueError("Invalid immutable fruit identity/mass/geometry")


@dataclass(frozen=True)
class BasketProposal:
    """Concept parameters only; mount pose is deliberately unset until assessed."""
    inner_size_m: tuple[float, float, float] = (0.32, 0.28, 0.12)
    wall_thickness_m: float = 0.01
    dry_mass_kg: float = 0.5
    payload_limit_kg: float = 0.48
    parent_link: str = "pelvis"
    mount_position_m: tuple[float, float, float] | None = None
    mount_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    balance_validated: bool = False
    workspace_validated: bool = False
    self_collision_validated: bool = False
    loaded_walking_validated: bool = False

    def __post_init__(self):
        values = (*self.inner_size_m, self.wall_thickness_m,
                  self.dry_mass_kg, self.payload_limit_kg)
        if len(self.inner_size_m) != 3 or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Basket dimensions and masses must be finite and positive")
        if self.mount_position_m is not None and (len(self.mount_position_m) != 3
                or any(not math.isfinite(v) for v in self.mount_position_m)):
            raise ValueError("Mount offset must be a finite vector")
        if (not self.parent_link or len(self.mount_orientation_xyzw) != 4
                or any(not math.isfinite(v) for v in self.mount_orientation_xyzw)
                or abs(sum(v*v for v in self.mount_orientation_xyzw) - 1) > 1e-6):
            raise ValueError("Require named mount link and normalized xyzw orientation")

    @property
    def ready_for_walking(self):
        return self.mount_position_m is not None and all((self.balance_validated,
            self.workspace_validated, self.self_collision_validated, self.loaded_walking_validated))


@dataclass(frozen=True)
class SkillRequest:
    skill: str
    target_id: str | None
    tree_id: str | None
    arm_owner: str
    base_mode: str


@dataclass(frozen=True)
class OrchardLayoutRequest:
    """Requested scale/split, not an asset generator or feasibility certificate."""
    rows: int = 1
    trees_per_row: int = 1
    dynamic_fruit_per_tree: int = 2
    train_geometry_families: tuple[str, ...] = ()
    test_geometry_families: tuple[str, ...] = ()
    camera_roles: tuple[str, ...] = ("head", "left_wrist", "overview")
    cameras_enabled: bool = False

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.rows, self.trees_per_row, self.dynamic_fruit_per_tree)):
            raise ValueError("Requested tree/fruit counts must be positive integers")
        if set(self.train_geometry_families) & set(self.test_geometry_families):
            raise ValueError("Tree morphology families must not cross train/test splits")


@dataclass(frozen=True)
class PhysicalEvidence:
    """Telemetry contract, NOT evidence created by this scaffold.

    support_chain_verified allows support via already deposited fruit only if a
    future contact-graph verifier establishes a load path to the basket. Relative
    settling/containment must be computed in its measured moving frame.
    """
    sample: int
    target_id: str
    calibrated_contacts: bool
    opposing_grasp: bool = False
    physical_detachment_verified: bool = False
    extra_apple_assistance: bool = False
    released: bool = False
    inside_basket: bool = False
    support_chain_verified: bool = False
    settled_relative_samples: int = 0
    named_task_success: bool = False
    dropped: bool = False
    fallen: bool = False
    unsafe_collision: bool = False


def joint_ownership(joint_sets: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """Fail before writes if navigation and manipulation both own a joint."""
    owners = {}
    for skill, joints in joint_sets.items():
        for joint in joints:
            if joint in owners:
                raise ValueError(f"Joint {joint} has competing owners: {owners[joint]}, {skill}")
            owners[joint] = skill
    return owners


def combined_center_of_mass(parts):
    """Static mass/COM calculation for design screening, never a balance proof."""
    masses, moments = [], [0.0, 0.0, 0.0]
    for mass, position in parts:
        if not math.isfinite(mass) or mass <= 0 or len(position) != 3 or any(
                not math.isfinite(x) for x in position):
            raise ValueError("Invalid mass/COM")
        masses.append(mass)
        for i, x in enumerate(position):
            moments[i] += mass * x
    if not masses:
        raise ValueError("At least one physical body required")
    return tuple(x / sum(masses) for x in moments)


class HarvestCycle:
    """One-target-at-a-time event ledger with no automatic manipulation retries.

    Fruit status persists across target switches. Failed/inaccessible apples keep
    a tree incomplete. A restart must restore history in the future runtime
    adapter; this class intentionally does not implement permissive auto-resume.
    """
    def __init__(self, targets, *, basket=BasketProposal(), prior_failures=0):
        targets = tuple(targets)
        if not targets or len({t.target_id for t in targets}) != len(targets):
            raise ValueError("Require unique nonempty target inventory")
        if len({(t.tree_id, t.fruit_id) for t in targets}) != len(targets):
            raise ValueError("Same physical fruit cannot have two target IDs")
        if type(prior_failures) is not int or prior_failures < 0:
            raise ValueError("Invalid historical failure count")
        self.targets = {t.target_id: t for t in targets}
        self.status = {t.target_id: FruitStatus.AVAILABLE for t in targets}
        self.basket = basket
        self.failures = prior_failures
        self.phase = Phase.REVIEW if prior_failures >= 3 else Phase.SELECT
        self.active = None
        self.tree = None
        self.events = []
        self.last_sample = -1
        self.grasp_seen = self.detach_seen = False

    def _event(self, event, **detail):
        self.events.append(dict(sequence=len(self.events), event=event,
                                phase=self.phase.value, target_id=self.active, **detail))

    @property
    def deposited(self):
        return {key for key, status in self.status.items() if status == FruitStatus.DEPOSITED}

    @property
    def payload_kg(self):
        return sum(self.targets[key].mass_kg for key in self.deposited)

    def select(self, target_id, *, admitted, stance_stable):
        if self.phase != Phase.SELECT or target_id not in self.targets:
            raise ValueError("Selection requires ready state and known target")
        if self.status[target_id] != FruitStatus.AVAILABLE or not admitted or not stance_stable:
            raise ValueError("Target must be independently admitted, unattempted and stance stable")
        target = self.targets[target_id]
        if self.tree is not None and self.tree != target.tree_id:
            raise ValueError("Navigate and settle before selecting another tree")
        if self.payload_kg + target.mass_kg > self.basket.payload_limit_kg + 1e-9:
            self.phase = Phase.FULL
            self._event("payload_limit", next_target=target_id)
            return
        self.tree = target.tree_id
        self.active = target_id
        self.status[target_id] = FruitStatus.ACTIVE
        self.grasp_seen = self.detach_seen = False
        self.phase = Phase.MANIPULATE
        self._event("target_selected", reach_mode="ppo_reach_grasp_pull")

    def observe(self, evidence: PhysicalEvidence):
        if self.phase not in (Phase.MANIPULATE, Phase.VERIFY):
            raise ValueError("No active manipulation to observe")
        if evidence.target_id != self.active or evidence.sample <= self.last_sample:
            raise ValueError("Stale or wrong-target evidence")
        if type(evidence.sample) is not int or type(evidence.settled_relative_samples) is not int:
            raise ValueError("Invalid sample counters")
        self.last_sample = evidence.sample
        if (not evidence.calibrated_contacts or evidence.extra_apple_assistance
                or evidence.fallen or evidence.unsafe_collision or evidence.dropped):
            self.fail("invalid_or_unsafe_physical_evidence")
            return
        self.grasp_seen |= evidence.opposing_grasp
        if evidence.physical_detachment_verified:
            if not self.grasp_seen or (not self.detach_seen and not evidence.opposing_grasp):
                self.fail("detached_without_verified_grasp")
                return
            self.detach_seen = True
        if evidence.released:
            self.phase = Phase.VERIFY
        accepted = (self.detach_seen and evidence.released and evidence.inside_basket
                    and evidence.support_chain_verified and evidence.settled_relative_samples >= 30
                    and evidence.named_task_success)
        if evidence.named_task_success and not accepted:
            self.fail("named_success_disagrees_with_mounted_basket_evidence")
        elif accepted:
            self.status[self.active] = FruitStatus.DEPOSITED
            self._event("apple_deposited", evidence=asdict(evidence))
            self.active = None
            self.phase = Phase.STOW

    def fail(self, reason):
        if self.active is None:
            raise ValueError("Failure must refer to an active attempt")
        self.status[self.active] = FruitStatus.FAILED
        self.failures += 1
        self.phase = Phase.REVIEW
        self._event("attempt_failed", reason=reason, cumulative_failures=self.failures)
        # Intentionally no automatic recovery/retry or clearing of counters.

    def exclude(self, target_id, *, evidence_reference):
        if self.phase != Phase.SELECT or self.status[target_id] != FruitStatus.AVAILABLE or not evidence_reference:
            raise ValueError("Only untouched targets may be structurally excluded with evidence")
        self.status[target_id] = FruitStatus.INACCESSIBLE
        self._event("target_inaccessible", excluded_target=target_id, evidence=evidence_reference)

    def stow_complete(self, *, arm_stowed, no_held_apple, retained_ids):
        if self.phase != Phase.STOW or not arm_stowed or not no_held_apple:
            raise ValueError("Must stow and release manipulation before selection/walking")
        if set(retained_ids) != self.deposited:
            self._payload_failure(retained_ids, "payload_retention_failure")
            return
        self.phase = Phase.SELECT
        self._event("stowed")

    def tree_complete(self, tree_id):
        ids = [t.target_id for t in self.targets.values() if t.tree_id == tree_id]
        return bool(ids) and all(self.status[key] == FruitStatus.DEPOSITED for key in ids)

    def navigate(self, next_tree, *, free_base_policy_validated, arm_stowed,
                 no_held_apple, retained_ids):
        if (self.phase != Phase.SELECT or not self.tree_complete(self.tree)
                or next_tree not in {t.tree_id for t in self.targets.values()}
                or next_tree == self.tree or not free_base_policy_validated
                or not self.basket.ready_for_walking or not arm_stowed or not no_held_apple
                or set(retained_ids) != self.deposited):
            raise ValueError("Walking requires completed tree, validated loaded robot, stow and retention")
        self.phase = Phase.NAVIGATE
        self.tree = next_tree
        self._event("navigation_requested", next_tree=next_tree)

    def navigation_sample(self, *, arrived, settled, retained_ids, fallen=False, unsafe_collision=False):
        if self.phase != Phase.NAVIGATE:
            raise ValueError("No active navigation")
        if fallen or unsafe_collision or set(retained_ids) != self.deposited:
            self._payload_failure(retained_ids, "navigation_failure")
        elif arrived and settled:
            self.phase = Phase.SELECT
            self._event("stance_settled")

    def _payload_failure(self, retained_ids, reason):
        missing = self.deposited - set(retained_ids)
        for target_id in missing:
            self.status[target_id] = FruitStatus.LOST
        self.failures += 1
        self.phase = Phase.REVIEW
        self._event(reason, missing_target_ids=sorted(missing), cumulative_failures=self.failures)

    def command(self):
        if self.phase in (Phase.MANIPULATE, Phase.VERIFY):
            return SkillRequest("legacy_hybrid", self.active, self.tree, "legacy_hybrid", "stand")
        if self.phase == Phase.NAVIGATE:
            return SkillRequest("velocity_navigation", None, self.tree, "stow", "walk")
        if self.phase == Phase.STOW:
            return SkillRequest("stow", None, self.tree, "stow", "stand")
        return SkillRequest("hold", self.active, self.tree, "hold", "stand")

    def snapshot(self):
        return dict(schema="harvest_cycle_v1", phase=self.phase.value, active=self.active,
                    tree=self.tree, failures=self.failures, payload_kg=self.payload_kg,
                    fruit_status={k: v.value for k, v in self.status.items()}, events=list(self.events),
                    runtime_validated=False)
