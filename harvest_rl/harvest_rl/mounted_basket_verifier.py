"""Independent mounted-basket telemetry verifier; standard library only.

Contract v1: SI units, a common right-handed world frame W, active xyzw
orientations mapping body coordinates into W. Basket frame B has its origin at
the *inner floor surface centre*, z toward the opening. Basket linear velocity
is at that origin (not at the mount/COM). Fruit pose/velocity are at its native
body-aligned bounding-box centre, and extents use that frame. Readers must apply
any offset from the rigid-body origin, including angular point velocity.
Contact force is on body_a, exerted by body_b, expressed in W; pairs are summed
once per frame. Contact IDs may be calibrated collider paths: a compound
basket's floor must be distinguished from walls even though they share a rigid
body. An unclassified basket-root force cannot be relabelled floor support.
Readers must supply real calibrated identities/forces, complete
contacts and measured hand separation. Nothing here moves bodies, manufactures
forces, certifies calibration, or creates a named task-success signal.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

SCHEMA = "mounted_basket_telemetry_v1"


def _vec(v, n=3):
    if (len(v) != n or any(isinstance(x, bool) or not isinstance(x, (int, float))
                          or not math.isfinite(x) for x in v)):
        raise ValueError(f"Expected finite {n}-vector")
    return tuple(v)


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _norm(v):
    return math.sqrt(_dot(v, v))


def _quat(q):
    q = _vec(q, 4)
    if abs(_dot(q, q)-1) > 1e-6:
        raise ValueError("Require unit xyzw quaternion")
    return q


def rotate(q, v, *, inverse=False):
    """Quaternion vector rotation without torch/Isaac; q and -q are equivalent."""
    q, v = _quat(q), _vec(v)
    u, w = q[:3], q[3]
    if inverse:
        u = tuple(-x for x in u)
    uv = _cross(u, v)
    uuv = _cross(u, uv)
    return tuple(v[i]+2*(w*uv[i]+uuv[i]) for i in range(3))


@dataclass(frozen=True)
class BasketGeometry:
    basket_id: str
    floor_contact_id: str
    wall_contact_ids: tuple[str, ...]
    inner_size_m: tuple[float, float, float] = (.32, .28, .12)

    def __post_init__(self):
        ids = (self.floor_contact_id, *self.wall_contact_ids)
        if (not self.basket_id or any(not isinstance(x, str) or not x for x in ids)
                or len(ids) != len(set(ids)) or len(self.wall_contact_ids) != 4
                or any(x <= 0 for x in _vec(self.inner_size_m))):
            raise ValueError("Require unique physical floor/four-wall IDs and positive inner dimensions")


@dataclass(frozen=True)
class FruitGeometry:
    target_id: str
    body_id: str
    half_extents_m: tuple[float, float, float]
    mass_kg: float

    def __post_init__(self):
        if (not self.target_id or not self.body_id or any(x <= 0 for x in _vec(self.half_extents_m))
                or not math.isfinite(self.mass_kg) or self.mass_kg <= 0):
            raise ValueError("Invalid physical fruit identity/extents/mass")


@dataclass(frozen=True)
class PoseTwist:
    time_s: float
    position_w_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    linear_velocity_w_m_s: tuple[float, float, float] = (0., 0., 0.)
    angular_velocity_w_rad_s: tuple[float, float, float] = (0., 0., 0.)


@dataclass(frozen=True)
class FruitObservation:
    target_id: str
    body_id: str
    motion: PoseTwist


@dataclass(frozen=True)
class ReleaseObservation:
    time_s: float
    target_id: str
    body_id: str
    hand_open: bool
    held: bool
    minimum_hand_separation_m: float


@dataclass(frozen=True)
class ContactObservation:
    time_s: float
    body_a: str
    body_b: str
    force_on_a_w_n: tuple[float, float, float]


@dataclass(frozen=True)
class BasketFrame:
    sample: int
    time_s: float
    basket_id: str
    basket: PoseTwist
    fruits: tuple[FruitObservation, ...]
    releases: tuple[ReleaseObservation, ...]
    contacts: tuple[ContactObservation, ...]
    calibrated_contacts: bool
    calibration_reference: str
    contacts_complete: bool
    # Current external inventory receipt, not permission to infer task success.
    deposited_target_ids: tuple[str, ...] = ()
    schema: str = SCHEMA


@dataclass(frozen=True)
class VerificationPolicy:
    """Proposed new-task telemetry limits; not calibrated hardware/task criteria."""
    min_settle_samples: int = 30
    min_settle_duration_s: float = 1.
    max_sample_gap_s: float = .05
    max_age_s: float = .05
    max_relative_speed_m_s: float = .03
    max_relative_angular_speed_rad_s: float = .3
    containment_margin_m: float = .002
    floor_penetration_tolerance_m: float = .002
    support_gap_tolerance_m: float = .004
    min_hand_separation_m: float = .01
    hand_contact_threshold_n: float = .01
    min_support_load_fraction: float = .5
    gravity_w_m_s2: tuple[float, float, float] = (0., 0., -9.81)

    def __post_init__(self):
        if type(self.min_settle_samples) is not int or self.min_settle_samples < 2:
            raise ValueError("Require multiple consecutive settle samples")
        positive = (self.min_settle_duration_s, self.max_sample_gap_s, self.max_age_s,
                    self.max_relative_speed_m_s, self.max_relative_angular_speed_rad_s,
                    self.min_hand_separation_m, self.hand_contact_threshold_n,
                    self.min_support_load_fraction)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Invalid positive verification limit")
        if (self.min_support_load_fraction > 1 or any(not math.isfinite(v) or v < 0 for v in
                (self.containment_margin_m, self.floor_penetration_tolerance_m, self.support_gap_tolerance_m))
                or _norm(_vec(self.gravity_w_m_s2)) == 0):
            raise ValueError("Invalid margin/support fraction/gravity")


@dataclass(frozen=True)
class FruitVerification:
    target_id: str
    inside_basket: bool
    released: bool
    support_chain_verified: bool
    position_b_m: tuple[float, float, float]
    relative_velocity_b_m_s: tuple[float, float, float]
    relative_angular_velocity_b_rad_s: tuple[float, float, float]
    settled_samples: int
    settled_duration_s: float
    physical_deposit_verified: bool
    newly_verified_deposit: bool
    spilled: bool
    retention_verified: bool
    support_paths: tuple[tuple[str, ...], ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class BasketVerification:
    sample: int
    time_s: float
    basket_id: str
    valid: bool
    errors: tuple[str, ...]
    fruits: tuple[FruitVerification, ...]
    calibration_reference: str
    schema: str = "mounted_basket_verification_v1"

    @property
    def retained_ids(self):
        return frozenset(f.target_id for f in self.fruits if self.valid and f.retention_verified)


class MountedBasketVerifier:
    """Stateful per-fruit settling/spill monitor. No persistence/auto-resume claim.

    Every registered fruit must have one pose and release observation per frame.
    Invalid evidence breaks settling and provides no retention result; it does
    not invent a physical spill. A measured escape after a verified deposit is
    latched permanently, including if the same fruit later reenters. A positive
    physical-deposit event is emitted at most once per registered body/target.
    Missing/invalid intervals after deposit latch retention as unknown until
    review; later good data cannot retrospectively prove continuous retention.
    """
    def __init__(self, basket, fruits, *, hand_body_ids, other_body_ids=(), policy=None):
        self.basket, self.policy = basket, policy or VerificationPolicy()
        fruits = tuple(fruits)
        if not fruits or len({f.target_id for f in fruits}) != len(fruits):
            raise ValueError("Require unique target inventory")
        self.fruits = {f.target_id: f for f in fruits}
        self.by_body = {f.body_id: f.target_id for f in fruits}
        ids = [basket.floor_contact_id, *basket.wall_contact_ids, *hand_body_ids, *other_body_ids,
               *(f.body_id for f in fruits)]
        if (not hand_body_ids or any(not isinstance(x, str) or not x for x in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Physical bodies/hand identities must be distinct")
        self.hand_ids, self.body_ids = frozenset(hand_body_ids), frozenset(ids)
        self._counts = dict.fromkeys(self.fruits, 0)
        self._start = dict.fromkeys(self.fruits, None)
        self._ever_verified, self._spilled, self._retention_unknown = set(), set(), set()
        self._last_sample, self._last_time = None, None
        self._broken = False

    def _time(self, t, frame, now_s):
        if (not math.isfinite(t) or abs(t-frame.time_s) > 1e-9
                or t > now_s+1e-9 or now_s-t > self.policy.max_age_s+1e-9):
            raise ValueError("Stale, future or unsynchronized telemetry")

    def _validate(self, frame, now_s):
        if (frame.schema != SCHEMA or frame.basket_id != self.basket.basket_id
                or type(frame.sample) is not int or frame.sample < 0
                or not math.isfinite(now_s) or not math.isfinite(frame.time_s)):
            raise ValueError("Wrong telemetry version/identity/sample/time")
        self._time(frame.time_s, frame, now_s)
        if self._last_sample is not None and (frame.sample <= self._last_sample or frame.time_s <= self._last_time):
            raise ValueError("Replayed or nonmonotonic sample")
        if (frame.calibrated_contacts is not True or frame.contacts_complete is not True
                or not isinstance(frame.calibration_reference, str) or not frame.calibration_reference.strip()):
            raise ValueError("Missing complete calibrated contact evidence")
        poses = [frame.basket] + [f.motion for f in frame.fruits]
        for p in poses:
            self._time(p.time_s, frame, now_s)
            _vec(p.position_w_m); _quat(p.orientation_xyzw)
            _vec(p.linear_velocity_w_m_s); _vec(p.angular_velocity_w_rad_s)
        ids = [f.target_id for f in frame.fruits]
        released = [f.target_id for f in frame.releases]
        if (len(ids) != len(self.fruits) or set(ids) != set(self.fruits)
                or len(released) != len(ids) or set(released) != set(ids)):
            raise ValueError("Missing/duplicate fruit or release observations")
        for f in (*frame.fruits, *frame.releases):
            if f.body_id != self.fruits[f.target_id].body_id:
                raise ValueError("Fruit/body identity mismatch")
        for release in frame.releases:
            self._time(release.time_s, frame, now_s)
            if (type(release.hand_open) is not bool or type(release.held) is not bool
                    or not math.isfinite(release.minimum_hand_separation_m)
                    or release.minimum_hand_separation_m < 0):
                raise ValueError("Invalid release/separation evidence")
        pairs = set()
        for c in frame.contacts:
            self._time(c.time_s, frame, now_s)
            _vec(c.force_on_a_w_n)
            pair = frozenset((c.body_a, c.body_b))
            if (len(pair) != 2 or not pair <= self.body_ids or pair in pairs):
                raise ValueError("Unknown/self/duplicate contact pair (aggregate each pair once)")
            pairs.add(pair)
        if (len(set(frame.deposited_target_ids)) != len(frame.deposited_target_ids)
                or not set(frame.deposited_target_ids) <= self._ever_verified):
            raise ValueError("Inventory receipt predates independent physical verification")

    def observe(self, frame: BasketFrame, *, now_s: float) -> BasketVerification:
        """now_s is measured collection/simulation time in the same clock domain."""
        try:
            self._validate(frame, now_s)
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
            self._counts = dict.fromkeys(self.fruits, 0)
            self._start = dict.fromkeys(self.fruits, None)
            self._broken = True
            self._retention_unknown.update(self._ever_verified)
            return BasketVerification(frame.sample, frame.time_s, self.basket.basket_id,
                                      False, (str(exc),), (), "")
        p = self.policy
        gap = (self._broken or (self._last_sample is not None and
               (frame.sample != self._last_sample+1 or frame.time_s-self._last_time > p.max_sample_gap_s+1e-9)))
        self._last_sample, self._last_time = frame.sample, frame.time_s
        self._broken = False
        if gap:
            self._retention_unknown.update(self._ever_verified)
            self._counts = dict.fromkeys(self.fruits, 0)
            self._start = dict.fromkeys(self.fruits, None)
        observations = {f.target_id: f for f in frame.fruits}
        releases = {f.target_id: f for f in frame.releases}
        basket = frame.basket
        up_b_w = rotate(basket.orientation_xyzw, (0., 0., 1.))
        g = _norm(p.gravity_w_m_s2)
        up_w = tuple(-x/g for x in p.gravity_w_m_s2)
        contacts = {}
        for c in frame.contacts:
            contacts[(c.body_a, c.body_b)] = c.force_on_a_w_n
            contacts[(c.body_b, c.body_a)] = tuple(-x for x in c.force_on_a_w_n)
        local = {}
        for target, spec in self.fruits.items():
            motion, release = observations[target].motion, releases[target]
            offset = _sub(motion.position_w_m, basket.position_w_m)
            position = rotate(basket.orientation_xyzw, offset, inverse=True)
            # d(R_BW r)/dt includes the basket angular contribution.
            relative = rotate(basket.orientation_xyzw, _sub(_sub(motion.linear_velocity_w_m_s,
                basket.linear_velocity_w_m_s), _cross(basket.angular_velocity_w_rad_s, offset)), inverse=True)
            angular = rotate(basket.orientation_xyzw, _sub(motion.angular_velocity_w_rad_s,
                                                         basket.angular_velocity_w_rad_s), inverse=True)
            axes = [rotate(basket.orientation_xyzw, rotate(motion.orientation_xyzw, axis), inverse=True)
                    for axis in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))]
            extent = tuple(sum(abs(axes[j][i])*spec.half_extents_m[j] for j in range(3)) for i in range(3))
            size = self.basket.inner_size_m
            inside = (all(abs(position[i])+extent[i] <= size[i]/2-p.containment_margin_m+1e-12 for i in (0, 1))
                      and position[2]-extent[2] >= -p.floor_penetration_tolerance_m-1e-12
                      and position[2]+extent[2] <= size[2]-p.containment_margin_m+1e-12)
            hand_contact = any(_norm(contacts.get((spec.body_id, body), (0., 0., 0.))) >
                               p.hand_contact_threshold_n for body in self.hand_ids)
            # Opening is required for the original deposit. Once independently
            # deposited, closing the hand around the NEXT apple does not undo
            # that release; current contact/separation/support still govern
            # retention, and a spill or missing interval remains latched.
            released = ((release.hand_open or target in self._ever_verified)
                        and not release.held and not hand_contact
                        and release.minimum_hand_separation_m >= p.min_hand_separation_m)
            quiet = (_norm(relative) <= p.max_relative_speed_m_s
                     and _norm(angular) <= p.max_relative_angular_speed_rad_s)
            local[target] = dict(position=position, relative=relative, angular=angular,
                                 extent=extent, inside=inside, released=released, quiet=quiet)
            if target in self._ever_verified and not inside:
                self._spilled.add(target)

        def support(target, visiting):
            state, spec = local[target], self.fruits[target]
            if (target in visiting or target in self._spilled or not state['inside']
                    or not state['released'] or not state['quiet']):
                return False, ()
            load, paths = 0., []
            for (a, b), force in contacts.items():
                if a != spec.body_id or _dot(force, up_b_w) <= 0 or _dot(force, up_w) <= 0:
                    continue
                if b == self.basket.floor_contact_id:
                    if state['position'][2]-state['extent'][2] > p.support_gap_tolerance_m:
                        continue
                    child_paths = ((b,),)
                elif b in self.by_body:
                    below = self.by_body[b]
                    if (below not in frame.deposited_target_ids
                            or local[below]['position'][2] >= state['position'][2]-1e-9):
                        continue
                    lower = local[below]
                    gap_z = state['position'][2]-state['extent'][2]-(lower['position'][2]+lower['extent'][2])
                    if (abs(gap_z) > p.support_gap_tolerance_m
                            or any(abs(state['position'][i]-lower['position'][i]) >
                                   state['extent'][i]+lower['extent'][i]+p.support_gap_tolerance_m for i in (0, 1))):
                        continue
                    accepted, child_paths = support(below, visiting | {target})
                    if not accepted:
                        continue
                else:
                    continue  # Rim/wall/hand/branch contact is not a floor-rooted load path.
                load += _dot(force, up_w)
                paths.extend((spec.body_id, *path) for path in child_paths)
            return load+1e-12 >= p.min_support_load_fraction*spec.mass_kg*g, tuple(paths)

        results = []
        for target in self.fruits:
            state = local[target]
            supported, paths = support(target, set())
            eligible = (state['inside'] and state['released'] and state['quiet'] and supported
                        and target not in self._spilled and target not in self._retention_unknown)
            if eligible:
                if self._counts[target] == 0:
                    self._start[target] = frame.time_s
                self._counts[target] += 1
            else:
                self._counts[target], self._start[target] = 0, None
            duration = 0. if self._start[target] is None else frame.time_s-self._start[target]
            settled = (eligible and self._counts[target] >= p.min_settle_samples
                       and duration+1e-9 >= p.min_settle_duration_s)
            new = settled and target not in self._ever_verified
            if new:
                self._ever_verified.add(target)
            reasons = [name for name, ok in (('outside_or_rim', state['inside']),
                       ('held_or_not_released', state['released']), ('relative_motion', state['quiet']),
                       ('no_floor_support_chain', supported), ('spill_latched', target not in self._spilled),
                       ('unobserved_retention_interval', target not in self._retention_unknown)) if not ok]
            if eligible and not settled:
                reasons.append('settling')
            results.append(FruitVerification(target, state['inside'], state['released'], supported,
                state['position'], state['relative'], state['angular'], self._counts[target], duration,
                settled, new, target in self._spilled,
                target in self._ever_verified and eligible, paths, tuple(reasons)))
        return BasketVerification(frame.sample, frame.time_s, self.basket.basket_id, True, (),
                                  tuple(results), frame.calibration_reference)


def to_physical_evidence(task_evidence, result: BasketVerification):
    """Narrow pure adapter; task owns detachment/grasp/assistance/named success.

    Sample IDs must match the same physical control step. A future concrete
    mounted-basket task must provide its own named termination. This function
    never promotes physical settling into named task success or deposit credit.
    Later walking retention goes through result.retained_ids, not observe() on
    an inactive HarvestCycle target. Invalid evidence makes contacts untrusted.
    """
    from .harvest_cycle import PhysicalEvidence
    if not isinstance(task_evidence, PhysicalEvidence) or task_evidence.sample != result.sample:
        raise ValueError("Require same-step PhysicalEvidence")
    fruit = next((f for f in result.fruits if f.target_id == task_evidence.target_id), None)
    if result.valid and fruit is None:
        raise ValueError("Target missing from mounted-basket verification")
    return replace(task_evidence,
        calibrated_contacts=task_evidence.calibrated_contacts and result.valid,
        released=task_evidence.released and bool(fruit and fruit.released),
        inside_basket=bool(result.valid and fruit and fruit.inside_basket),
        support_chain_verified=bool(result.valid and fruit and fruit.support_chain_verified),
        settled_relative_samples=fruit.settled_samples if result.valid and fruit and fruit.physical_deposit_verified else 0,
        dropped=task_evidence.dropped or bool(fruit and fruit.spilled))
