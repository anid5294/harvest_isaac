"""Thin, opt-in native contact-report capture; pure buffer, lazy PhysX binding.

Raw impulses are SI N*s in the engine's reported world convention. Their sign
with respect to actor0 is NOT assumed here: an independently calibrated reader
must bind that convention before converting impulses into forces. Actual
collider0/collider1 identities are retained, including compound-basket surfaces.
No physics launch, body motion, force injection or task-success claim occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Require nonnegative integer {name}")
    return value


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Require finite {name}")
    return float(value)


def _vector(value, name):
    if len(value) != 3:
        raise ValueError(f"Require three-vector {name}")
    return tuple(_finite(x, name) for x in value)


def _path(value):
    if (not isinstance(value, str) or not value.startswith('/') or value.endswith('/') or '//' in value
            or any(x in value for x in '*?[]{}') or any(x in ('.', '..') for x in value.split('/'))):
        raise ValueError("Require exact physical actor/collider path")
    return value


@dataclass(frozen=True)
class RawContact:
    physics_step: int
    actor0: str
    actor1: str
    collider0: str
    collider1: str
    impulse_w_ns: tuple[float, float, float]
    position_w_m: tuple[float, float, float]
    separation_m: float

    def __post_init__(self):
        _integer(self.physics_step, 'physics_step')
        for p in (self.actor0, self.actor1, self.collider0, self.collider1):
            _path(p)
        if self.actor0 == self.actor1 or self.collider0 == self.collider1:
            raise ValueError("Self-paired contact identity")
        object.__setattr__(self, 'impulse_w_ns', _vector(self.impulse_w_ns, 'impulse'))
        object.__setattr__(self, 'position_w_m', _vector(self.position_w_m, 'position'))
        object.__setattr__(self, 'separation_m', _finite(self.separation_m, 'separation'))


@dataclass(frozen=True)
class ContactWindow:
    sample: int
    time_s: float
    duration_s: float
    physics_steps: int
    contacts: tuple[RawContact, ...]
    complete: bool = True


class ContactReportBuffer:
    """One exact completed control interval; no stale force/contact carryover.

    start_step is the native counter BEFORE the first step. Event steps belong
    to (start_step, start_step+decimation]. Empty intervals are valid capture
    windows, not proof the sensor works; independent positive-response admission
    remains mandatory. Any malformed/dropped/out-of-order interval stops this
    buffer permanently rather than allowing a caller to reset the budget/state.
    """
    def __init__(self, physics_dt, decimation):
        self.physics_dt = _finite(physics_dt, 'physics_dt')
        self.decimation = _integer(decimation, 'decimation')
        if self.physics_dt <= 0 or self.decimation == 0:
            raise ValueError("Positive physics_dt/decimation required")
        self._lock = threading.RLock()
        self._active = None
        self._contacts = []
        self._last_sample = self._last_end_step = self._last_end_time = None
        self._last_event_step = None
        self.errors = []
        self.capture_started = False

    @property
    def active(self):
        with self._lock:
            return self._active is not None

    def fail(self, message):
        with self._lock:
            self.errors.append(str(message))

    def _raise_if_failed(self):
        if self.errors:
            raise RuntimeError("Contact report capture stopped: " + self.errors[0])

    def _abort(self, message):
        self.fail(message)
        raise ValueError(message)

    def begin_interval(self, sample, start_step, start_time_s):
        with self._lock:
            self._raise_if_failed()
            try:
                _integer(sample, 'sample'); _integer(start_step, 'start_step')
                start_time_s = _finite(start_time_s, 'start_time_s')
            except (TypeError, ValueError) as exc:
                self._abort(str(exc))
            if self._active is not None:
                self._abort("Overlapping capture interval")
            if start_time_s < 0:
                self._abort("Negative capture clock")
            if self._last_sample is not None and (
                    sample != self._last_sample+1 or start_step != self._last_end_step
                    or abs(start_time_s-self._last_end_time) > 1e-9):
                self._abort("Nonconsecutive sample/native counter/clock")
            self._active = (sample, start_step, start_time_s)
            self.capture_started = True
            self._contacts = []
            self._last_event_step = None

    def add_contacts(self, physics_step, contacts):
        with self._lock:
            self._raise_if_failed()
            if self._active is None:
                self._abort("Contact added outside active capture interval")
            _, start, _ = self._active
            if (type(physics_step) is not int or not start < physics_step <= start+self.decimation
                    or (self._last_event_step is not None and physics_step < self._last_event_step)):
                self._abort("Stale/out-of-range native contact step")
            try:
                contacts = tuple(contacts)
            except TypeError:
                self._abort("Malformed raw contact collection")
            if any(not isinstance(c, RawContact) or c.physics_step != physics_step for c in contacts):
                self._abort("Mismatched raw contact step/type")
            self._contacts.extend(contacts)
            self._last_event_step = physics_step

    def finish_interval(self, end_step, end_time_s):
        with self._lock:
            self._raise_if_failed()
            if self._active is None:
                self._abort("No active capture interval")
            sample, start, time0 = self._active
            duration = self.physics_dt*self.decimation
            try:
                _integer(end_step, 'end_step'); end_time_s = _finite(end_time_s, 'end_time_s')
            except (TypeError, ValueError) as exc:
                self._abort(str(exc))
            if end_step != start+self.decimation or abs(end_time_s-time0-duration) > 1e-9:
                self._abort("Incomplete native physics-step/time window")
            result = ContactWindow(sample, end_time_s, duration, self.decimation, tuple(self._contacts))
            self._last_sample, self._last_end_step, self._last_end_time = sample, end_step, end_time_s
            self._active, self._contacts = None, []
            return result


class NativePhysxReportSource:
    """Lazy ContactReportAPI subscriber; requires an existing approved USD stage.

    ContactHeader actor/collider IDs and data offsets follow NVIDIA's native
    contact-report API. The installed SDK has NOT been executed by these tests.
    Enable reporting only on explicit existing physical actors, with threshold0;
    this is instrumentation, not a collision/material/force modification.
    """
    def __init__(self, buffer, stage_id, step_index):
        if not isinstance(buffer, ContactReportBuffer) or not callable(step_index):
            raise ValueError("Require buffer and measured native-step reader")
        self.buffer = buffer
        self.stage_id = _integer(stage_id, 'stage_id')
        self.step_index = step_index
        self._subscription = None
        self._actors = frozenset()
        self._attached = self._closed = False
        self.ignored_outside_windows = self.ignored_unmonitored = 0

    def attach(self, stage, actor_paths):
        if self._attached or self._closed:
            raise RuntimeError("Contact source cannot attach twice/retry after close")
        actors = tuple(actor_paths)
        if not actors or len(set(actors)) != len(actors):
            raise ValueError("Require unique nonempty monitored actor paths")
        for p in actors: _path(p)
        from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        prims = [stage.GetPrimAtPath(p) for p in actors]
        if any(not p.IsValid() or not p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims):
            raise ValueError("All report actors must be existing native rigid bodies")
        for prim in prims:
            if not UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get():
                raise ValueError("Disabled contact-report rigid body")
        self._decode = PhysicsSchemaTools.intToSdfPath
        self._found, self._persists, self._lost = (ContactEventType.CONTACT_FOUND,
            ContactEventType.CONTACT_PERSIST, ContactEventType.CONTACT_LOST)
        self._actors = frozenset(actors)
        try:
            for prim in prims:
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
            self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._on_report)
            if self._subscription is None:
                raise RuntimeError("Native contact subscription missing")
            self._attached = True
        except Exception as exc:
            self.buffer.fail(f"Native contact attach failed: {type(exc).__name__}: {exc}")
            self._closed = True
            raise

    def _on_report(self, headers, data):
        # Exceptions thrown through native callbacks can disappear into Kit logs;
        # latch them in the buffer so finish_interval must reject the whole run.
        if self._closed:
            return
        if not self.buffer.active:
            if self.buffer.capture_started:
                self.buffer.fail("Late native contact callback outside a completed capture interval")
            else:
                self.ignored_outside_windows += 1
            return
        try:
            step = _integer(self.step_index(), 'native callback step')
            rows, used_indices = [], set()
            for header in headers:
                a0, a1 = str(self._decode(header.actor0)), str(self._decode(header.actor1))
                if not ({a0, a1} & self._actors):
                    self.ignored_unmonitored += 1
                    continue
                if _integer(header.stage_id, 'native stage_id') != self.stage_id:
                    raise ValueError("Monitored contact came from a different stage")
                offset, count = header.contact_data_offset, header.num_contact_data
                _integer(offset, 'contact_data_offset'); _integer(count, 'num_contact_data')
                if offset+count > len(data):
                    raise ValueError("Native contact data range is truncated")
                if header.type == self._lost:
                    if count:
                        raise ValueError("CONTACT_LOST unexpectedly contains impulses")
                    continue
                if header.type not in (self._found, self._persists):
                    raise ValueError("Unknown native contact event type")
                c0, c1 = str(self._decode(header.collider0)), str(self._decode(header.collider1))
                for i in range(offset, offset+count):
                    if i in used_indices:
                        raise ValueError("Native contact data ranges overlap; refusing double force credit")
                    used_indices.add(i)
                    item = data[i]
                    # Copy native buffers while valid for this physics callback.
                    rows.append(RawContact(step, a0, a1, c0, c1,
                        _vector(item.impulse, 'native impulse'), _vector(item.position, 'native position'),
                        _finite(item.separation, 'native separation')))
            self.buffer.add_contacts(step, rows)
        except Exception as exc:
            self.buffer.fail(f"Native contact callback failed: {type(exc).__name__}: {exc}")

    def close(self):
        self._closed = True
        self.step_index = None  # Release the callback closure's environment reference.
        subscription, self._subscription = self._subscription, None
        if subscription is not None and callable(getattr(subscription, 'unsubscribe', None)):
            subscription.unsubscribe()
        # Native carb subscription also unsubscribes on release when the SDK
        # does not expose an explicit unsubscribe method. No external copy held.
