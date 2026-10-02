"""Opt-in, bounded native PhysX observations for an orchard attempt.

Times are callback-observation times on a counted physics-step clock. They are
not a reconstruction of the solver instant at which a constraint broke.
"""

from collections import deque


class PhysicsDiagnostics:
    def __init__(self, capacity=1200):
        self.steps = deque(maxlen=capacity)
        self.events = deque(maxlen=capacity)
        self.step_index = 0
        self.observed_time_s = 0.0
        self._pending_contacts = []
        self._pending_breaks = []
        self.callback_error = None

    def on_contact(self, *, actor0, actor1, collider0, collider1, event_type,
                   impulses_ns):
        record = dict(actor0=actor0, actor1=actor1, collider0=collider0,
                      collider1=collider1, event_type=event_type,
                      impulses_ns=[list(v) for v in impulses_ns])
        self._pending_contacts.append(record)
        self.events.append(dict(kind="contact", observed_step=self.step_index,
                                observed_time_s=self.observed_time_s, **record))

    def on_break(self, joint_path):
        record = dict(joint_path=joint_path)
        self._pending_breaks.append(record)
        self.events.append(dict(kind="joint_break", observed_step=self.step_index,
                                observed_time_s=self.observed_time_s, **record))

    def on_step(self, dt):
        self.step_index += 1
        self.observed_time_s += float(dt)
        self.steps.append(dict(step=self.step_index, time_s=self.observed_time_s,
                               dt_s=float(dt), contacts=self._pending_contacts,
                               joint_breaks=self._pending_breaks))
        self._pending_contacts = []
        self._pending_breaks = []

    def snapshot(self):
        if self.callback_error is not None:
            raise RuntimeError(f"CPU orchard diagnostics callback failed: {self.callback_error}")
        return dict(timestamp_basis="callback_observation_physics_step_count",
                    step_count=self.step_index, observed_time_s=self.observed_time_s,
                    steps=list(self.steps), events=list(self.events),
                    pending_contacts=list(self._pending_contacts),
                    pending_joint_breaks=list(self._pending_breaks))


def install(env):
    """Subscribe only when explicitly enabled; raise if native APIs are absent."""
    import omni.physx
    import omni.usd
    from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = env.scene.env_prim_paths[0]
    robot = stage.GetPrimAtPath(f"{root}/Robot")
    if not robot.IsValid():
        raise RuntimeError("CPU orchard diagnostics: robot prim is missing")
    paths = {str(prim.GetPath()) for prim in Usd.PrimRange(robot)
             if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    for index in range(len(env.cfg.orchard_layout.apples)):
        for name in (f"Apple_{index:02d}", f"Stem_{index}"):
            path = f"{root}/{name}"
            if not stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"CPU orchard diagnostics: expected prim missing: {path}")
            paths.add(path)
    if not paths:
        raise RuntimeError("CPU orchard diagnostics: no reportable prims found")
    for path in sorted(paths):
        api = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path))
        if not api:
            raise RuntimeError(f"CPU orchard diagnostics: contact API failed on {path}")
        api.CreatePhysxContactReportThresholdAttr().Set(0.0)

    buffer = PhysicsDiagnostics()

    def on_contacts(headers, data):
        try:
            for header in headers:
                start = header.contact_data_offset
                end = start + header.num_contact_data
                buffer.on_contact(
                    actor0=str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    actor1=str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                    collider0=str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    collider1=str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                    event_type=str(header.type),
                    impulses_ns=[tuple(float(x) for x in data[i].impulse)
                                 for i in range(start, end)],
                )
        except Exception as exc:
            buffer.callback_error = repr(exc)
            raise

    try:
        contact_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
        step_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(buffer.on_step)
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise RuntimeError("CPU orchard diagnostics require native PhysX contact and physics-step subscriptions") from exc
    if contact_sub is None or step_sub is None:
        raise RuntimeError("CPU orchard diagnostics: native PhysX subscription returned no holder")
    env._orchard_physics_diagnostics = buffer
    env._orchard_contact_report_subscription = contact_sub
    env._orchard_physics_step_subscription = step_sub
