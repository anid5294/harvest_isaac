"""Native collider-report/geometry bindings for the opt-in parked task.

Pure decoding is testable without Kit. Native subscription and USD inspection
are lazy, after an approved scene exists. No force injection, pose following,
task success, sensor calibration or admission is manufactured here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

from .mounted_basket_verifier import (BasketFrame, ContactObservation, FruitObservation,
                                     ReleaseObservation, rotate)
from .mounted_contact_reports import ContactReportBuffer, NativePhysxReportSource


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def vector(value):
    if len(value) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("Require finite measured three-vector")
    return tuple(value)



def native_fruit_records(env, *, metadata_only=False):
    """Stable full physical inventory; legacy singleton is an exact fallback."""
    if callable(getattr(env, 'dense_fruit_inventory', None)):
        records = tuple(env.dense_fruit_inventory())
    else:
        records = (dict(target_id=env.cfg.mounted_target.target_id,
            body_id=env.cfg.scene.apple.prim_path.replace('{ENV_REGEX_NS}', '/World/envs/env_0'),
            object=env._apple, half_extents_m=tuple(env.cfg.fruit.half_extents),
            mass_kg=env.cfg.fruit.mass if hasattr(env.cfg.fruit, 'mass') else None,
            attached=None if metadata_only else bool(env.get_state()['apple_attached'][0])),)
    ids = [r['target_id'] for r in records]
    bodies = [r['body_id'] for r in records]
    if not records or len(set(ids)) != len(ids) or len(set(bodies)) != len(bodies):
        raise ValueError('Missing/duplicate physical fruit inventory identity')
    if any((not metadata_only and type(r['attached']) is not bool) or not r['body_id'].startswith('/') for r in records):
        raise ValueError('Require actual per-fruit attachment and exact native body path')
    return records


def active_fruit_id(env):
    return str(env.active_target_id) if hasattr(env, 'active_target_id') else env.cfg.mounted_target.target_id

def world_bounds(local_min, local_max, position, quat):
    """Conservative collider envelope, driven by actual native rigid poses."""
    lo, hi, p = vector(local_min), vector(local_max), vector(position)
    if any(a >= b for a, b in zip(lo, hi)):
        raise ValueError("Empty native collision bounds")
    points = [tuple(x+y for x, y in zip(rotate(quat, corner), p))
              for corner in itertools.product(*zip(lo, hi))]
    return tuple(min(c[i] for c in points) for i in range(3)), tuple(max(c[i] for c in points) for i in range(3))


def separation_lower_bound(box0, box1):
    """AABB separation <= actual surface separation; overlap proves no clearance."""
    a, b = box0, box1
    gap = [max(0., a[0][i]-b[1][i], b[0][i]-a[1][i]) for i in range(3)]
    return math.sqrt(sum(x*x for x in gap))


def contacts_from_window(window, *, fruit_actor, basket_actor, basket_colliders,
                         collider_owners, impulse_on_actor0_sign):
    """Impulse sum / ACTUAL full control interval, retaining floor/wall identity.

    The sign and world-frame convention require external physical calibration;
    this function does not infer sign from gravity or flip negative readings.
    """
    if type(impulse_on_actor0_sign) is not int or impulse_on_actor0_sign not in (-1, 1):
        raise ValueError("Measured actor0 impulse sign required")
    if window.complete is not True or not math.isfinite(window.duration_s) or window.duration_s <= 0:
        raise ValueError("Incomplete native report interval")
    if type(window.physics_steps) is not int or window.physics_steps < 1:
        raise ValueError("Missing actual physics-step count")
    total, step_contact_strength = {}, {}
    for row in window.contacts:
        if (collider_owners.get(row.collider0) != row.actor0
                or collider_owners.get(row.collider1) != row.actor1):
            raise ValueError("Native actor/collider mapping differs from inspected geometry")
        if fruit_actor not in (row.actor0, row.actor1):
            continue
        fruit_is_zero = row.actor0 == fruit_actor
        actor = row.actor1 if fruit_is_zero else row.actor0
        collider = row.collider1 if fruit_is_zero else row.collider0
        endpoint = collider if actor == basket_actor else actor
        if actor == basket_actor and collider not in basket_colliders:
            raise ValueError("Unclassified compound basket contact; do not guess floor")
        sign = impulse_on_actor0_sign*(1 if fruit_is_zero else -1)
        impulse = tuple(sign*x for x in row.impulse_w_ns)
        previous = total.get(endpoint, (0., 0., 0.))
        total[endpoint] = tuple(a+b for a, b in zip(previous, impulse))
        # Peak substep contact vetoes release, even if interval averaging is small.
        key = (actor, row.physics_step)
        step_contact_strength[key] = step_contact_strength.get(key, 0.)+math.sqrt(sum(x*x for x in impulse))
    hand_peak = {}
    for (actor, _), strength in step_contact_strength.items():
        hand_peak[actor] = max(hand_peak.get(actor, 0.), strength/(window.duration_s/window.physics_steps))
    return tuple(ContactObservation(window.time_s, fruit_actor, endpoint,
                 tuple(x/window.duration_s for x in impulse)) for endpoint, impulse in sorted(total.items())), hand_peak


def compare_raw_contact_direction(window, *, fruit_actor, sensor_forces, last_physics_step,
                                  impulse_on_actor0_sign, minimum_force_n=.05):
    """Positive NONZERO cross-check against the existing apple ContactSensor.

    Only the same final physics substep and same native fruit/hand pair are
    compared. This never selects/flips the proposed sign, accepts zero as proof,
    or claims a new floor calibration from an optimizer or empty contact stream.
    """
    if impulse_on_actor0_sign not in (-1, 1):
        raise ValueError('Explicit proposed impulse sign required')
    dt = window.duration_s/window.physics_steps
    impulses = {}
    for row in window.contacts:
        if row.physics_step != last_physics_step or fruit_actor not in (row.actor0,row.actor1):
            continue
        other = row.actor1 if row.actor0 == fruit_actor else row.actor0
        if other not in sensor_forces:continue
        sign = impulse_on_actor0_sign*(1 if row.actor0 == fruit_actor else -1)
        value = tuple(sign*x/dt for x in row.impulse_w_ns)
        impulses[other] = tuple(a+b for a,b in zip(impulses.get(other,(0.,0.,0.)), value))
    checks = []
    for actor, raw in impulses.items():
        sensor = vector(sensor_forces[actor])
        rn, sn = math.sqrt(sum(x*x for x in raw)), math.sqrt(sum(x*x for x in sensor))
        if min(rn,sn) < minimum_force_n:continue
        alignment = sum(a*b for a,b in zip(raw,sensor))/(rn*sn)
        if alignment < .9:
            raise ValueError('Proposed raw-report sign/pairing disagrees with positive ContactSensor force')
        checks.append(dict(actor=actor, fruit_actor=fruit_actor, sample=window.sample,
                          physics_step=last_physics_step, raw_force_w_n=list(raw),
                          sensor_force_w_n=list(sensor), direction_cosine=alignment))
    return tuple(checks)


@dataclass(frozen=True)
class SafetyPolicy:
    """Proposed parked criteria, not hardware safety ratings or calibrated limits."""
    max_root_drop_m: float = .05
    max_root_tilt_rad: float = .10
    unapproved_contact_force_n: float = .01
    allowed_actor_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (
                self.max_root_drop_m, self.max_root_tilt_rad, self.unapproved_contact_force_n)):
            raise ValueError("Invalid parked safety policy")
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in self.allowed_actor_pairs):
            raise ValueError("Explicit distinct allowed actor pairs required")


def measured_safety(window, *, robot_actors, hand_actors, fruit_actor, basket_actor,
                    root_position_w_m, root_orientation_xyzw, expected_root_height_m, policy, fruit_actors=(),
                    collider_owners=None, branch_colliders=(), target_stem_colliders=()):
    position = vector(root_position_w_m)
    up = rotate(root_orientation_xyzw, (0., 0., 1.))
    tilt = math.acos(max(-1., min(1., up[2])))
    fallen = (position[2] < expected_root_height_m-policy.max_root_drop_m or tilt > policy.max_root_tilt_rad)
    allowed = {frozenset(pair) for pair in policy.allowed_actor_pairs}
    unexpected, accepted_branches = [], []
    branch_colliders, target_stem_colliders = set(branch_colliders), set(target_stem_colliders)
    allowlisted_wood = branch_colliders | target_stem_colliders
    for row in window.contacts:
        pair = frozenset((row.actor0, row.actor1))
        intentional_fruit = ((fruit_actor in pair and bool(pair & set(hand_actors)))
                             or (basket_actor in pair and bool(pair & ({fruit_actor} | set(fruit_actors)))))
        monitored = bool(pair & (set(robot_actors) | {basket_actor}))
        peak = math.sqrt(sum(x*x for x in row.impulse_w_ns))/(window.duration_s/window.physics_steps)
        branch_contact = (is_exact_branch_contact(row, collider_owners, robot_actors, allowlisted_wood)
                          if collider_owners is not None and allowlisted_wood else False)
        if branch_contact:
            contact_kind = ('target_stem' if row.collider0 in target_stem_colliders
                            or row.collider1 in target_stem_colliders else 'branch')
            accepted_branches.append(dict(actor0=row.actor0, actor1=row.actor1,
                collider0=row.collider0, collider1=row.collider1, peak_force_n=peak,
                impulse_w_ns=list(row.impulse_w_ns),
                substep_force_w_n=[x/(window.duration_s/window.physics_steps) for x in row.impulse_w_ns],
                physics_step=row.physics_step, contact_kind=contact_kind))
        if monitored and not intentional_fruit and pair not in allowed and not branch_contact and peak > policy.unapproved_contact_force_n:
            unexpected.append(dict(actor0=row.actor0, actor1=row.actor1, collider0=row.collider0,
                                   collider1=row.collider1, peak_force_n=peak, physics_step=row.physics_step))
    return dict(sample=window.sample, time_s=window.time_s, fallen=bool(fallen),
                unsafe_collision=bool(unexpected), unexpected_contacts=unexpected,
                accepted_branch_contacts=accepted_branches,
                root_tilt_rad=tilt, root_height_m=position[2])


def is_exact_branch_contact(row, collider_owners, robot_actors, branch_colliders):
    """True only for a robot collider paired with an explicitly allowlisted wood collider."""
    owners = (collider_owners.get(row.collider0), collider_owners.get(row.collider1))
    if owners != (row.actor0, row.actor1):
        raise ValueError('Native actor/collider mapping differs from inspected geometry')
    return ((row.collider0 in branch_colliders and row.actor1 in robot_actors)
            or (row.collider1 in branch_colliders and row.actor0 in robot_actors))


def inspected_branch_colliders(geometry):
    """Exact native collider paths for non-root branch meshes named B<number>."""
    return tuple(sorted(path for path in geometry['collider_owners']
        if re.fullmatch(r'B[0-9]+', path.rsplit('/', 1)[-1])
        and path.rsplit('/', 1)[-1] != 'B0'))


def inspected_target_stem_colliders(geometry, target_id):
    """Return exact inspected collider children owned by active Stem_<target_id>."""
    actor_name = 'Stem_' + str(target_id)
    colliders = tuple(sorted(path for path, owner in geometry['collider_owners'].items()
        if owner.rsplit('/', 1)[-1] == actor_name and path.startswith(owner + '/')))
    if not colliders:
        raise ValueError('Inspected active target stem collider missing: ' + actor_name)
    return colliders


def inspect_collision_geometry(stage):
    """Read native enabled collider ownership and actor-local envelopes once.

    Native actor poses subsequently drive these bounds; no stale USD world
    transforms are used during motion. Unknown native actor aliases fail closed.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    # Physical geometry may live under guide-purpose Xforms, and authored
    # drawing extents can be stale (the pinned thumb boxes are undersized).
    # Reuse the existing collision-vertex export convention: read actual mesh
    # points / analytic primitive dimensions, never visual-purpose BBoxCache.
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    owners, bounds, actors, unbounded = {}, {}, set(), {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            actors.add(str(prim.GetPath()))
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        actor = prim
        while actor and not actor.IsPseudoRoot() and not actor.HasAPI(UsdPhysics.RigidBodyAPI):
            actor = actor.GetParent()
        # Static colliders must report their exact native path; any different
        # actor alias needs measured mapping before this reader can accept it.
        if not actor or actor.IsPseudoRoot():
            actor = prim
        actor_path = str(actor.GetPath())
        path = str(prim.GetPath());owners[path] = actor_path
        points, has_unbounded_plane = [], False
        for shape in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            current, disabled = shape, False
            while current and not current.IsPseudoRoot():
                if (current.HasAPI(UsdPhysics.CollisionAPI)
                        and UsdPhysics.CollisionAPI(current).GetCollisionEnabledAttr().Get() is False):
                    disabled = True
                if current == prim:break
                current = current.GetParent()
            if disabled:continue
            vertices = None
            if getattr(UsdGeom, 'Plane', None) is not None and shape.IsA(UsdGeom.Plane):
                # PhysX ground planes are not finite clearance boxes. Preserve
                # exact contact identity and collision enablement, but never
                # invent drawing-size bounds or treat ground as basket support.
                if actor.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError('Rigid/moving plane requires explicit physical bounds: '+str(shape.GetPath()))
                axis = str(UsdGeom.Plane(shape).GetAxisAttr().Get())
                if axis not in ('X','Y','Z'):
                    raise ValueError('Unknown static collision plane axis')
                owners[str(shape.GetPath())] = actor_path
                unbounded[str(shape.GetPath())] = dict(actor_path=actor_path,axis=axis,bounded=False)
                has_unbounded_plane = True
                continue
            elif shape.IsA(UsdGeom.Mesh):
                vertices = UsdGeom.Mesh(shape).GetPointsAttr().Get()
            elif shape.IsA(UsdGeom.Cube):
                half = float(UsdGeom.Cube(shape).GetSizeAttr().Get())/2
                vertices = itertools.product((-half,half), repeat=3)
            elif shape.IsA(UsdGeom.Sphere):
                radius = float(UsdGeom.Sphere(shape).GetRadiusAttr().Get())
                vertices = itertools.product((-radius,radius), repeat=3)
            elif shape.IsA(UsdGeom.Cylinder) or shape.IsA(UsdGeom.Capsule):
                capsule = shape.IsA(UsdGeom.Capsule)
                schema = UsdGeom.Capsule(shape) if capsule else UsdGeom.Cylinder(shape)
                radius, height = float(schema.GetRadiusAttr().Get()), float(schema.GetHeightAttr().Get())
                half = [radius]*3
                half['XYZ'.index(str(schema.GetAxisAttr().Get()))] = height/2+(radius if capsule else 0.)
                vertices = itertools.product(*[(-x,x) for x in half])
            elif shape.IsA(UsdGeom.Gprim):
                raise ValueError('Unsupported physical collider geometry: '+str(shape.GetPath()))
            if vertices is None:continue
            matrix = transforms.GetLocalToWorldTransform(shape)*transforms.GetLocalToWorldTransform(actor).GetInverse()
            transformed = [matrix.Transform(Gf.Vec3d(*v)) for v in vertices]
            if not transformed or any(not math.isfinite(float(x)) for v in transformed for x in v):
                raise ValueError('Missing/nonfinite physical collider vertices: '+str(shape.GetPath()))
            points.extend(transformed)
            # Actual leaf geometry is owned by the same verified collider
            # hierarchy, regardless of whether native reports name leaf or root.
            owners[str(shape.GetPath())] = actor_path
        if has_unbounded_plane:
            continue
        if not points:
            raise ValueError('Missing physical collider bounds: '+path)
        low = tuple(min(float(v[i]) for v in points) for i in range(3))
        high = tuple(max(float(v[i]) for v in points) for i in range(3))
        if actor_path in bounds:
            low = tuple(min(a, b) for a, b in zip(low, bounds[actor_path][0]))
            high = tuple(max(a, b) for a, b in zip(high, bounds[actor_path][1]))
        bounds[actor_path] = (low, high)
    return dict(collider_owners=owners, actor_bounds=bounds, rigid_actor_paths=sorted(actors),
                unbounded_static_colliders=unbounded)


class NativeMountedTelemetryReader:
    """One owned native subscription. Requires reviewed sign/identity/limits.

    Instantiate offline, attach only inside the separately approved concrete
    scene. No calibration/admission files are created. Empty capture intervals
    are zero-force readings, never positive sensor-calibration evidence.
    """
    native_interval_contract = 'mounted_native_window_v1'

    def __init__(self, *, safety_policy=None, window_sink=None, allow_robot_branch_contact=False):
        if window_sink is not None and not callable(window_sink):
            raise ValueError('Native window sink must be callable or absent')
        self.window_sink = window_sink
        self.safety_policy = safety_policy or SafetyPolicy()
        if type(allow_robot_branch_contact) is not bool:
            raise ValueError('Branch-contact training option must be boolean')
        self.allow_robot_branch_contact = allow_robot_branch_contact
        self.source = self.buffer = self.window = None
        self.env = None
        self.closed = False

    def attach(self, env, stage):
        if self.closed or self.env is not None:
            raise ValueError("Reader cannot reattach/retry another scene")
        from omni.usd import get_context
        self.env = env
        self.geometry = inspect_collision_geometry(stage)
        self.branch_colliders = inspected_branch_colliders(self.geometry)
        self.fruit_records = native_fruit_records(env, metadata_only=True)
        self.fruit_actors = tuple(r['body_id'] for r in self.fruit_records)
        self.fruit_actor = next(r['body_id'] for r in self.fruit_records if r['target_id'] == active_fruit_id(env))
        self.basket_actor = env._mounted_plan['body_path']
        root = env.cfg.mounted_spec.robot_prim_path
        self.robot_actors = tuple(a for a in self.geometry['rigid_actor_paths'] if a.startswith(root+'/'))
        hand_names = tuple(dict.fromkeys((*env.cfg.robot.contact_bodies, env.cfg.robot.grasp_palm_body)))
        self.hand_actor_ids = tuple(root+'/'+name for name in hand_names)
        for actor in (*self.hand_actor_ids, *self.fruit_actors, self.basket_actor):
            if actor not in self.geometry['actor_bounds'] or actor not in self.geometry['rigid_actor_paths']:
                raise ValueError("Native monitored collision body missing: " + actor)
        for record in self.fruit_records:
            lo, hi = self.geometry['actor_bounds'][record['body_id']]
            if any(abs((a+b)/2) > 1e-5 for a, b in zip(lo, hi)):
                raise ValueError('Native fruit collider is not centred; explicit asset/frame review required')
            if any(abs((b-a)/2-e) > 1e-5 for a, b, e in zip(lo, hi, record['half_extents_m'])):
                raise ValueError('Native fruit collider extents differ from frozen inventory')
        diagnostic = getattr(env.cfg, 'dense_contact_review', None)
        if diagnostic is not None:
            # A first integrated diagnostic cannot already possess native proof
            # for its changed inventory. Reuse only the explicit measured impulse
            # convention; retain the newly inspected identity map independently.
            if (diagnostic.get('scope') != 'runtime_test_unvalidated'
                    or diagnostic.get('runtime_review_reference') != env.cfg.mounted_runtime_review_reference
                    or diagnostic.get('raw_reader_contract') != self.native_interval_contract
                    or diagnostic.get('report_impulse_frame') != 'world'
                    or type(diagnostic.get('report_impulse_on_actor0_sign')) is not int
                    or diagnostic['report_impulse_on_actor0_sign'] not in (-1, 1)
                    or len(diagnostic.get('calibration_reference', '')) != 64):
                raise ValueError('Explicit reviewed diagnostic contact convention required')
            self.impulse_sign = diagnostic['report_impulse_on_actor0_sign']
            self.calibration_reference = diagnostic['calibration_reference']
            self.identity_validation_scope = 'native_usd_inspection_runtime_test_unvalidated'
            self.contact_mapping_verified = False
            self.mapping_observations = []
        else:
            proof_ref = env._mounted_admission['proofs']['shape_contacts']
            raw = Path(proof_ref['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != proof_ref['sha256']:
                raise ValueError('Changed contact proof at reader attach')
            proof = json.loads(raw)
            for collider, alias in proof.get('static_collider_actor_overrides', {}).items():
                if (self.geometry['collider_owners'].get(collider) != collider
                        or collider in self.geometry['rigid_actor_paths']
                        or not collider.startswith(alias+'/') or not stage.GetPrimAtPath(alias).IsValid()):
                    raise ValueError('Static actor alias is not backed by inspected collider hierarchy')
                self.geometry['collider_owners'][collider] = alias
            self.calibration_reference = proof_ref['sha256']
            self.impulse_sign = proof.get('report_impulse_on_actor0_sign')
            if (type(self.impulse_sign) is not int or self.impulse_sign not in (-1, 1)
                    or proof.get('raw_reader_contract') != self.native_interval_contract
                    or proof.get('report_impulse_frame') != 'world'
                    or proof.get('native_identity_map_sha256') != digest(self.geometry)
                    or proof.get('safety_policy_sha256') != digest(asdict(self.safety_policy))):
                raise ValueError('Native sign/frame/identity/safety policy lacks exact physical proof')
            self.identity_validation_scope = 'prior_exact_native_identity_proof'
            self.contact_mapping_verified = True
            self.mapping_observations = []
        self.buffer = ContactReportBuffer(env.physics_dt, env.cfg.decimation)
        self.source = NativePhysxReportSource(self.buffer, int(get_context().get_stage_id()), lambda: int(env._sim_step_counter))
        self.source.attach(stage, (*self.robot_actors, *self.fruit_actors, self.basket_actor))
        self.other_actor_ids = tuple(sorted(set(self.geometry['collider_owners'].values())-
            {*self.fruit_actors, *self.hand_actor_ids, self.basket_actor}))

    def begin_interval(self, env):
        if env is not self.env or self.buffer is None:
            raise ValueError("Reader not attached to this native scene")
        sample = int(env.episode_length_buf[0])
        self.buffer.begin_interval(sample+1, int(env._sim_step_counter), sample*env.step_dt)

    def finish_interval(self, env):
        self.window = self.buffer.finish_interval(int(env._sim_step_counter), int(env.episode_length_buf[0])*env.step_dt)
        # Persist immediately after capture, before telemetry interpretation,
        # collateral/task guards, adapter callbacks or native cleanup can raise.
        if self.window_sink is not None:
            self.window_sink(self.window)

    def close(self):
        if self.source is not None:self.source.close()
        self.closed = True
        self.env = self.window = None

    def _fresh(self, env, sample, time_s):
        if (self.closed or env is not self.env or self.window is None or self.window.sample != sample or self.window.time_s != time_s
                or self.buffer.errors or self.source is None):
            raise ValueError("No complete fresh native contact interval; explicit neutral prime required")

    def __call__(self, env, sample, time_s, basket_motion):
        self._fresh(env, sample, time_s)
        from .compat import as_torch, canonical_quat
        from .orchard_mounted_env import _row, native_com_offset
        from .harvest_cycle_adapter import motion_at_frame_origin
        records = native_fruit_records(env)
        if [(r['target_id'], r['body_id']) for r in records] != [(r['target_id'], r['body_id']) for r in self.fruit_records]:
            raise ValueError('Native physical inventory changed within continuous episode')
        hand_names = [path.rsplit('/', 1)[-1] for path in self.hand_actor_ids]
        ids, matched = env._robot.find_bodies(hand_names, preserve_order=True)
        if list(matched) != hand_names:
            raise ValueError('Native hand-body order changed')
        positions = as_torch(env._robot.data.body_pos_w)[0, ids].detach().cpu().tolist()
        quats = canonical_quat(as_torch(env._robot.data.body_quat_w))[0, ids].detach().cpu().tolist()
        hand_boxes = tuple(world_bounds(*self.geometry['actor_bounds'][actor], pos, q)
                           for actor, pos, q in zip(self.hand_actor_ids, positions, quats))
        observations, releases, all_contacts = [], [], {}
        hand_open = bool(env.get_state()['hand_open'][0])
        if not self.contact_mapping_verified:
            # _contact_forces is the unchanged current active apple sensor. Its
            # basket column0 and exact hand-filter prefix are preserved by config.
            matrix = as_torch(env._contact_forces())[0].detach().cpu().tolist()
            root = env.cfg.mounted_spec.robot_prim_path
            sensor_forces = {root+'/'+name: matrix[i+1] for i,name in enumerate(env.contact_body_names)}
            actor = next(r['body_id'] for r in records if r['target_id'] == active_fruit_id(env))
            checks = compare_raw_contact_direction(self.window, fruit_actor=actor,
                sensor_forces=sensor_forces, last_physics_step=int(env._sim_step_counter),
                impulse_on_actor0_sign=self.impulse_sign)
            self.mapping_observations.extend(checks)
            self.contact_mapping_verified = bool(checks)
        for record in records:
            actor, target, data = record['body_id'], record['target_id'], record['object'].data
            contacts, peaks = contacts_from_window(self.window, fruit_actor=actor,
                basket_actor=self.basket_actor, basket_colliders={s['path'] for s in env._mounted_plan['shapes']},
                collider_owners=self.geometry['collider_owners'], impulse_on_actor0_sign=self.impulse_sign)
            fruit_motion = motion_at_frame_origin(time_s=time_s, position_w_m=_row(data.root_pos_w),
                orientation_xyzw=_row(canonical_quat(as_torch(data.root_quat_w))),
                com_linear_velocity_w_m_s=_row(data.root_lin_vel_w), angular_velocity_w_rad_s=_row(data.root_ang_vel_w),
                center_of_mass_b_m=native_com_offset(data))
            fruit_box = world_bounds(*self.geometry['actor_bounds'][actor],
                                     fruit_motion.position_w_m, fruit_motion.orientation_xyzw)
            clearance = min(separation_lower_bound(fruit_box, box) for box in hand_boxes)
            held = record['attached'] or any(peaks.get(hand, 0.) > .01 for hand in self.hand_actor_ids)
            observations.append(FruitObservation(target, actor, fruit_motion))
            releases.append(ReleaseObservation(time_s, target, actor, hand_open, held, clearance))
            for contact in contacts:
                key = frozenset((contact.body_a, contact.body_b))
                if key in all_contacts:
                    prior = all_contacts[key]
                    if (contact.body_a == prior.body_a or math.dist(contact.force_on_a_w_n,
                            tuple(-x for x in prior.force_on_a_w_n)) > 1e-6):
                        raise ValueError('Inconsistent two-fruit native contact force pairing')
                else:
                    all_contacts[key] = contact
        # Fruit/fruit pairs are decoded for both endpoints but emitted once.
        return BasketFrame(sample, time_s, self.basket_actor, basket_motion,
            tuple(observations), tuple(releases), tuple(all_contacts.values()), self.contact_mapping_verified, self.calibration_reference, True)

    def safety(self, env, sample, time_s):
        self._fresh(env, sample, time_s)
        from .compat import as_torch, canonical_quat
        from .orchard_mounted_env import _row
        origin = _row(env.scene.env_origins)
        expected_height = env.cfg.scene.robot.init_state.pos[2]+origin[2]
        result = measured_safety(self.window, robot_actors=self.robot_actors, hand_actors=self.hand_actor_ids,
            fruit_actor=next(r['body_id'] for r in native_fruit_records(env) if r['target_id'] == active_fruit_id(env)),
            fruit_actors=self.fruit_actors, basket_actor=self.basket_actor,
            root_position_w_m=_row(env._robot.data.root_pos_w),
            root_orientation_xyzw=_row(canonical_quat(as_torch(env._robot.data.root_quat_w))),
            expected_root_height_m=expected_height, policy=self.safety_policy,
            collider_owners=self.geometry['collider_owners'] if self.allow_robot_branch_contact else None,
            branch_colliders=self.branch_colliders if self.allow_robot_branch_contact else (),
            target_stem_colliders=(inspected_target_stem_colliders(
                self.geometry, active_fruit_id(env)) if self.allow_robot_branch_contact else ()))
        result['source_reference'] = self.calibration_reference
        return result
