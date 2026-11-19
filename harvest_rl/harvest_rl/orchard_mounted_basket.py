"""Opt-in orchard compound basket; pure proposal plus lazy USD authoring.

Nothing imports Kit, creates a simulator, follows a link by pose writes, or moves
fruit. The physical basket is one dynamic rigid body with five collision shapes
and a real unbreakable fixed constraint. Defaults are disabled and leave legacy
scenes untouched. Dimensions/mass are design placeholders, not mount approval.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import PurePosixPath
import re


@dataclass(frozen=True)
class MountedBasketSpec:
    enabled: bool = False
    robot_prim_path: str = "/World/envs/env_0/Robot"
    parent_link_path: str | None = None
    basket_prim_path: str = "/World/envs/env_0/MountedBasket"
    joint_path: str = "/World/envs/env_0/MountedBasketJoint"
    mount_position_m: tuple[float, float, float] | None = None
    mount_orientation_xyzw: tuple[float, float, float, float] = (0., 0., 0., 1.)
    inner_size_m: tuple[float, float, float] = (.32, .28, .12)
    wall_thickness_m: float = .01
    dry_mass_kg: float = .5


def _vector(value, n):
    if (not isinstance(value, (tuple, list)) or len(value) != n
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError(f"Require finite {n}-vector")
    return tuple(float(v) for v in value)


def _path(value):
    if (not isinstance(value, str) or not re.fullmatch(r"/[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z_][A-Za-z0-9_]*)+", value)
            or str(PurePosixPath(value)) != value):
        raise ValueError("Require exact absolute USD prim path without regex/variants")
    return value


def build_basket_plan(spec: MountedBasketSpec):
    """Return a JSON-serializable immutable-request plan, not measured evidence.

    The body/root frame is the verifier's B frame: inner floor surface centre,
    +Z toward the opening. Mass is the dry receptacle only; dynamic apples retain
    their own masses and transmit load by contact, never by added basket mass.
    Uniform material density across the five boxes supplies proposed COM/inertia.
    """
    if not isinstance(spec, MountedBasketSpec) or type(spec.enabled) is not bool:
        raise ValueError("Require explicit boolean MountedBasketSpec")
    if not spec.enabled:
        return None
    robot = _path(spec.robot_prim_path)
    parent, body, joint = map(_path, (spec.parent_link_path, spec.basket_prim_path, spec.joint_path))
    if not parent.startswith(robot + "/") or body == robot or body.startswith(robot + "/"):
        raise ValueError("Exact mount link must belong to the declared robot; basket is a separate body")
    paths = (parent, body, joint)
    if any(a == b or a.startswith(b + "/") or b.startswith(a + "/")
           for i, a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError("Robot link, basket and joint require distinct nonnested paths")
    mount = _vector(spec.mount_position_m, 3)
    quat = _vector(spec.mount_orientation_xyzw, 4)
    if abs(sum(v*v for v in quat)-1) > 1e-6:
        raise ValueError("Require normalized xyzw mount orientation")
    width, depth, height = _vector(spec.inner_size_m, 3)
    t, mass = spec.wall_thickness_m, spec.dry_mass_kg
    if (any(v <= 0 for v in (width, depth, height)) or any(type(v) not in (int, float)
            or not math.isfinite(v) or v <= 0 for v in (t, mass))):
        raise ValueError("Require positive finite basket dimensions/dry mass")
    boxes = [
        ("Floor", (width+2*t, depth+2*t, t), (0., 0., -t/2)),
        ("WallXNeg", (t, depth+2*t, height), (-(width+t)/2, 0., height/2)),
        ("WallXPos", (t, depth+2*t, height), ((width+t)/2, 0., height/2)),
        ("WallYNeg", (width, t, height), (0., -(depth+t)/2, height/2)),
        ("WallYPos", (width, t, height), (0., (depth+t)/2, height/2)),
    ]
    volume = sum(math.prod(size) for _, size, _ in boxes)
    masses = [mass*math.prod(size)/volume for _, size, _ in boxes]
    com = tuple(sum(m*center[i] for m, (_, _, center) in zip(masses, boxes))/mass for i in range(3))
    inertia = [0., 0., 0.]
    shapes = []
    for (name, size, center), shape_mass in zip(boxes, masses):
        offset = tuple(center[i]-com[i] for i in range(3))
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            inertia[axis] += shape_mass*(sum(size[i]**2 for i in others)/12
                                        + sum(offset[i]**2 for i in others))
        shapes.append(dict(path=body+"/"+name, size_m=list(size), position_b_m=list(center),
                           dry_mass_contribution_kg=shape_mass, collision_enabled=True))
    return dict(schema="orchard_mounted_basket_plan_v1", robot_prim_path=robot, body_path=body, joint_path=joint,
        parent_link_path=parent, mount_position_m=list(mount), mount_orientation_xyzw=list(quat),
        inner_size_m=[width, depth, height], dry_mass_kg=mass, center_of_mass_b_m=list(com),
        diagonal_inertia_kg_m2=inertia, shapes=shapes,
        fixed_joint=dict(body0=parent, body1=body, exclude_from_articulation=True,
            local_position0_m=list(mount), local_orientation0_xyzw=list(quat),
            local_position1_m=[0., 0., 0.], local_orientation1_xyzw=[0., 0., 0., 1.],
            breakable=False, collision_enabled=True),
        contact_ids=dict(basket_body_id=body, floor_contact_id=body+"/Floor",
                         wall_contact_ids=[s["path"] for s in shapes[1:]]),
        frame="inner_floor_surface_center_z_up", payload_mass_added_kg=0.,
        physics_validated=False, mount_selected=False)


def _rigid_matrix(matrix, Gf):
    """Reject scale/reflection/shear rather than guessing joint-frame conversions."""
    axes = [matrix.TransformDir(Gf.Vec3d(*v)) for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    dot = lambda a, b: sum(float(a[i])*float(b[i]) for i in range(3))
    if any(abs(dot(a, a)-1) > 1e-6 for a in axes) or any(
            abs(dot(axes[i], axes[j])) > 1e-6 for i in range(3) for j in range(i)):
        raise ValueError("Basket attachment requires unit-scale rigid transforms (no shear)")
    determinant = sum(float(axes[0][i])*(float(axes[1][(i+1)%3])*float(axes[2][(i+2)%3])
                     - float(axes[1][(i+2)%3])*float(axes[2][(i+1)%3])) for i in range(3))
    if determinant < 1-1e-6:
        raise ValueError("Reflected attachment transform is invalid")
    if any(not math.isfinite(float(matrix[i][j])) for i in range(4) for j in range(4)):
        raise ValueError("Nonfinite attachment transform")


def spawn_mounted_basket(stage, spec: MountedBasketSpec):
    """Author into an existing, already approved scene before physics starts.

    No runtime loop, world reset, fruit access or per-step pose writer is exposed.
    Caller must be in its concrete orchard scene's setup phase. This function
    cannot prove balance, collisions, workspace, load retention or calibration.
    A fixed-root scene is only eligible for a separately reviewed parked test.
    """
    plan = build_basket_plan(spec)
    if plan is None:
        return None  # No USD imports or stage reads/mutations when disabled.
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    if (UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z
            or abs(UsdGeom.GetStageMetersPerUnit(stage)-1.) > 1e-9):
        raise ValueError("Mounted basket requires metre/Z-up stage")
    parent = stage.GetPrimAtPath(plan["parent_link_path"])
    if (not stage.GetPrimAtPath(plan["robot_prim_path"]).IsValid()
            or not parent.IsValid() or not parent.HasAPI(UsdPhysics.RigidBodyAPI)):
        raise ValueError("Exact mount parent must be an existing physical RigidBody link")
    if not UsdPhysics.RigidBodyAPI(parent).GetRigidBodyEnabledAttr().Get():
        raise ValueError("Mount parent rigid body is disabled")
    for name in ("body_path", "joint_path"):
        if stage.GetPrimAtPath(plan[name]).IsValid():
            raise FileExistsError("Refusing to overwrite existing basket/joint: " + plan[name])
    container_path = str(PurePosixPath(plan["body_path"]).parent)
    container = stage.GetPrimAtPath(container_path)
    if not container.IsValid():
        raise ValueError("Basket container must already exist")
    if not stage.GetPrimAtPath(str(PurePosixPath(plan["joint_path"]).parent)).IsValid():
        raise ValueError("Joint container must already exist; no implicit ancestor authoring")
    cursor = container
    while cursor and not cursor.IsPseudoRoot():
        if cursor.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Basket must not be a nested rigid body; use a separate body plus joint")
        cursor = cursor.GetParent()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    parent_world = cache.GetLocalToWorldTransform(parent)
    container_world = cache.GetLocalToWorldTransform(container)
    _rigid_matrix(parent_world, Gf); _rigid_matrix(container_world, Gf)
    x, y, z, w = plan["mount_orientation_xyzw"]
    mount = Gf.Transform()
    mount.SetTranslation(Gf.Vec3d(*plan["mount_position_m"]))
    mount.SetRotation(Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z))))
    body_world = mount.GetMatrix()*parent_world  # USD row-vector composition.
    body_local = body_world*container_world.GetInverse()
    try:
        body = UsdGeom.Xform.Define(stage, plan["body_path"])
        body.MakeMatrixXform().Set(body_local)
        rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        rigid.CreateRigidBodyEnabledAttr(True); rigid.CreateKinematicEnabledAttr(False)
        mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass_api.CreateMassAttr(plan["dry_mass_kg"])
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*plan["center_of_mass_b_m"]))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*plan["diagonal_inertia_kg_m2"]))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.))
        for shape in plan["shapes"]:
            box = UsdGeom.Cube.Define(stage, shape["path"])
            box.CreateSizeAttr(1.)
            box.AddTranslateOp().Set(Gf.Vec3d(*shape["position_b_m"]))
            box.AddScaleOp().Set(Gf.Vec3f(*shape["size_m"]))
            box.CreateDisplayColorAttr([Gf.Vec3f(.06, .25, .65)])
            UsdPhysics.CollisionAPI.Apply(box.GetPrim()).CreateCollisionEnabledAttr(True)
        joint = UsdPhysics.FixedJoint.Define(stage, plan["joint_path"])
        joint.CreateBody0Rel().SetTargets([Sdf.Path(plan["parent_link_path"])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(plan["body_path"])])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*plan["mount_position_m"]))
        joint.CreateLocalRot0Attr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.))
        joint.CreateBreakForceAttr(float("inf")); joint.CreateBreakTorqueAttr(float("inf"))
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateCollisionEnabledAttr(True)
    except BaseException:
        # Both paths were absent before this call; never touch existing scene data.
        stage.RemovePrim(plan["joint_path"]); stage.RemovePrim(plan["body_path"])
        raise
    return dict(plan, usd_authored=True, physics_validated=False)


def inventory_capacity_plan(spec, fruits, *, clearance_m=.01):
    """Conservative one-layer slots for the entire frozen feasible inventory.

    This is a geometric/payload request, not a structural or balance certificate.
    Each slot encloses the largest fruit's 3-D bounding sphere, so orientation or
    slot assignment cannot make apples overlap by design. Payload is transmitted
    by native contacts; it is never added again to the basket rigid-body mass.
    """
    plan = build_basket_plan(spec)
    if plan is None:
        raise ValueError('Enabled physical mount required for capacity planning')
    fruits = tuple(fruits)
    if not fruits or not math.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError('Require nonempty frozen feasible inventory and positive clearance')
    ids, dimensions, masses = [], [], []
    for fruit in fruits:
        ids.append(str(fruit['target_id']))
        half = _vector(fruit['half_extents_m'], 3)
        mass = fruit['mass_kg']
        if (not ids[-1] or any(x <= 0 for x in half) or type(mass) not in (float, int)
                or not math.isfinite(mass) or mass <= 0):
            raise ValueError('Invalid physical fruit capacity record')
        dimensions.append(half); masses.append(mass)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate frozen feasible fruit ID')
    radius = max(math.sqrt(sum(x*x for x in h)) for h in dimensions)
    fruit_height = 2*radius
    width, depth, height = spec.inner_size_m
    pitch = 2*radius+clearance_m
    nx, ny = math.floor((width-clearance_m)/pitch), math.floor((depth-clearance_m)/pitch)
    slots = [(float((i-(nx-1)/2)*pitch), float((j-(ny-1)/2)*pitch), 0.)
             for j in range(max(0,ny)) for i in range(max(0,nx))]
    enough = len(slots) >= len(fruits) and fruit_height+clearance_m <= height
    return dict(schema='mounted_inventory_capacity_v1', target_ids=ids,
        feasible_count=len(fruits), payload_limit_kg=sum(masses), dry_mass_kg=spec.dry_mass_kg,
        loaded_mass_kg=spec.dry_mass_kg+sum(masses), payload_mass_added_to_body_kg=0.,
        slot_clearance_m=clearance_m, slot_radius_m=radius, slot_pitch_m=pitch,
        slots_b_m=[list(s) for s in slots], geometric_capacity=len(slots) if fruit_height+clearance_m <= height else 0,
        fits=bool(enough), structural_load_validated=False, balance_validated=False,
        physical_mount_validated=False)


def choose_empty_deposit_slot(capacity, deposited_centers_b, fruit_half_extents_m):
    """Choose an empty physical floor slot using measured retained-fruit centres.

    Deposited apples remain active obstacles. Occupancy is rechecked rather than
    inferred from a prior slot reservation; no pose/velocity writes occur here.
    Return None when no slot is clear, requiring a named incomplete FULL stop.
    """
    if capacity.get('schema') != 'mounted_inventory_capacity_v1' or capacity.get('fits') is not True:
        raise ValueError('Frozen complete-inventory capacity has not passed geometry')
    half = _vector(fruit_half_extents_m, 3)
    if any(v <= 0 for v in half):
        raise ValueError('Positive current fruit dimensions required')
    radius = math.sqrt(sum(x*x for x in half))
    if radius > capacity['slot_radius_m']+1e-12:
        raise ValueError('Fruit exceeds frozen slot radius')
    occupied = tuple(_vector(p, 3) for p in deposited_centers_b)
    for slot in capacity['slots_b_m']:
        if all(math.hypot(slot[0]-p[0], slot[1]-p[1]) >=
               radius+capacity['slot_radius_m']+capacity['slot_clearance_m'] for p in occupied):
            return tuple(slot)
    return None


def screen_mount_against_robot_model(spec, robot_model, joint_positions, *, minimum_clearance_m=.003):
    """One requested mount vs original robot collision-vertex bounds at home.

    Joint frames come from the existing read-only USD export. This is a
    conservative AABB enclosure test of one posture, not native contact proof,
    balance, self-collision or a placement/withdrawal path certificate.
    """
    import itertools
    plan = build_basket_plan(spec)
    if plan is None or robot_model.get('stage_meters_per_unit') != 1 or robot_model.get('unsupported_shapes'):
        raise ValueError('Require enabled mount and complete metre-scale collision model')
    if not math.isfinite(minimum_clearance_m) or minimum_clearance_m < 0:
        raise ValueError('Finite nonnegative clearance required')
    def identity():return [[float(i==j) for j in range(4)] for i in range(4)]
    def multiply(a,b):return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
    def inverse(a):
        b=identity()
        for i in range(3):
            for j in range(3):b[i][j]=a[j][i]
            b[i][3]=-sum(a[j][i]*a[j][3] for j in range(3))
        return b
    def transform(position, quaternion):
        matrix = identity()
        x,y,z,w = _vector(quaternion,4)
        if abs(x*x+y*y+z*z+w*w-1.) > 1e-6:raise ValueError('Normalized joint quaternion required')
        rotation = ((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
                    (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
                    (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
        for i in range(3):
            for j in range(3):matrix[i][j]=rotation[i][j]
            matrix[i][3]=_vector(position,3)[i]
        return matrix
    poses = {'pelvis':identity()}
    pending = list(robot_model['joints'])
    while pending:
        ready = [j for j in pending if j['parent'] in poses]
        if not ready:raise ValueError('Disconnected/ambiguous native robot joint graph')
        for joint in ready:
            f0, f1 = joint['frame0'], joint['frame1']
            move = identity()
            if joint['axis']:
                angle = joint_positions.get(joint['name'],0.)
                if not math.isfinite(angle):raise ValueError('Nonfinite requested home joint')
                xyz = [0.,0.,0.]; xyz['XYZ'.index(joint['axis'])] = math.sin(angle/2)
                move = transform((0.,0.,0.),(*xyz,math.cos(angle/2)))
            poses[joint['child']] = multiply(multiply(multiply(poses[joint['parent']],transform(f0['position'],f0['quaternion_xyzw'])),move),inverse(transform(f1['position'],f1['quaternion_xyzw'])))
            pending.remove(joint)
    parent = spec.parent_link_path.rsplit('/',1)[-1]
    if parent not in poses:raise ValueError('Exact mount link absent from native joint model')
    parent_inverse = inverse(poses[parent])
    mount = transform(spec.mount_position_m,spec.mount_orientation_xyzw)
    def box(low,high,matrix):
        points = [tuple(sum(matrix[i][j]*(*p,1.)[j] for j in range(4)) for i in range(3)) for p in itertools.product(*zip(low,high))]
        return (tuple(min(p[i] for p in points) for i in range(3)),tuple(max(p[i] for p in points) for i in range(3)))
    basket_boxes = []
    for shape in plan['shapes']:
        center,size = shape['position_b_m'],shape['size_m']
        low,high = [tuple(c-s/2 for c,s in zip(center,size)),tuple(c+s/2 for c,s in zip(center,size))]
        basket_boxes.append((shape['path'],box(low,high,mount)))
    pairs=[]
    home_body_boxes={}
    for body in robot_model['bodies']:
        bounds=body.get('local_collision_aabb')
        if bounds is None:continue
        if body['name'] not in poses:raise ValueError('Native collision body lacks a joint frame')
        actual=box(bounds['min'],bounds['max'],multiply(parent_inverse,poses[body['name']]))
        home_body_boxes[body['name']]=dict(min=list(actual[0]),max=list(actual[1]))
        for name,basket_box in basket_boxes:
            gaps=[max(0.,actual[0][i]-basket_box[1][i],basket_box[0][i]-actual[1][i]) for i in range(3)]
            distance=math.sqrt(sum(x*x for x in gaps))
            pairs.append(dict(robot_body=body['name'],basket_shape=name,separation_lower_bound_m=distance,
                              clears=bool(distance>=minimum_clearance_m)))
    unresolved=[p for p in pairs if not p['clears']]
    return dict(schema='mounted_home_geometry_screen_v1',parent_link_path=spec.parent_link_path,
        source_robot_asset_sha256=robot_model['asset_sha256'],joint_positions=dict(joint_positions),
        minimum_clearance_m=minimum_clearance_m,pairs=pairs,unresolved_pairs=unresolved,
        home_collision_enclosure_clear=not unresolved,home_body_aabbs_parent=home_body_boxes,
        basket_aabbs_parent={name:dict(min=list(bounds[0]),max=list(bounds[1])) for name,bounds in basket_boxes},native_validated=False,
        placement_path_validated=False,balance_validated=False,
        scope='One home posture, full native collision-vertex AABB enclosure; no path/native/balance certificate')
