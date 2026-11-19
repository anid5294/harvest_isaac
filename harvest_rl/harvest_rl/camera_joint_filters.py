"""Opt-in hand fixed-island adjacency repair; external geometry is untouched.

Camera and palm are separate rigid siblings fixed at the same wrist frame.
Complete their existing joint collision-disabled semantics across the fixed
assembly and one adjacent hinge, never two hinges or distal fingers/world.
"""
def camera_pair_plan(joints, robot_root):
    """Prove exact pair semantics from enabled USD joints, including thumb DOF.

    Input uses full body paths. A movable thumb is NOT treated as fixed: the
    path to it must contain exactly one revolute joint with collisions disabled.
    """
    by_name = {row['path'].rsplit('/', 1)[-1]: row for row in joints}
    pairs = []
    for side in ('left', 'right'):
        camera, palm, thumb = (robot_root+'/'+side+'_hand_'+suffix
                               for suffix in ('camera_base_link', 'palm_link', 'thumb_0_link'))
        wrist = robot_root+'/'+side+'_wrist_yaw_link'
        selected = [by_name.get(side+'_hand_'+suffix) for suffix in
                    ('camera_base_joint', 'palm_joint', 'thumb_0_joint')]
        if any(j is None for j in selected):raise ValueError('Missing exact camera assembly joint')
        for joint, kind, parent, child in zip(selected,
                ('PhysicsFixedJoint', 'PhysicsFixedJoint', 'PhysicsRevoluteJoint'),
                (wrist, wrist, palm), (camera, palm, thumb)):
            if (joint['type'] != kind or joint['body0'] != [parent] or joint['body1'] != [child]
                    or joint['joint_enabled'] is not True or joint['collision_enabled'] is not False):
                raise ValueError('Changed camera assembly adjacency/collision semantics')
        # The fixed siblings have the same complete authored local frames.
        if (selected[0]['local_frames'] != selected[1]['local_frames']):
            raise ValueError('Camera and palm no longer share the exact fixed frame')
        for target, movable in ((palm, 0), (thumb, 1)):
            pairs.append(dict(camera=camera, target=target,
                reason='same_fixed_island' if movable == 0 else 'one_revolute_joint_adjacent_to_fixed_island',
                movable_joints=movable, joint_paths=[j['path'] for j in selected[:2+movable]]))
    return pairs


def inspect_camera_joints(stage, robot_root):
    """Read only the six exact joints. SDK is lazy; no simulation is started."""
    from pxr import UsdPhysics
    rows = []
    for side in ('left', 'right'):
        for suffix in ('camera_base_joint', 'palm_joint', 'thumb_0_joint'):
            path = robot_root+'/joints/'+side+'_hand_'+suffix
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsA(UsdPhysics.Joint):raise ValueError('Missing exact native joint: '+path)
            j = UsdPhysics.Joint(prim)
            rows.append(dict(path=path, type=prim.GetTypeName(),
                body0=[str(x) for x in j.GetBody0Rel().GetTargets()],
                body1=[str(x) for x in j.GetBody1Rel().GetTargets()],
                joint_enabled=j.GetJointEnabledAttr().Get(),collision_enabled=j.GetCollisionEnabledAttr().Get(),
                local_frames=dict(pos0=list(j.GetLocalPos0Attr().Get()),pos1=list(j.GetLocalPos1Attr().Get()),
                    rot0=str(j.GetLocalRot0Attr().Get()),rot1=str(j.GetLocalRot1Attr().Get()))))
    return rows


def hand_island_pair_plan(joints, robot_root):
    """Complete the two hand fixed-island adjacency closures, not pair chasing.

    Same-island pairs and pairs across ONE collision-disabled revolute joint
    are internal/adjacent. Do not propagate across a second movable joint.
    Already-direct joint pairs are omitted because USD already excludes them.
    """
    camera_pair_plan(joints, robot_root)
    edges=[j for j in joints if len(j['body0'])==len(j['body1'])==1 and j['joint_enabled'] is True]
    parent={v:v for j in edges for v in (j['body0'][0],j['body1'][0])}
    def find(v):
        while parent[v]!=v:
            parent[v]=parent[parent[v]];v=parent[v]
        return v
    for j in edges:
        if j['type']=='PhysicsFixedJoint' and j['collision_enabled'] is False:
            parent[find(j['body0'][0])]=find(j['body1'][0])
    direct={frozenset((j['body0'][0],j['body1'][0])) for j in edges if j['collision_enabled'] is False}
    result=[]
    for side in ('left','right'):
        camera=robot_root+'/'+side+'_hand_camera_base_link'
        island=sorted(v for v in parent if find(v)==find(camera))
        if set(island)!={camera,robot_root+'/'+side+'_hand_palm_link',robot_root+'/'+side+'_wrist_yaw_link'}:
            raise ValueError('Changed hand fixed-island membership')
        neighbors=[]
        for j in edges:
            a,b=j['body0'][0],j['body1'][0]
            if (a in island)==(b in island):continue
            if j['type']!='PhysicsRevoluteJoint' or j['collision_enabled'] is not False:
                raise ValueError('Changed hand adjacent-joint collision semantics')
            outside=b if a in island else a
            adjacent=sorted(v for v in parent if find(v)==find(outside))
            if len(adjacent)!=1:raise ValueError('Changed adjacent hand rigid island')
            neighbors.append((adjacent[0],j['path']))
        expected={robot_root+'/'+side+'_'+name for name in ('wrist_pitch_link','hand_thumb_0_link','hand_index_0_link','hand_middle_0_link')}
        if {v for v,_ in neighbors}!=expected or len(neighbors)!=4:
            raise ValueError('Changed hand adjacency boundary')
        candidates=[(a,b,0,None) for i,a in enumerate(island) for b in island[i+1:]]
        candidates += [(a,b,1,j) for b,j in neighbors for a in island]
        for a,b,movable,joint in candidates:
            if frozenset((a,b)) not in direct:
                result.append(dict(body0=a,body1=b,movable_joints=movable,
                    boundary_joint=joint,fixed_island=island))
    if len(result)!=18:raise ValueError('Unexpected hand adjacency closure size')
    return result


def inspect_hand_joint_graph(stage, robot_root):
    from pxr import Usd,UsdPhysics
    rows={j['path']:j for j in inspect_camera_joints(stage,robot_root)}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root),Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.Joint) or str(prim.GetPath()) in rows:continue
        j=UsdPhysics.Joint(prim);path=str(prim.GetPath())
        rows[path]=dict(path=path,type=prim.GetTypeName(),
            body0=[str(x) for x in j.GetBody0Rel().GetTargets()],body1=[str(x) for x in j.GetBody1Rel().GetTargets()],
            collision_enabled=j.GetCollisionEnabledAttr().Get(),joint_enabled=j.GetJointEnabledAttr().Get())
    return list(rows.values())


def apply_camera_joint_filters(stage, robot_root):
    """Author complete hand fixed-island exclusions before initialization.

    SafetyPolicy.allowed_actor_pairs is deliberately unchanged. If native
    filtering is ignored or any other contact occurs, the safety guard stops.
    Original asset bytes, colliders, mass/inertia and all joint DOFs are intact.
    """
    from pxr import UsdPhysics
    plan = hand_island_pair_plan(inspect_hand_joint_graph(stage, robot_root), robot_root)
    # Complete preflight first: never partially apply a malformed assembly.
    for row in plan:
        for name in ('body0', 'body1'):
            prim=stage.GetPrimAtPath(row[name])
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Missing exact camera pair rigid body')
    for row in plan:
        api=UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(row['body0']))
        api.CreateFilteredPairsRel().AddTarget(row['body1'])
    return dict(schema='hand_fixed_island_adjacency_v2', pairs=plan,
        external_filters_added=False, geometry_changed=False, native_validated=False)
