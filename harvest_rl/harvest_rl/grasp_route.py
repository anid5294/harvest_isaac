"""Check a preselected extraction route against measured grasp deformation.

No route search or relaxed physical clearance: same fruit/palm/reach proxies
and3mm margin as extraction_corridor. Analytic segment/capsule distance avoids
extra sampling allowance at an accurately measured starting point. This is
not whole-body collision certification.
"""
import math

from .extraction_corridor import _distance, _point


def segment_distance(a,b,c,d):
    """Exact finite-segment distance: endpoint cases plus interior stationary point."""
    u,v,w = ([y-x for x,y in zip(a,b)], [y-x for x,y in zip(c,d)],
             [x-y for x,y in zip(a,c)])
    dot = lambda x,y: sum(a*b for a,b in zip(x,y))
    aa,bb,cc,dd,ee = dot(u,u),dot(u,v),dot(v,v),dot(u,w),dot(v,w)
    best = min(_distance(a,c,d),_distance(b,c,d),_distance(c,a,b),_distance(d,a,b))
    denominator = aa*cc-bb*bb
    if denominator > 1e-20:
        s,t = (bb*ee-cc*dd)/denominator,(aa*ee-bb*dd)/denominator
        if 0 <= s <= 1 and 0 <= t <= 1:
            best = min(best,math.sqrt(sum((x+s*y-t*z)**2 for x,y,z in zip(w,u,v))))
    return best


def validate_grasp_route(start, basket, root, branches, radius, corridor,
                         palm_offset, sample_step=.002):
    start, basket, root, offset = (_point(v, n) for v, n in
        ((start,'start'), (basket,'basket'), (root,'root'), (palm_offset,'palm_offset')))
    if not math.isfinite(radius) or radius <= 0 or not 0 < sample_step <= .005:
        raise ValueError('Invalid radius or sample step')
    extract = _point(corridor['extraction_goal'],'extraction_goal')
    lift = _point(corridor['lift_goal'],'lift_goal')
    route = (start, extract, lift, (basket[0],basket[1],lift[2]))
    if any(max(math.dist(a,root),math.dist(b,root))>.65 or
           _distance(root,a,b)<.20 for a,b in zip(route,route[1:])):
        return dict(valid=False, reason='reach', clearance_m=None)
    low, high = [min(p[i] for p in route) for i in range(3)], [max(p[i] for p in route) for i in range(3)]
    padding = max(radius,.045+math.sqrt(sum(x*x for x in offset)))+.006
    nearby = []
    palm_start = tuple(x+y for x,y in zip(start,offset))
    for branch in branches:
        a,b = _point(branch['start'],'branch start'),_point(branch['end'],'branch end')
        r = max(float(branch['radius_start']),float(branch['radius_end']))
        if not math.isfinite(r) or r < 0:
            raise ValueError('Invalid branch radius')
        if all(max(a[i],b[i])+r>=low[i]-padding and min(a[i],b[i])-r<=high[i]+padding for i in range(3)):
            nearby.append((a,b,r,_distance(palm_start,a,b)-r-.045))
    clearance, palm_clearance = math.inf,math.inf
    for leg,(a,b) in enumerate(zip(route,route[1:])):
        palm_a,palm_b = (tuple(x+y for x,y in zip(p,offset)) for p in (a,b))
        for c,d,r,baseline in nearby:
            gap = segment_distance(a,b,c,d)-r-radius
            pgap = segment_distance(palm_a,palm_b,c,d)-r-.045
            clearance,palm_clearance = min(clearance,gap),min(palm_clearance,pgap)
            if gap < .003:
                return dict(valid=False, reason='fruit_clearance',leg=leg,
                            clearance_m=clearance,palm_clearance_m=palm_clearance)
            if pgap < min(0.,baseline)-1e-12:
                return dict(valid=False, reason='palm_clearance',leg=leg,
                            clearance_m=clearance,palm_clearance_m=palm_clearance)
    return dict(valid=True,reason=None,clearance_m=clearance,palm_clearance_m=palm_clearance,
                geometry_method='analytic_segment_capsule')
