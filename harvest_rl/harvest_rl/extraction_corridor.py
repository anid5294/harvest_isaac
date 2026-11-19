"""Bounded geometric transfer planning; checks fruit and a fixed palm proxy only.

This is not a whole-robot collision or inverse-kinematics feasibility check.
Branch segments are conservative capsules using their maximum endpoint radius.
"""
import math


def _point(value, name):
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        raise ValueError(f"Invalid {name}")
    return result


def _distance(point, a, b):
    ab = tuple(y-x for x, y in zip(a, b))
    denom = sum(x*x for x in ab)
    t = max(0., min(1., sum((x-y)*v for x, y, v in zip(point, a, ab))/denom)) if denom else 0.
    return math.dist(point, tuple(x+t*v for x, v in zip(a, ab)))


def select_corridor(start, basket, root, branches, radius, lift_z, palm_offset=None, sample_step=.005):
    """Return a deterministic 3-leg route, or fail closed with ``ValueError``.

    All inputs are world-frame metres. ``root`` is the arm reach-sphere centre;
    ``palm_offset`` is the fixed world-frame vector from fruit to palm. Sampling
    is at most 5 mm, with a half-step clearance allowance between samples.
    An initially penetrating palm proxy may escape, but may never deepen its
    initial gap against any branch. Fruit always requires 3 mm clearance.
    """
    start, basket, root = (_point(v, n) for v, n in
                           ((start, "start"), (basket, "basket"), (root, "root")))
    offset = _point(palm_offset, "palm_offset") if palm_offset is not None else None
    radius, lift_z = float(radius), float(lift_z)
    if not math.isfinite(sample_step) or not 0 < sample_step <= .005:
        raise ValueError("sample_step must be finite and at most5mm")
    if not math.isfinite(radius) or radius <= 0 or not math.isfinite(lift_z) or lift_z < start[2]:
        raise ValueError("Invalid radius or lift height")
    capsules = []
    for branch in branches:
        a, b = _point(branch["start"], "branch start"), _point(branch["end"], "branch end")
        radii = [float(branch[k]) for k in ("radius_start", "radius_end")]
        if any(not math.isfinite(r) or r < 0 for r in radii):
            raise ValueError("Invalid branch radius")
        capsules.append((a, b, max(radii)))
    heading = math.atan2(basket[1]-start[1], basket[0]-start[0])
    best = None
    evaluated = 0
    for direction in range(16):
        angle = heading + direction*math.tau/16
        for length in (.14, .20, .26, .30):
            evaluated += 1
            extract = (start[0]+length*math.cos(angle), start[1]+length*math.sin(angle), start[2])
            lift = (extract[0], extract[1], lift_z)
            route = (start, extract, lift, (basket[0], basket[1], lift_z))
            # Reject sphere intersections continuously, including inner sphere.
            if any(max(math.dist(a, root), math.dist(b, root)) > .65 or
                   _distance(root, a, b) < .20 for a, b in zip(route, route[1:])):
                continue
            lows = [min(p[i] for p in route) for i in range(3)]
            highs = [max(p[i] for p in route) for i in range(3)]
            padding = max(radius, .045 + math.sqrt(sum(x*x for x in offset)) if offset else 0) + .006
            nearby = [(a, b, r) for a, b, r in capsules if all(
                max(a[i], b[i])+r >= lows[i]-padding and
                min(a[i], b[i])-r <= highs[i]+padding for i in range(3))]
            baseline = [(_distance(tuple(x+y for x, y in zip(start, offset)), a, b)-r-.045)
                        if offset else 0 for a, b, r in nearby]
            clearance = math.inf
            palm_clearance = math.inf
            valid = True
            for a, b in zip(route, route[1:]):
                count = max(1, math.ceil(math.dist(a, b)/sample_step))
                allowance = math.dist(a, b)/count/2
                for j in range(count+1):
                    p = tuple(x+(y-x)*j/count for x, y in zip(a, b))
                    for k, (c, d, r) in enumerate(nearby):
                        gap = _distance(p, c, d)-r-radius
                        clearance = min(clearance, gap-allowance)
                        if gap-allowance < .003:
                            valid = False
                            break
                        if offset:
                            palm = tuple(x+y for x, y in zip(p, offset))
                            pgap = _distance(palm, c, d)-r-.045
                            palm_clearance = min(palm_clearance, pgap)
                            if pgap < min(0., baseline[k])-1e-12:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                distance = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
                # Quantized score avoids floating round-off changing symmetric ties.
                score = (round(distance-.1*min(clearance, .05), 12), direction, length)
                result = dict(extraction_goal=extract, lift_goal=lift,
                              clearance_m=clearance, palm_clearance_m=palm_clearance,
                              path_length_m=distance)
                if best is None or score < best[0]:
                    best = score, result
    if best is None:
        raise ValueError("No extraction corridor meets fruit clearance and reach bounds")
    return dict(best[1], candidates_evaluated=evaluated)
