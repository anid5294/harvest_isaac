"""Measured-pose distal collision geometry guides; forces alone establish grasp.

The authored convex hull is a geometric guide, not PhysX cooked contact data.
Only physical finger joint targets are commanded. No object/task state is changed.
"""
import json
from pathlib import Path

import torch

from .compat import as_torch, canonical_quat
from .bounded_ik import bounded_dls


def rotate(q, vectors):
    u, w = q[:3].expand_as(vectors), q[3]
    return vectors + 2 * (w * torch.cross(u, vectors, dim=-1)
        + torch.cross(u, torch.cross(u, vectors, dim=-1), dim=-1))


def closest_triangle_point(point, triangles):
    """Closest point on a triangle soup, including edges/vertices, in one frame."""
    a, b, c = triangles.unbind(1)
    ab, ac = b-a, c-a
    normal = torch.cross(ab, ac, dim=-1)
    normal2 = (normal*normal).sum(-1).clamp(min=1e-20)
    projected = point - normal * (((point-a)*normal).sum(-1)/normal2)[:, None]
    v = projected-a
    aa, bb, cc = (ab*ab).sum(-1), (ab*ac).sum(-1), (ac*ac).sum(-1)
    va, vc = (v*ab).sum(-1), (v*ac).sum(-1)
    denom = (aa*cc-bb*bb).clamp(min=1e-20)
    s, t = (cc*va-bb*vc)/denom, (aa*vc-bb*va)/denom
    inside = (s >= 0) & (t >= 0) & (s+t <= 1) & (normal2 > 1e-19)
    candidates = [projected]
    for start, end in ((a,b), (b,c), (c,a)):
        edge = end-start
        fraction = (((point-start)*edge).sum(-1)/(edge*edge).sum(-1).clamp(min=1e-20)).clamp(0,1)
        candidates.append(start+fraction[:,None]*edge)
    candidates = torch.stack(candidates, dim=1)
    distances = ((candidates-point)**2).sum(-1)
    distances[:, 0] = torch.where(inside, distances[:, 0], torch.inf)
    return candidates.reshape(-1,3)[distances.flatten().argmin()]


def radial_joint_step(jacobian, radial, gap, lower_delta, upper_delta, strength=1.):
    """Small bounded inward step; no geometry-derived contact or grasp flag."""
    gradient = radial @ jacobian
    # Keep gently seeking actual contact even if the sphere/hull guide overlaps.
    distance_step = -max(.0003, min(.002, .3*(gap+.002))) * strength
    delta, _ = bounded_dls(gradient.unsqueeze(0), gradient.new_tensor([distance_step]),
                          lower_delta, upper_delta, damping=.005)
    return delta


class FingerContactIK:
    def __init__(self, env):
        self.env, self.robot = env, env._robot
        path = Path(__file__).parent/'assets'/'dex3_distal_hulls.json'
        data = json.loads(path.read_text())
        self.guides = []
        if data['schema'] != 'dex3_distal_authored_hulls_v1':
            raise ValueError('Unexpected distal collision geometry schema')
        for name, entry in data['links'].items():
            prefix = name.rsplit('_', 2)[0]  # left_hand_thumb/index/middle
            body_id = self.robot.body_names.index(name)
            controlled = [i for i,n in enumerate(env.ctrl_joint_names) if n.startswith(prefix+'_')]
            force_ids = [i for i,n in enumerate(env.contact_body_names) if n.startswith(prefix+'_')]
            if len(controlled) not in (2,3) or not force_ids:
                raise ValueError(f'Cannot resolve finger guide by name: {name}')
            vertices = torch.tensor(entry['vertices'], device=env.device, dtype=torch.float32)
            faces = torch.tensor(entry['triangles'], device=env.device, dtype=torch.long)
            self.guides.append((name, body_id, controlled, force_ids, vertices[faces]))

    def update(self, state, current):
        env = self.env
        raw = as_torch(self.robot.root_physx_view.get_jacobians())[0]
        positions = as_torch(self.robot.data.body_pos_w)[0]-env.scene.env_origins[0]
        quats = canonical_quat(as_torch(self.robot.data.body_quat_w)[0])
        result = current.clone()
        diagnostics = {}
        for name, body, controlled, contacts, triangles in self.guides:
            force = float(state['finger_contact_force'][0, contacts].max())
            # 3N acquisition contact is insufficient for a17N stem pull: with
            # authored dynamic friction1.1, two12N opposing contacts provide
            # about26N tangential capacity. This is pressure feedback ONLY;
            # the tracker still independently measures sustained real grasp.
            force_goal = 12.
            loaded = force >= force_goal
            q = quats[body]
            inverse = torch.cat((-q[:3], q[3:]))
            local_apple = rotate(inverse, state['apple_pos'][0]-positions[body])
            witness = closest_triangle_point(local_apple, triangles)
            offset = rotate(q, witness)
            displacement = positions[body]+offset-state['apple_pos'][0]
            distance = torch.linalg.norm(displacement).clamp(min=1e-8)
            radial = displacement/distance
            gap = float(distance)-env.cfg.fruit.radius
            if not loaded:
                ids = [env.ctrl_joint_ids[i] for i in controlled]
                jac = raw[body-1, :, ids]
                point_jac = jac[:3]+torch.cross(jac[3:].T, offset.expand_as(jac[3:].T), dim=-1).T
                measured = state['ctrl_joint_pos'][0, controlled]
                reference = env._joint_targets[0, controlled].clamp(measured-.04, measured+.04)
                lower, upper = env._ctrl_lower[0, controlled], env._ctrl_upper[0, controlled]
                reference = reference.clamp(lower, upper)
                delta = radial_joint_step(point_jac, radial, gap,
                    torch.maximum(lower-reference, torch.full_like(reference,-.01)),
                    torch.minimum(upper-reference, torch.full_like(reference,.01)),
                    strength=max(0., min(1., (force_goal-force)/force_goal)))
                result[[i-7 for i in controlled]] = (reference+delta).clamp(lower,upper)
            diagnostics[name] = dict(guide_gap_m=gap, loaded=loaded,
                measured_force_n=force, force_goal_n=force_goal)
        return result, diagnostics
