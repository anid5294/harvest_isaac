"""Offline Dex3/apple clearance audit. Run with Blender's bundled Python (pxr, numpy).

This is a screening tool, not a PhysX collision or reachability certificate. It
measures mesh vertices against the spherical fruit and conservative tree branch
capsules. Triangle interiors, collision approximations, wrist/arm, and dynamics
can change clearance. A negative capsule gap is a flag for further inspection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom


ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / "assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
TREE = ROOT / "assets/orchard/orchardbench_seed42/tree.json"
BASE = "/g1_29dof_with_hand_rev_1_0/"
NAMES = ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")
LINKS = ("palm",) + NAMES
PARENTS = {"index_1": "index_0", "middle_1": "middle_0", "thumb_1": "thumb_0", "thumb_2": "thumb_1"}
OPEN = dict(index_0=-0.0785, index_1=-0.0873, middle_0=-0.0785,
            middle_1=-0.0873, thumb_0=0.0, thumb_1=0.0, thumb_2=0.0873)
profile_spec = importlib.util.spec_from_file_location(
    "cpu_harvest_grasp", ROOT / "src/vla_isaaclab/policies/cpu_harvest_grasp.py")
profile = importlib.util.module_from_spec(profile_spec)
profile_spec.loader.exec_module(profile)
CLOSED = {name: profile.CLOSED_HAND_RAD[f"left_hand_{name}_joint"] for name in NAMES}


def numpy_matrix(gf_matrix):
    return np.array([[gf_matrix[i][j] for j in range(4)] for i in range(4)])


def rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=float)
    q = np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


def load_hand():
    stage = Usd.Stage.Open(str(ROBOT))
    cache = UsdGeom.XformCache()
    palm = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE + "left_hand_palm_link"))
    anchors, axes, limits, vertices = {}, {}, {}, {}
    for name in LINKS:
        if name != "palm":
            joint = stage.GetPrimAtPath(BASE + "joints/left_hand_" + name + "_joint")
            base = np.array(joint.GetAttribute("physics:localPos0").Get(), dtype=float)
            parent = PARENTS.get(name)
            anchors[name] = base + (anchors[parent] if parent else 0)
            axes[name] = str(joint.GetAttribute("physics:axis").Get())
            limits[name] = [float(joint.GetAttribute("physics:lowerLimit").Get()),
                            float(joint.GetAttribute("physics:upperLimit").Get())]
        link_path = BASE + "left_hand_" + name + "_link/collisions/"
        chunks = []
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if not str(prim.GetPath()).startswith(link_path):
                continue
            if prim.GetTypeName() == "Mesh":
                local = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            elif prim.GetTypeName() == "Cube":
                size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
                local = np.array([[x, y, z] for x in (-size/2, size/2)
                                  for y in (-size/2, size/2) for z in (-size/2, size/2)])
            else:
                continue
            transform = numpy_matrix(cache.GetLocalToWorldTransform(prim) * palm.GetInverse())
            chunks.append((np.c_[local, np.ones(len(local))] @ transform)[:, :3])
        vertices[name] = np.concatenate(chunks) if chunks else np.empty((0, 3))
    return anchors, axes, limits, vertices


def pose_hand(joints, anchors, axes, vertices):
    posed = {}
    for name, raw in vertices.items():
        points = raw
        if name == "palm":
            posed[name] = points
            continue
        chain = []
        current = name
        while current:
            chain.append(current)
            current = PARENTS.get(current)
        for joint in chain:
            anchor = anchors[joint]
            points = (points - anchor) @ rotation(axes[joint], joints[joint]).T + anchor
        posed[name] = points
    return posed


def segment_distance(points, start, end):
    vector = end - start
    fraction = np.clip(((points - start) @ vector) / (vector @ vector), 0, 1)
    return np.linalg.norm(points - (start + fraction[:, None] * vector), axis=1)


def audit(args):
    tree = json.loads(TREE.read_text())
    fruit = next(f for f in tree["fruits"] if f["id"] == 28)
    center_world = np.asarray(fruit["center"])
    radius = float(fruit["radius"])
    orientation = quat_matrix(args.palm_quat)
    pocket = np.asarray(args.pocket)
    anchors, axes, limits, vertices = load_hand()
    branches = []
    for branch in tree["branches"]:
        a, b = np.asarray(branch["start"]), np.asarray(branch["end"])
        if segment_distance(center_world[None, :], a, b)[0] < 0.25:
            branches.append((branch["id"], a, b, max(branch["radius_start"], branch["radius_end"])))

    def sample(phase, value, joints):
        hand = pose_hand(joints, anchors, axes, vertices)
        local_fruit = pocket + ([value, 0, 0] if phase == "entry" else 0)
        palm_world = center_world - orientation @ local_fruit
        fruit_gaps = {name: float(np.linalg.norm(points - local_fruit, axis=1).min()) - radius
                      for name, points in hand.items() if len(points)}
        fruit_gap = min(fruit_gaps.values())
        best = (float("inf"), None, None)
        branch_gaps = {}
        for name, points in hand.items():
            if not len(points):
                continue
            world = points @ orientation.T + palm_world
            for branch_id, a, b, width in branches:
                gap = float(segment_distance(world, a, b).min()) - width
                branch_gaps[name] = min(gap, branch_gaps.get(name, float("inf")))
                if gap < best[0]:
                    best = (gap, name, branch_id)
        return {"phase": phase, "fraction_or_offset": float(value), "fruit_vertex_gap_m": fruit_gap,
                "fruit_gap_by_link_m": fruit_gaps, "tree_gap_by_link_m": branch_gaps,
                "tree_vertex_capsule_gap_m": best[0], "nearest_link": best[1], "nearest_branch": best[2]}

    rows = []
    for offset in np.linspace(args.approach_m, 0, 13):
        rows.append(sample("entry", offset, OPEN))
    for fraction in np.linspace(0, 1, 11):
        joints = {name: (1-fraction)*OPEN[name] + fraction*CLOSED[name] for name in NAMES}
        rows.append(sample("close", fraction, joints))
    first_contact = {}
    first_wood = {}
    for row in rows:
        if row["phase"] != "close":
            continue
        for name in LINKS:
            if row["fruit_gap_by_link_m"].get(name, float("inf")) <= 0:
                first_contact.setdefault(name, row["fraction_or_offset"])
            if row["tree_gap_by_link_m"].get(name, float("inf")) <= 0:
                first_wood.setdefault(name, row["fraction_or_offset"])
    return {"apple_radius_m": radius, "pocket_palm_m": pocket.tolist(),
            "candidate_only": True, "open_hand_rad": OPEN, "closed_hand_rad": CLOSED,
            "palm_quat_wxyz": list(args.palm_quat), "joint_limits_deg": limits,
            "mesh_vertex_counts": {name: len(vertices[name]) for name in LINKS},
            "minimum_entry_tree_gap_m": min(r["tree_vertex_capsule_gap_m"] for r in rows if r["phase"] == "entry"),
            "minimum_close_tree_gap_m": min(r["tree_vertex_capsule_gap_m"] for r in rows if r["phase"] == "close"),
            "minimum_entry_fruit_gap_m": min(r["fruit_vertex_gap_m"] for r in rows if r["phase"] == "entry"),
            "first_sampled_fruit_contact_fraction_by_link": first_contact,
            "first_sampled_wood_overlap_fraction_by_link": first_wood,
            "samples": rows,
            "limitation": "Conservative branch capsule uses maximum taper radius; vertex distances miss triangle interiors and PhysX shapes. Closure after first fruit contact may be physically blocked. Arm and dynamics unverified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pocket", type=float, nargs=3, default=(0.105, -0.05, 0))
    parser.add_argument("--palm-quat", type=float, nargs=4,
                        default=(0.8387395, -0.0562444, 0.0882924, 0.5343754))
    parser.add_argument("--approach-m", type=float, default=0.12)
    print(json.dumps(audit(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
