"""Small bounded multistart IK for the open-hand reach only (no simulator).

Robot-model joint priors choose a flexed-elbow branch; endpoints are computed
from the measured root and target pose, never from an orchard seed. Transfer
and all physical contact/success checks remain independent.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .extraction_corridor import _distance


class OpenReachIK:
    def __init__(self, joint_names, joint_position, root_position, root_quaternion):
        self.chain = json.loads((Path(__file__).parent/'assets/dex3_arm_chain.json').read_text())
        self.names = self.chain['arm_order']
        self.angles = dict(zip(joint_names,joint_position))
        self.root = np.eye(4)
        self.root[:3,3] = root_position
        self.root[:3,:3] = Rotation.from_quat(root_quaternion).as_matrix()
        self.frames = [(j,np.array(j['frame0']),np.linalg.inv(j['frame1'])) for j in self.chain['joints']]

    def forward(self, q):
        angles = dict(self.angles)
        angles.update(zip(self.names,q))
        pose = self.root.copy()
        for joint,frame0,frame1 in self.frames:
            motion = np.eye(4)
            if joint['axis']:
                axis = np.eye(3)['XYZ'.index(joint['axis'])]
                motion[:3,:3] = Rotation.from_rotvec(axis*angles[joint['name']]).as_matrix()
            pose = pose@frame0@motion@frame1
        return pose

    def solve(self, start, lower, upper, goal, quaternion, branches, fruit, fruit_radius):
        start,lower,upper,goal = map(np.asarray,(start,lower,upper,goal))
        rotation = Rotation.from_quat(quaternion).as_matrix()
        # A robot-wide bent-elbow posture prior, not a seed/world trajectory.
        flexed = np.array([-.6,.15,-.2,.5,-1.3,.2,.3])
        def residual(q):
            pose = self.forward(q)
            return np.r_[pose[:3,3]-goal,
                .2*Rotation.from_matrix(rotation@pose[:3,:3].T).as_rotvec(),
                1e-5*(q-flexed)]
        nearby = [b for b in branches if min(math.dist(b['start'],fruit),math.dist(b['end'],fruit)) < 1.]
        answers = []
        for label,seed in (('measured',start),('flexed',flexed),('midpoint',(lower+upper)/2)):
            fit = least_squares(residual,np.clip(seed,lower+1e-7,upper-1e-7),bounds=(lower,upper),
                max_nfev=160,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            pose = self.forward(fit.x)
            position_error = float(np.linalg.norm(pose[:3,3]-goal))
            rotation_error = float(np.linalg.norm(Rotation.from_matrix(rotation@pose[:3,:3].T).as_rotvec()))
            wood,fruit_gap = math.inf,math.inf
            for fraction in np.linspace(0,1,33):
                palm = self.forward(start+fraction*(fit.x-start))[:3,3]
                fruit_gap = min(fruit_gap,math.dist(palm,fruit)-fruit_radius-.045)
                for b in nearby:
                    wood = min(wood,_distance(palm,b['start'],b['end'])-max(b['radius_start'],b['radius_end'])-.045)
            item = dict(seed=label,q=fit.x.tolist(),position_error_m=position_error,
                rotation_error_rad=rotation_error,elbow_rad=float(fit.x[3]),
                sampled_palm_wood_clearance_m=wood,sampled_palm_fruit_clearance_m=fruit_gap)
            answers.append(item)
        valid = [a for a in answers if a['position_error_m']<.002 and a['rotation_error_rad']<math.radians(2)
                 and a['sampled_palm_wood_clearance_m']>=0 and a['sampled_palm_fruit_clearance_m']>=.003]
        if not valid:
            raise ValueError(f'No screened open-reach IK candidate: {answers}')
        # Prefer the naturally flexed branch without imposing a hard elbow bound
        # on later manipulation, where the verified transfer solver is retained.
        best = min(valid,key=lambda a:(a['elbow_rad']<0,np.linalg.norm(np.array(a['q'])-flexed)))
        return np.array(best['q']),dict(selected=best,candidates=answers,
            limitation='Sampled palm-only geometry screen, not swept whole-arm collision certification.')
