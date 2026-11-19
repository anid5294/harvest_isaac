"""Learned physical grasp followed by measured-state pull, carry, and release.

The learned actor reaches and grasps. After a sustained opposing grip, this
controller commands the 14 left-arm and Dex3 joints through the physical pull.
It never writes object state, applies forces, or changes task bookkeeping.
"""

from __future__ import annotations

import math
import torch

from .compat import as_torch
from .orchard_env_cfg import BASKET_WALL_HEIGHT


def _quat_conj_xyzw(q):
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def _quat_mul_xyzw(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat((aw * bv + bw * av + torch.cross(av, bv, dim=-1),
                      aw * bw - (av * bv).sum(-1, keepdim=True)), dim=-1)


def _rotate_xyzw(q, v):
    """Rotate vectors using public state quaternions (xyzw on both Lab versions)."""
    u, w = q[..., :3], q[..., 3:]
    return v + 2.0 * (w * torch.cross(u, v, dim=-1) + torch.cross(u, torch.cross(u, v, dim=-1), dim=-1))


def _orientation_error_xyzw(target, current):
    delta = _quat_mul_xyzw(target, _quat_conj_xyzw(current))
    delta = torch.where(delta[..., 3:] < 0, -delta, delta)
    vector = delta[..., :3]
    length = torch.linalg.norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(length, delta[..., 3:].clamp(min=1e-8))
    return torch.where(length > 1e-8, vector / length * angle, 2.0 * vector)


def _release_ready_for_withdrawal(state):
    """Use physical placement evidence, not the tighter centring waypoint.

    Release distance is intentionally NOT required here: withdrawing creates it.
    The authoritative tracker still requires its full 30-step success hold.
    """
    return (bool(state["hand_open"][0]) and not bool(state["opposing_contact"][0])
            and bool(state["basket_supported"][0]) and bool(state["apple_settled_in_basket"][0]))


def _held_point_jacobian(jacobian, offset):
    """World Jacobian for a material point rigidly held relative to the palm."""
    return jacobian[:3] + torch.cross(jacobian[3:].T,
                                     offset.expand_as(jacobian[3:].T), dim=-1).T


def _contact_limited_closure(current, goal, forces, step=.01):
    """Close only unloaded Dex3 groups; retain real contact-generated pressure."""
    result = current.clone()
    for joints, links in (((0, 1, 2), slice(0, 3)), ((3, 4), slice(3, 5)), ((5, 6), slice(5, 7))):
        if float(forces[links].max()) < 3.:
            ids = list(joints)
            result[ids] += (goal[ids] - result[ids]).clamp(-step, step)
    return result


def _learned_arm_rate(hand_distance, load_ratio, sustained_grasp, unilateral_contact=False):
    """Slow near-contact reaching, not learned fingers or a verified pull."""
    if sustained_grasp:
        return 1.
    proximity = .35+.65*max(0.,min(1.,(hand_distance-.18)/.06))
    load = max(0.,min(1.,(.8-load_ratio)/.6))
    return min(proximity,load,.15 if unilateral_contact else 1.)


def _slew_learned_arm_rate(requested, previous, rise_per_step=.10):
    """Limit acceleration of the optional governor, never delay load braking.

    At 30 Hz a .35 -> 1 recovery takes seven updates instead of one. This
    changes neither the learned finger actions nor measured grasp criteria.
    """
    return min(float(requested), float(previous) + rise_per_step)


class HybridHarvestPolicy:
    """Single-environment, closed-loop learned pick plus bounded Cartesian IK.

    `learned_policy` accepts the environment observation and returns raw 14-D
    delta actions. Call once before each `env.step`; reset is detected from the
    episode counter on the next call after an automatic environment reset.
    """

    def __init__(self, env, learned_policy, *, handoff_before_break=False, hand_step_m=0.006,
                 learned_retention=False, bounded_transfer=False, contact_retention=False,
                 branch_aware_transfer=False, upright_transfer=False, secure_grasp=False,
                 contact_ik=False, planned_grasp=False, learned_grasp_governor=False,
                 learned_contact_clutch=False, learned_grasp_prealign=False,
                 learned_grasp_slew=False):
        if env.num_envs != 1 or env.num_controlled_joints != 14 or env.cfg.action_mode != "delta":
            raise ValueError("HybridHarvestPolicy requires one 14-joint delta-action orchard environment")
        if not all(name.startswith("left_") and "hand_" not in name for name in env.ctrl_joint_names[:7]):
            raise ValueError("First seven controlled joints must be the left arm")
        if not all(name.startswith("left_hand_") for name in env.ctrl_joint_names[7:]):
            raise ValueError("Last seven controlled joints must be the left hand")
        self.env = env
        self.action_dim = learned_policy.action_dim
        self.observation_dim = learned_policy.observation_dim
        self.learned_policy = learned_policy
        self.handoff_before_break = handoff_before_break
        self.hand_step_m = hand_step_m
        self.learned_retention = learned_retention
        self.bounded_transfer = bounded_transfer
        self.contact_retention = contact_retention
        self.branch_aware_transfer = branch_aware_transfer
        self.upright_transfer = upright_transfer
        self.secure_grasp = secure_grasp
        self.contact_ik = contact_ik
        self.planned_grasp = planned_grasp
        self.learned_grasp_governor = learned_grasp_governor
        self.learned_contact_clutch = learned_contact_clutch
        self.learned_grasp_prealign = learned_grasp_prealign
        self.learned_grasp_slew = learned_grasp_slew
        self.finger_ik = None
        self.robot = env._robot
        if not self.robot.is_fixed_base:
            raise ValueError("HybridHarvestPolicy requires the calibrated fixed-base robot")
        self.arm_ids = env.ctrl_joint_ids[:7]
        self.palm_body_id = env._grasp_palm_id
        if int(env.tracked_body_ids[0]) != int(self.palm_body_id):
            raise ValueError("Grasping palm must be the first tracked body")
        self.palm_jacobian_id = self.palm_body_id - 1
        self.step_seen = -1
        self.reset()

    def reset(self):
        if getattr(self,'learned_grasp_prealign',False):
            self.planned_grasp = True
        self.phase = "learned_grasp" if self.handoff_before_break else "learned_pluck"
        self.phase_steps = 0
        self.step_seen = -1
        self.pocket_local = None
        self.learned_previous_arm_rate = .35
        self.handoff_quat = None
        self.hand_goal = None
        self.release_hand_pos = None
        self.closed_fingers = None
        self.handoff_step = None
        self.extraction_goal = None
        self.arm_posture = None
        self.hand_velocity = None
        self.lift_goal = None
        self.transfer_yaw = 0.0
        self.grip_hold = False
        self.grip_stable_steps = 0
        self.finger_correction = None
        self.previous_pocket = None
        self.fruit_goal = None
        self.fruit_velocity = None
        self.secure_quat = None
        self.secure_goal = None
        self.secure_fingers = None
        self.planned_corridor = None
        self.planned_blocked = None
        self.planned_previous_pocket = None
        self.planned_stable_samples = 0
        self.planned_capsules = None
        self.planned_closure_fraction = 0.
        self.planned_joint_goal = None
        self.planned_joint_reference = None
        self.learned_contact_latched = False
        self.diagnostics = {"handoff_step": None, "phase": self.phase}

    def _advance_phase(self, name):
        self.phase = name
        self.phase_steps = 0
        self.diagnostics["phase"] = name

    def _handoff(self, state, step):
        palm = state["body_pos"][0, 0]
        apple = state["apple_pos"][0]
        quat = state["body_quat"][0, 0]
        self.handoff_quat = quat.clone()
        self.command_quat = quat.clone()
        self.pocket_local = _rotate_xyzw(_quat_conj_xyzw(quat), apple - palm).clone()
        self.closed_fingers = state["ctrl_joint_target"][0, 7:].clone()
        self.finger_correction = torch.zeros_like(self.closed_fingers)
        self.hand_goal = palm.clone()
        self.hand_velocity = torch.zeros_like(palm)
        self.arm_posture = state["ctrl_joint_pos"][0, :7].clone()
        self.fruit_goal = apple.clone()
        self.fruit_velocity = torch.zeros_like(apple)
        self.initial_normal = _rotate_xyzw(quat, torch.tensor([0., -1., 0.], device=self.env.device))
        self.comfort_elbow_lower = min(0., float(self.arm_posture[3]))
        away = state["root_pos"][0] - apple
        away[2] = 0.0
        away /= torch.linalg.norm(away).clamp(min=1e-6)
        self.extraction_goal = apple + 0.14 * away
        if self.branch_aware_transfer:
            from .extraction_corridor import select_corridor
            corridor = self.planned_corridor if self.planned_grasp else select_corridor(apple.tolist(), self.env.basket_pos[0].tolist(),
                state["root_pos"][0].tolist(), self.env.cfg.case_branches,
                radius=max(self.env.cfg.fruit.half_extents),
                lift_z=float(self.env._apple_center[0, 2]) + .10,
                palm_offset=(palm - apple).tolist(),
                sample_step=.002 if bool(state["apple_attached"][0]) else .005)
            self.extraction_goal = torch.tensor(corridor["extraction_goal"], device=self.env.device)
            self.lift_goal = torch.tensor(corridor["lift_goal"], device=self.env.device)
            self.diagnostics["corridor"] = corridor
        self.handoff_step = step
        self.diagnostics["handoff_step"] = step
        self._advance_phase("pull" if bool(state["apple_attached"][0]) else "extract")

    def _planned_clearance(self, points):
        """Conservative fruit/wood capsule gaps, including attached deformation."""
        if self.planned_capsules is None:
            return float("inf")
        starts, vectors, radii = self.planned_capsules
        points = points.reshape(-1, 1, 3)
        fraction = ((points-starts)*vectors).sum(-1) / (vectors*vectors).sum(-1).clamp(min=1e-12)
        nearest = starts + fraction.clamp(0., 1.)[..., None]*vectors
        return float((torch.linalg.norm(points-nearest, dim=-1)-radii-
                      max(self.env.cfg.fruit.half_extents)).min())

    def _enter_planned_grasp(self, state):
        """Plan from the undeformed fruit before any approach or finger closure."""
        from .extraction_corridor import select_corridor
        env = self.env
        local_quat = torch.tensor([-.56564459, .11749819, -.19313107, .79305791], device=env.device)
        local_quat /= torch.linalg.norm(local_quat)
        self.secure_quat = _quat_mul_xyzw(state["root_quat"][0], local_quat)
        pocket = torch.tensor([.12918, -.04557, -.00031], device=env.device)
        rest = env._apple_center[0].clone()
        self.planned_final_goal = rest-_rotate_xyzw(self.secure_quat, pocket)
        self.secure_goal = state["body_pos"][0, 0].clone()
        self.secure_fingers = state['ctrl_joint_pos'][0, 7:].clone()
        names = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
        values = (.05746, .96429, .29688, -.07854, -.23410, -.21130, -.19139)
        calibrated = {f"left_hand_{name}_joint": value for name, value in zip(names, values)}
        self.secure_closed = torch.tensor([calibrated[name] for name in env.ctrl_joint_names[7:]], device=env.device)
        # Oppose the thumb BEFORE the final approach, rather than pushing with
        # short-travel fingers while the thumb traverses its full0.84rad sweep.
        preshape = {f'left_hand_{name}_joint': value for name,value in
                    zip(names, (.12,.84,0.,0.,0.,0.,0.))}
        self.planned_open_fingers = torch.tensor(
            [preshape[name] for name in env.ctrl_joint_names[7:]], device=env.device
        ).clamp(env._ctrl_lower[0,7:], env._ctrl_upper[0,7:])
        branches = env.cfg.case_branches
        if branches:
            starts = torch.tensor([b["start"] for b in branches], device=env.device)
            ends = torch.tensor([b["end"] for b in branches], device=env.device)
            radii = torch.tensor([max(b["radius_start"], b["radius_end"]) for b in branches], device=env.device)
            self.planned_capsules = starts, ends-starts, radii
        self._advance_phase("planned_pregrasp")
        self.diagnostics["planned_grasp"] = dict(rest_fruit=rest.tolist(),
            pocket_local=pocket.tolist(), palm_quat_xyzw=self.secure_quat.tolist(),
            final_palm_goal=self.planned_final_goal.tolist(),
            closed_by_joint=calibrated, open_by_joint=dict(zip(env.ctrl_joint_names[7:],
                self.planned_open_fingers.tolist())), clearance_requirement_m=.003)
        try:
            self.planned_corridor = select_corridor(rest.tolist(), env.basket_pos[0].tolist(),
                state["root_pos"][0].tolist(), branches, radius=max(env.cfg.fruit.half_extents),
                lift_z=float(rest[2])+.10, palm_offset=(self.planned_final_goal-rest).tolist(), sample_step=.002)
            # Reserve compliance in the direction of the already clear escape,
            # instead of centring a stiff closure immediately beside the wood.
            # This moves only the planned HAND pose, never the attached fruit.
            escape = torch.tensor(self.planned_corridor['extraction_goal'],device=env.device)-rest
            escape /= torch.linalg.norm(escape).clamp(min=1e-8)
            reserve = 0. if self.learned_grasp_prealign else .08*env.cfg.fruit.radius
            self.planned_final_goal += reserve*escape
            self.planned_corridor = select_corridor(rest.tolist(),env.basket_pos[0].tolist(),
                state['root_pos'][0].tolist(),branches,radius=max(env.cfg.fruit.half_extents),
                lift_z=float(rest[2])+.10,palm_offset=(self.planned_final_goal-rest).tolist(),sample_step=.002)
            self.diagnostics['planned_grasp'].update(final_palm_goal=self.planned_final_goal.tolist(),
                compliance_reserve_m=reserve,compliance_reserve_direction=escape.tolist())
            direction_local = torch.tensor([-1., .25, 0.], device=env.device)
            direction_local /= torch.linalg.norm(direction_local)
            direction = _rotate_xyzw(self.secure_quat, direction_local)
            self.planned_pregrasp_goal = self.planned_final_goal+.05*direction/torch.linalg.norm(direction).clamp(min=1e-8)
            self.diagnostics["planned_grasp"].update(corridor=self.planned_corridor,
                pregrasp_palm_goal=self.planned_pregrasp_goal.tolist(),
                pregrasp_backoff_local=direction_local.tolist(), backoff_distance_m=.05)
            from .reach_ik import OpenReachIK
            model = OpenReachIK(env.joint_names,state['joint_pos'][0].tolist(),
                               state['root_pos'][0].tolist(),state['root_quat'][0].tolist())
            measured_arm = state['ctrl_joint_pos'][0,:7]
            observed_error = float(torch.linalg.norm(torch.tensor(model.forward(
                measured_arm.tolist())[:3,3], device=env.device)-state['body_pos'][0,0]))
            if observed_error > .002:
                raise ValueError(f'Open reach model disagrees with measured palm by{observed_error}m')
            q,screen = model.solve(measured_arm.tolist(),env._ctrl_lower[0,:7].tolist(),
                env._ctrl_upper[0,:7].tolist(),self.planned_pregrasp_goal.tolist(),
                self.secure_quat.tolist(),branches,rest.tolist(),max(env.cfg.fruit.half_extents))
            self.planned_joint_goal = torch.tensor(q,device=env.device,dtype=measured_arm.dtype)
            self.planned_joint_reference = measured_arm.clone()
            self.diagnostics['planned_grasp']['open_reach_ik'] = screen
            self.diagnostics['planned_grasp']['open_reach_fk_error_m'] = observed_error
        except ValueError as exc:
            self.planned_blocked = f"initial_corridor: {exc}"

    def _planned_hold_action(self, state):
        """Stop arm progression after a geometric violation; never override task state."""
        env = self.env
        target = torch.cat((state["ctrl_joint_pos"][0, :7], env._joint_targets[0, 7:]))
        self.diagnostics.update(planned_blocked=self.planned_blocked, phase=self.phase)
        return ((target-env._joint_targets[0])/env.cfg.action_scale).clamp(-1., 1.).unsqueeze(0)

    def _planned_action(self, state, step):
        """Measured alignment gates closure; stable physical grasp gates transfer."""
        from .bounded_ik import bounded_dls
        env = self.env
        palm, quat = state["body_pos"][0, 0], state["body_quat"][0, 0]
        goal = self.planned_pregrasp_goal if self.phase == "planned_pregrasp" else self.planned_final_goal
        position_error = float(torch.linalg.norm(goal-palm))
        rotation_error = float(torch.linalg.norm(_orientation_error_xyzw(self.secure_quat, quat)))
        if self.phase in ('planned_pregrasp','planned_approach'):
            self.secure_fingers += (self.planned_open_fingers-self.secure_fingers).clamp(-.01,.01)
        preshaped = float((state['ctrl_joint_pos'][0,7:]-self.planned_open_fingers).abs().max()) < .02
        if self.phase == "planned_pregrasp" and position_error < .005 and rotation_error < math.radians(5) and preshaped:
            self._advance_phase("planned_approach")
            goal = self.planned_final_goal
        elif self.phase == "planned_approach" and position_error < .003 and rotation_error < math.radians(3):
            if self.learned_grasp_prealign:
                # Keep only the successful open-hand positioning component.
                # Neural feedback, not scripted closure, now acquires the grip.
                self.planned_grasp = False
                self.diagnostics['learned_prealign_complete_step'] = step
                self._advance_phase('learned_pluck')
                return None
            self._advance_phase("planned_close")
        if self.phase == "planned_close":
            amplitude = self.secure_closed-self.planned_open_fingers
            if not bool(state['grasp_contact'][0]):
                self.planned_closure_fraction = min(1., self.planned_closure_fraction+
                    .01/max(float(amplitude.abs().max()), 1e-8))
                self.secure_fingers = self.planned_open_fingers+self.planned_closure_fraction*amplitude
            else:
                # Physical sustained grip, not completion of arbitrary joint
                # angles, stops closure. More closing pushed the held fruit
                # toward wood despite already ample opposing contact forces.
                self.secure_fingers = state['ctrl_joint_target'][0,7:].clone()
            pocket = _rotate_xyzw(_quat_conj_xyzw(quat), state["apple_pos"][0]-palm)
            drift = (float(torch.linalg.norm(pocket-self.planned_previous_pocket))
                     if self.planned_previous_pocket is not None else float("inf"))
            self.planned_previous_pocket = pocket.clone()
            closure_complete = bool((self.secure_closed-self.secure_fingers).abs().max() < 1e-5)
            physical_grasp = (bool(state["grasp_contact"][0])
                              and bool(state["opposing_contact"][0]) and int(state["failure"][0]) == 0)
            self.planned_stable_samples = self.planned_stable_samples+1 if physical_grasp and drift < .001 else 0
            self.diagnostics.update(planned_pocket_local=pocket.tolist(), planned_pocket_drift_m=drift,
                planned_stable_samples=self.planned_stable_samples, planned_closure_complete=closure_complete,
                planned_closure_fraction=self.planned_closure_fraction)
            if self.planned_stable_samples >= 2:
                # Validate the measured connector to the saved route. Never replan
                # from a fruit that fingers have deflected toward a branch.
                apple = state["apple_pos"][0]
                from .grasp_route import validate_grasp_route
                validation = validate_grasp_route(apple.tolist(), env.basket_pos[0].tolist(),
                    state["root_pos"][0].tolist(), env.cfg.case_branches,
                    max(env.cfg.fruit.half_extents), self.planned_corridor,
                    (palm-apple).tolist(), sample_step=.002)
                self.diagnostics["planned_handoff_route_validation"] = validation
                if not validation["valid"]:
                    self.planned_blocked = f"measured_handoff_route: {validation['reason']}"
                    return self._planned_hold_action(state)
                self._handoff(state, step)
                return None
        displacement = goal-self.secure_goal
        self.secure_goal += displacement*torch.clamp(.0015/torch.linalg.norm(displacement).clamp(min=1e-8), max=1.)
        pos_error = (self.secure_goal-palm).clamp(-.03, .03)
        rot_error = _orientation_error_xyzw(self.secure_quat, quat).clamp(-.2, .2)
        jac = as_torch(self.robot.root_physx_view.get_jacobians())[
            0, self.palm_jacobian_id, :, self.arm_ids].clone()
        jac[3:] *= .20
        measured = state["ctrl_joint_pos"][0, :7]
        lower, upper = env._ctrl_lower[0, :7], env._ctrl_upper[0, :7]
        reference = env._joint_targets[0, :7].clamp(lower, upper)
        if self.phase == 'planned_pregrasp' and self.planned_joint_goal is not None:
            # Follow a connected model-based joint branch first; the local
            # Cartesian solve alone stalled on an inappropriate folded branch.
            remaining = self.planned_joint_goal-self.planned_joint_reference
            self.planned_joint_reference += remaining*torch.clamp(
                .01/remaining.abs().max().clamp(min=1e-8),max=1.)
            arm = self.planned_joint_reference.clone()
            kkt = torch.zeros((),device=env.device)
            if float(remaining.abs().max()) < .005:
                self.planned_joint_goal = None  # measured Cartesian final alignment
        else:
            # Retain the integral target needed to counter static loads; do not
            # repeatedly erase it by rebasing targets onto measured joints.
            delta, kkt = bounded_dls(jac, .15*torch.cat((pos_error, .20*rot_error)),
                torch.maximum(lower-reference, torch.full_like(reference, -.01)),
                torch.minimum(upper-reference, torch.full_like(reference, .01)), damping=.05)
            arm = (reference+delta).clamp(lower, upper)
        target = torch.cat((arm, self.secure_fingers.clamp(env._ctrl_lower[0, 7:], env._ctrl_upper[0, 7:])))
        self.phase_steps += 1
        self.diagnostics.update(phase=self.phase, planned_position_error_m=float(torch.linalg.norm(goal-palm)),
            planned_rotation_error_rad=rotation_error, planned_ik_kkt=float(kkt),
            planned_attached_deflection_m=float(torch.linalg.norm(state["apple_pos"][0]-env._apple_center[0])))
        return ((target-env._joint_targets[0])/env.cfg.action_scale).clamp(-1., 1.).unsqueeze(0)

    def _retention_feedback(self, state):
        """Bounded physical closure feedback, not a grasp-state override."""
        pocket = _rotate_xyzw(_quat_conj_xyzw(state["body_quat"][0, 0]),
                              state["apple_pos"][0] - state["body_pos"][0, 0])
        slip = float(torch.linalg.norm(pocket - self.pocket_local))
        motion = float(torch.linalg.norm(pocket - self.previous_pocket)) if self.previous_pocket is not None else 0.
        self.previous_pocket = pocket.clone()
        forces = state["finger_contact_force"][0, :7]
        groups = (forces[:3].max(), forces[3:5].max(), forces[5:7].max())
        opposing_force = min(float(groups[0]), max(float(groups[1]), float(groups[2])))
        weakest = min(range(3), key=lambda i: float(groups[i]))
        # Dex3 thumb flexes positive, middle/index negative. Thumb opposition
        # at its base closes negative (measured loaded joint is above target).
        # Flexion alone can curl the thumb past the fruit without opposition.
        if float(groups[0]) < 20.:
            self.finger_correction[0] = min(.10, float(self.finger_correction[0]) + .002)
        for i, ids in enumerate(((1, 2), (3, 4), (5, 6))):
            if float(groups[i]) < 20. or (slip > .004 and i == weakest):
                for joint in ids:
                    self.finger_correction[joint] = min(.15, float(self.finger_correction[joint]) + .003)
        if not self.grip_hold and (slip > .006 or opposing_force < 1.):
            self.grip_hold = True
            self.grip_stable_steps = 0
        stable = opposing_force >= 5. and motion < .001 and slip < .025
        self.grip_stable_steps = self.grip_stable_steps + 1 if stable else 0
        if self.grip_hold and self.grip_stable_steps >= 10:
            # A real opposing regrasp can settle into a slightly new pocket.
            # Only the controller's measured offset changes; task evidence does not.
            self.pocket_local = pocket.clone()
            self.grip_hold = False
            self.hand_goal = state["body_pos"][0, 0].clone()
            self.hand_velocity.zero_()
        self.diagnostics.update(pocket_slip_m=slip, grip_hold=self.grip_hold,
            finger_group_force_n=[float(f) for f in groups],
            finger_closure_correction=self.finger_correction.tolist())

    def _enter_secure_grasp(self, state):
        env = self.env
        self.secure_quat = state["body_quat"][0, 0].clone()
        self.secure_goal = state["body_pos"][0, 0].clone()
        self.secure_fingers = state["ctrl_joint_pos"][0, 7:].clone()
        # Robot/fruit grasp calibration, NOT a world/seed-specific joint trajectory.
        values = (.09, .96, .44, -.08, -.48, -.18, -.43)
        suffixes = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
        calibrated = {f"left_hand_{name}_joint": value for name, value in zip(suffixes, values)}
        self.secure_closed = torch.tensor([calibrated[name] for name in env.ctrl_joint_names[7:]], device=env.device)
        self.secure_started = int(env.episode_length_buf[0])
        self._advance_phase("secure_grasp")

    def _secure_action(self, state):
        """Align below the undeformed fruit and close with real contact feedback."""
        from .bounded_ik import bounded_dls
        env = self.env
        palm, quat = state["body_pos"][0, 0], state["body_quat"][0, 0]
        pocket = torch.tensor([.13, -.05, 0.], device=env.device)
        rest = env._apple_center[0]
        goal = rest - _rotate_xyzw(self.secure_quat, pocket)
        ratio = float(state["stem_tension"][0] / env.stem_break_force[0])
        opposing = bool(state["opposing_contact"][0])
        if opposing:
            # Closure can deflect the attached fruit toward wood. Physically
            # unload that deflection before asking the unchanged route planner
            # for clearance. Never write fruit position or stem/task state.
            correction = rest-state["apple_pos"][0] if self.contact_ik else torch.zeros_like(palm)
            self.secure_goal = palm + correction * torch.clamp(
                .001 / torch.linalg.norm(correction).clamp(min=1e-8), max=1.)
        elif ratio > .35:
            displacement = state["apple_pos"][0] - rest
            self.secure_goal = palm - .001 * displacement / torch.linalg.norm(displacement).clamp(min=1e-6)
        else:
            displacement = goal - self.secure_goal
            self.secure_goal += displacement * torch.clamp(.0015 / torch.linalg.norm(displacement).clamp(min=1e-8), max=1.)
        pos_error = (self.secure_goal - palm).clamp(-.03, .03)
        rot_error = _orientation_error_xyzw(self.secure_quat, quat).clamp(-.2, .2)
        jac = as_torch(self.robot.root_physx_view.get_jacobians())[
            0, self.palm_jacobian_id, :, self.arm_ids].clone()
        jac[3:] *= .20
        error = torch.cat((pos_error, .20 * rot_error))
        measured = state["ctrl_joint_pos"][0, :7]
        lower, upper = env._ctrl_lower[0, :7], env._ctrl_upper[0, :7]
        if self.phase_steps == 0:
            arm = measured.clone()  # cancel the learned target's accumulated push
        else:
            reference = env._joint_targets[0, :7].clamp(measured-.05, measured+.05).clamp(lower, upper)
            delta, _ = bounded_dls(jac, .15 * error,
                torch.maximum(lower-reference, torch.full_like(reference, -.01)),
                torch.minimum(upper-reference, torch.full_like(reference, .01)), damping=.05)
            arm = (reference + delta).clamp(lower, upper)
        if self.contact_ik:
            if self.finger_ik is None:
                from .finger_contact_ik import FingerContactIK
                self.finger_ik = FingerContactIK(env)
            self.secure_fingers, finger_diagnostics = self.finger_ik.update(state, self.secure_fingers)
            self.diagnostics['finger_contact_ik'] = finger_diagnostics
        else:
            self.secure_fingers = _contact_limited_closure(self.secure_fingers, self.secure_closed,
                                                           state["finger_contact_force"][0, :7])
        target = torch.cat((arm, self.secure_fingers.clamp(env._ctrl_lower[0, 7:], env._ctrl_upper[0, 7:])))
        action = ((target - env._joint_targets[0]) / env.cfg.action_scale).clamp(-1., 1.)
        self.phase_steps += 1
        self.diagnostics.update(phase=self.phase, secure_steps=self.phase_steps,
            secure_palm_error_m=float(torch.linalg.norm(goal-palm)), stem_load_ratio=ratio,
            attached_deflection_m=float(torch.linalg.norm(rest-state["apple_pos"][0])))
        return action.unsqueeze(0)

    def _smooth_hand_goal(self, goal):
        displacement = goal - self.hand_goal
        distance = torch.linalg.norm(displacement)
        if self.bounded_transfer:
            if self.contact_retention and self.grip_hold:
                self.hand_velocity.zero_()
                return self.hand_goal
            # Velocity and acceleration limits, including braking near waypoints.
            speed = min(0.003, float(torch.sqrt(2 * 0.0003 * distance)))
            velocity = displacement * torch.clamp(speed / distance.clamp(min=1e-8), max=1.0)
            change = velocity - self.hand_velocity
            self.hand_velocity += change * torch.clamp(0.0003 / torch.linalg.norm(change).clamp(min=1e-8), max=1.0)
            self.hand_goal += self.hand_velocity
            return self.hand_goal
        self.hand_goal = self.hand_goal + displacement * torch.clamp(self.hand_step_m / distance.clamp(min=1e-8), max=1.0)
        return self.hand_goal

    def _ik_target(self, state, apple_goal):
        if self.upright_transfer:
            return self._upright_ik_target(state, apple_goal)
        env = self.env
        palm = state["body_pos"][0, 0]
        palm_quat = state["body_quat"][0, 0]
        desired_quat = self.handoff_quat
        if self.bounded_transfer:
            # Reorient the palm toward the robot's left-side basket rather than
            # holding a forward-facing pluck pose throughout the inward transfer.
            yaw_goal = (math.pi / 4 if self.phase == "lift" or
                        (self.branch_aware_transfer and self.phase in ("extract", "transport", "lower")) else math.pi / 2
                        if self.phase in ("transport", "lower") else self.transfer_yaw
                        if self.phase in ("release", "withdraw") else 0.0)
            if not (self.contact_retention and self.grip_hold):
                self.transfer_yaw += max(-0.012, min(0.012, yaw_goal - self.transfer_yaw))
            half = self.transfer_yaw / 2
            turn = torch.tensor([0., 0., math.sin(half), math.cos(half)], device=env.device)
            desired_quat = _quat_mul_xyzw(turn, self.handoff_quat)
        self.command_quat = desired_quat
        hand_goal = apple_goal - _rotate_xyzw(desired_quat, self.pocket_local)
        hand_goal = self._smooth_hand_goal(hand_goal)
        pos_error = (hand_goal - palm).clamp(-0.12, 0.12)
        rot_error = _orientation_error_xyzw(desired_quat, palm_quat).clamp(-0.4, 0.4)
        error = torch.cat((pos_error, 0.20 * rot_error), dim=-1)
        jac = as_torch(self.robot.root_physx_view.get_jacobians())[0, self.palm_jacobian_id, :, self.arm_ids].clone()
        jac[3:] *= 0.20
        eye = torch.eye(6, device=env.device, dtype=jac.dtype)
        correction = jac.T @ torch.linalg.solve(jac @ jac.T + 0.05**2 * eye, 0.25 * error)
        target_now = env._joint_targets[0, :7]
        measured = state["ctrl_joint_pos"][0, :7]
        lower, upper = env._ctrl_lower[0, :7], env._ctrl_upper[0, :7]
        if self.bounded_transfer:
            from .bounded_ik import bounded_dls
            # Solve WITH the limits, rather than clipping an unconstrained solution
            # that repeatedly asks saturated wrist joints to do unavailable work.
            # Retain bounded target integration to compensate gravity/contact load;
            # a measured-angle-only target leaves a persistent Cartesian sag.
            reference = torch.clamp(target_now, measured - 0.05, measured + 0.05).clamp(lower, upper)
            delta, kkt = bounded_dls(jac, 0.25 * error,
                torch.maximum(lower - reference, torch.full_like(measured, -0.02)),
                torch.minimum(upper - reference, torch.full_like(measured, 0.02)),
                damping=0.05, posture_delta=0.08 * (self.arm_posture - measured))
            self.diagnostics.update(orientation_error_rad=float(torch.linalg.norm(rot_error)),
                position_error_m=float(torch.linalg.norm(pos_error)), ik_kkt=float(kkt))
            return torch.clamp(reference + delta, lower, upper)
        # A blocked arm must not accumulate a large spring-loaded target error.
        reference = torch.clamp(target_now, measured - 0.05, measured + 0.05)
        desired = torch.clamp(reference + correction.clamp(-0.03, 0.03), lower, upper)
        return desired

    def _upright_ik_target(self, state, apple_goal):
        """Five-dimensional held-fruit/upright-palm task; wrist yaw is free."""
        from .bounded_ik import bounded_dls
        env = self.env
        quat = state["body_quat"][0, 0]
        self.command_quat = quat.clone()
        offset = _rotate_xyzw(quat, self.pocket_local)
        if self.phase == "release":
            apple_goal = self.fruit_goal
        displacement = apple_goal - self.fruit_goal
        distance = torch.linalg.norm(displacement)
        speed = min(.003, float(torch.sqrt(2 * .0003 * distance)))
        velocity = displacement * torch.clamp(speed / distance.clamp(min=1e-8), max=1.)
        change = velocity - self.fruit_velocity
        self.fruit_velocity += change * torch.clamp(.0003 / torch.linalg.norm(change).clamp(min=1e-8), max=1.)
        self.fruit_goal += self.fruit_velocity
        self.hand_goal = self.fruit_goal - offset
        point = state["body_pos"][0, 0] + offset
        position_error = (self.fruit_goal - point).clamp(-.12, .12)
        normal = _rotate_xyzw(quat, torch.tensor([0., -1., 0.], device=env.device))
        # Ramp the small initial tilt correction; no instantaneous wrist change.
        fraction = min(1., (int(env.episode_length_buf[0]) - self.handoff_step) / 60.)
        up = torch.tensor([0., 0., 1.], device=env.device)
        normal_goal = (1-fraction) * self.initial_normal + fraction * up
        normal_goal /= torch.linalg.norm(normal_goal).clamp(min=1e-8)
        orientation_error = torch.cross(normal, normal_goal, dim=-1)
        raw = as_torch(self.robot.root_physx_view.get_jacobians())[
            0, self.palm_jacobian_id, :, self.arm_ids].clone()
        projector = torch.eye(3, device=env.device) - normal[:, None] * normal[None, :]
        jac = torch.cat((_held_point_jacobian(raw, offset), .15 * projector @ raw[3:]))
        error = torch.cat((position_error, .15 * orientation_error))
        measured = state["ctrl_joint_pos"][0, :7]
        lower, upper = env._ctrl_lower[0, :7].clone(), env._ctrl_upper[0, :7]
        # Offline screening found that an elbow>=0 hard constraint introduces
        # disconnected posture changes. Keep the actual physical soft bounds;
        # use posture preference rather than imposing that unverified comfort box.
        reference = env._joint_targets[0, :7].clamp(measured-.05, measured+.05).clamp(lower, upper)
        delta, kkt = bounded_dls(jac, .25 * error,
            torch.maximum(lower-reference, torch.full_like(reference, -.02)),
            torch.minimum(upper-reference, torch.full_like(reference, .02)),
            damping=.05, posture_delta=.04 * (self.arm_posture-measured))
        self.diagnostics.update(held_point_error_m=float(torch.linalg.norm(position_error)),
            palm_tilt_rad=float(torch.acos(torch.dot(normal, up).clamp(-1., 1.))),
            ik_kkt=float(kkt), comfort_elbow_lower_rad=self.comfort_elbow_lower)
        return (reference + delta).clamp(lower, upper)

    @torch.inference_mode()
    def __call__(self, observation):
        env = self.env
        step = int(env.episode_length_buf[0])
        if self.step_seen >= 0 and step < self.step_seen:
            self.reset()
        self.step_seen = step
        state = env.get_state()
        if self.planned_grasp or getattr(self,'learned_grasp_prealign',False):
            if self.planned_corridor is None and self.planned_blocked is None:
                self._enter_planned_grasp(state)
            gap = self._planned_clearance(state["apple_pos"][0])
            self.diagnostics["planned_actual_fruit_clearance_m"] = gap
            if gap < .003:
                self.planned_blocked = "actual_fruit_clearance_below_3mm"
            if self.planned_blocked is not None:
                return self._planned_hold_action(state)
            if self.phase.startswith("planned_"):
                action = self._planned_action(state, step)
                if action is not None:
                    return action
        if self.secure_grasp and self.phase.startswith("learned_") and bool(state["apple_attached"][0]):
            proximity = float(torch.linalg.norm(state["apple_pos"][0] - state["body_pos"][0, 0]))
            contact = float(state["finger_contact_force"][0, :7].max())
            load_ratio = float(state["stem_tension"][0] / env.stem_break_force[0])
            if proximity < .20 or contact > .05 or load_ratio > .20:
                self._enter_secure_grasp(state)
        if self.phase == "secure_grasp":
            unloaded = (not self.contact_ik or not bool(state["apple_attached"][0]) or
                float(torch.linalg.norm(state["apple_pos"][0]-env._apple_center[0])) < .0005)
            if bool(state["grasp_contact"][0]) and bool(state["opposing_contact"][0]) and unloaded:
                self._handoff(state, step)
            else:
                return self._secure_action(state)
        if self.phase.startswith("learned_"):
            # Both flags are measured by the task's real apple/contact sensors.
            if not ((self.handoff_before_break or not bool(state["apple_attached"][0])) and
                    bool(state["grasp_contact"][0]) and bool(state["opposing_contact"][0]) and
                    int(state["failure"][0]) == 0):
                action = self.learned_policy(observation)
                if self.learned_grasp_governor:
                    # Tracker hand_distance is zero until the first physics
                    # update. Use geometry and make startup softening explicit.
                    distance = float(torch.linalg.norm(state['apple_pos'][0]-state['body_pos'][0,0]))
                    ratio = float(state['stem_tension'][0]/env.stem_break_force[0])
                    sustained = bool(state['grasp_contact'][0])
                    self.learned_contact_latched |= bool(state['finger_contact_force'][0,:7].max() >= .05)
                    unilateral = self.learned_contact_clutch and self.learned_contact_latched and not sustained
                    scale = _learned_arm_rate(distance,ratio,sustained,unilateral)
                    if step == 0:
                        scale = min(scale,.35)
                    requested_scale = scale
                    if getattr(self, 'learned_grasp_slew', False):
                        scale = _slew_learned_arm_rate(scale, self.learned_previous_arm_rate)
                    self.learned_previous_arm_rate = scale
                    # Clip before scaling: multiplying unbounded actor output
                    # could otherwise still saturate to full actuator speed.
                    action = action.clone().clamp(-1.,1.)
                    action[:,:7] *= scale
                    self.diagnostics.update(learned_arm_rate_scale=scale,
                        learned_requested_arm_rate_scale=requested_scale,
                        learned_hand_distance_m=distance,learned_stem_load_ratio=ratio,
                        learned_contact_latched=self.learned_contact_latched)
                return action
            self._handoff(state, step)

        apple = state["apple_pos"][0]
        if self.contact_retention and self.phase not in ("release", "withdraw"):
            self._retention_feedback(state)
        basket = env.basket_pos[0]
        floor_apple_z = basket[2] + 0.01 + env.cfg.fruit.radius
        in_basket_xy = torch.linalg.norm((apple - basket)[:2]) < 0.045
        # Release above the opening: the palm cannot descend through the rim.
        release_zone = in_basket_xy and bool(apple[2] < floor_apple_z + BASKET_WALL_HEIGHT + 0.025)
        if self.phase == "pull" and not bool(state["apple_attached"][0]):
            self._advance_phase("extract")
        if self.phase == "extract" and torch.linalg.norm(apple - self.extraction_goal) < 0.025:
            if not self.branch_aware_transfer:
                self.lift_goal = torch.stack((apple[0], apple[1], env._apple_center[0, 2] + 0.09))
            self._advance_phase("lift")
        if self.phase == "lift" and bool(state["carried"][0]):
            self._advance_phase("transport")
        if self.phase == "transport" and in_basket_xy:
            self._advance_phase("lower")
        if self.phase == "lower" and release_zone:
            self.release_hand_pos = state["body_pos"][0, 0].clone()
            if self.learned_retention:
                self.closed_fingers = state["ctrl_joint_pos"][0, 7:].clone()
            elif self.contact_retention:
                self.closed_fingers = (self.closed_fingers + self.finger_correction *
                    torch.tensor([-1., 1., 1., -1., -1., -1., -1.], device=env.device)).clone()
            self._advance_phase("release")
        if self.phase == "release" and self.phase_steps >= 30 and _release_ready_for_withdrawal(state):
            self._advance_phase("withdraw")

        if self.phase in ("pull", "extract"):
            apple_goal = self.extraction_goal
        elif self.phase == "lift":
            apple_goal = self.lift_goal if self.bounded_transfer else torch.stack((apple[0], apple[1],
                                      torch.maximum(env._apple_center[0, 2] + 0.12, apple[2] + 0.02)))
        elif self.phase == "transport":
            apple_goal = torch.stack((basket[0], basket[1],
                                      torch.maximum(env._apple_center[0, 2] + (0.09 if self.bounded_transfer else 0.12), floor_apple_z + 0.20)))
        elif self.phase == "lower":
            apple_goal = torch.stack((basket[0], basket[1], floor_apple_z + 0.025))
        elif self.phase == "release":
            apple_goal = self.hand_goal + _rotate_xyzw(self.command_quat, self.pocket_local)
        else:  # Withdraw from the basket after opening; the apple remains fully dynamic.
            root = state["root_pos"][0]
            away = root - basket
            away[2] = 0.0
            away = away / torch.linalg.norm(away).clamp(min=1e-6)
            retreat = self.release_hand_pos + 0.27 * away + torch.tensor([0.0, 0.0, 0.07], device=env.device)
            release_quat = state["body_quat"][0, 0] if self.upright_transfer else self.command_quat
            apple_goal = retreat + _rotate_xyzw(release_quat, self.pocket_local)

        arm_target = self._ik_target(state, apple_goal)
        if self.phase in ("release", "withdraw"):
            openness = min(1.0, (self.phase_steps + 1) / 30.0) if self.phase == "release" else 1.0
            hand_target = self.closed_fingers * (1.0 - openness)
        else:
            if self.learned_retention:
                hand_action = self.learned_policy(observation)[0, 7:].clamp(-1.0, 1.0)
                hand_target = env._joint_targets[0, 7:] + env.cfg.action_scale * hand_action
            else:
                hand_target = self.closed_fingers
                if self.contact_retention:
                    hand_target = hand_target + self.finger_correction * torch.tensor(
                        [-1., 1., 1., -1., -1., -1., -1.], device=env.device)
        q_target = torch.cat((arm_target, hand_target.clamp(env._ctrl_lower[0, 7:], env._ctrl_upper[0, 7:])))
        action = ((q_target - env._joint_targets[0]) / env.cfg.action_scale).clamp(-1.0, 1.0)
        self.phase_steps += 1
        self.diagnostics["phase"] = self.phase
        return action.unsqueeze(0)
