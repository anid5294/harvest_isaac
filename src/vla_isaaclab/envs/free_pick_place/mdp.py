"""Measurements for a physical free-object grasp, carry and basket placement."""

import math

import torch

from ..common.g1 import LEFT_END_EFFECTOR, LEFT_HAND_JOINT_NAMES, LOWER_BODY_JOINT_NAMES
from ..orchard_pick.contact import FINGER_LINKS, opposing_contact
from ..orchard_pick.layout import settled_in_basket


def reset_free_pick_place(env, env_ids):
    if env.num_envs != 1:
        raise ValueError("Free pick-and-place currently supports one environment")
    env.free_pick_place_metrics = {}
    env.free_pick_place_last_metrics = {}
    env.free_pick_place_trace = []
    env.free_pick_place_last_step = -1
    env.free_pick_place_phase = "settle"
    env.free_pick_place_failure = None
    env.free_pick_place_contact_steps = 0
    env.free_pick_place_lift_steps = 0
    env.free_pick_place_carry_steps = 0
    env.free_pick_place_stable_steps = 0
    env.free_pick_place_initial_position = env.scene["object"].data.root_pos_w[0].clone()
    env.free_pick_place_max_rise = 0.0
    env.free_pick_place_max_carry_distance = 0.0
    env.free_pick_place_grasped = False
    env.free_pick_place_lifted = False
    env.free_pick_place_carried = False
    env.free_pick_place_released = False
    env.policy_phase = torch.zeros(1, dtype=torch.long, device=env.device)


def update_metrics(env):
    step = int(env.episode_length_buf[0])
    if env.free_pick_place_last_step == step:
        return env.free_pick_place_metrics
    robot = env.scene["robot"]
    apple = env.scene["object"]
    palm_ids, _ = robot.find_bodies([LEFT_END_EFFECTOR], preserve_order=True)
    hand_ids, _ = robot.find_joints(list(LEFT_HAND_JOINT_NAMES), preserve_order=True)
    lower_ids, _ = robot.find_joints(list(LOWER_BODY_JOINT_NAMES), preserve_order=True)
    origin = env.scene.env_origins[0]
    pos_w = apple.data.root_pos_w[0]
    position = (pos_w - origin).detach().cpu().tolist()
    initial = (env.free_pick_place_initial_position - origin).detach().cpu().tolist()
    speed = float(torch.linalg.vector_norm(apple.data.root_lin_vel_w[0]))
    palm = robot.data.body_pos_w[0, palm_ids[0]]
    hand_distance = float(torch.linalg.vector_norm(palm - pos_w))
    finite = bool(torch.isfinite(robot.data.joint_pos).all()
                  and torch.isfinite(robot.data.root_state_w).all()
                  and torch.isfinite(apple.data.root_state_w).all())
    forces = env.scene["apple_contact"].data.force_matrix_w
    if forces is None or forces.shape[2] != 2 + len(FINGER_LINKS):
        raise RuntimeError("Apple contact filters did not resolve floor, support and seven fingers")
    finger_forces = {name: float(torch.linalg.vector_norm(forces[0, 0, index + 2]))
                     for index, name in enumerate(FINGER_LINKS)}
    contact = opposing_contact(finger_forces)
    env.free_pick_place_contact_steps = env.free_pick_place_contact_steps + 1 if contact else 0
    grasp_contact = env.free_pick_place_contact_steps >= 5
    env.free_pick_place_grasped |= grasp_contact
    rise = position[2] - initial[2]
    env.free_pick_place_max_rise = max(env.free_pick_place_max_rise, rise)
    lifted_now = grasp_contact and rise >= 0.06 and hand_distance < 0.18
    env.free_pick_place_lift_steps = env.free_pick_place_lift_steps + 1 if lifted_now else 0
    env.free_pick_place_lifted |= env.free_pick_place_lift_steps >= 5
    horizontal = math.dist(position[:2], initial[:2])
    if env.free_pick_place_lifted and grasp_contact and rise >= 0.04:
        env.free_pick_place_carry_steps += 1
        env.free_pick_place_max_carry_distance = max(env.free_pick_place_max_carry_distance, horizontal)
    else:
        env.free_pick_place_carry_steps = 0
    env.free_pick_place_carried |= (env.free_pick_place_carry_steps >= 10
                                   and env.free_pick_place_max_carry_distance >= 0.12)
    basket_force = float(forces[0, 0, 0, 2])
    support_force = float(forces[0, 0, 1, 2])
    basket_supported = basket_force > env.cfg.apple_mass * 9.81 * 0.4
    finger_open = float(torch.max(torch.abs(robot.data.joint_pos[0, hand_ids]))) < 0.30
    if env.free_pick_place_carried and finger_open and not contact and hand_distance > 0.16:
        env.free_pick_place_released = True
    in_basket = settled_in_basket(position, speed, env.cfg.basket, env.cfg.apple_radius)
    placement = (env.free_pick_place_carried and env.free_pick_place_released
                 and finger_open and not contact and basket_supported and in_basket)
    env.free_pick_place_stable_steps = env.free_pick_place_stable_steps + 1 if placement else 0
    lower_body_error = float(torch.max(torch.abs(
        robot.data.joint_pos[0, lower_ids] - robot.data.default_joint_pos[0, lower_ids])))
    robot_stable = (finite and lower_body_error < 0.05 and
                    float(torch.linalg.vector_norm(robot.data.root_lin_vel_w[0])) < 0.5)
    if not finite:
        env.free_pick_place_failure = "non_finite_state"
    elif position[2] < 0.40:
        env.free_pick_place_failure = "apple_dropped"
    metrics = dict(
        finite=finite, robot_stable=robot_stable,
        lower_body_error_rad=lower_body_error,
        position=position, speed=speed, initial_position=initial, hand_distance=hand_distance,
        finger_contact_force_n=finger_forces, grasp_contact=grasp_contact,
        grasp_contact_steps=env.free_pick_place_contact_steps,
        grasped=env.free_pick_place_grasped,
        grasp_confirmed=env.free_pick_place_grasped, rise_m=rise,
        max_rise_m=env.free_pick_place_max_rise, lifted=env.free_pick_place_lifted,
        carry_steps=env.free_pick_place_carry_steps,
        carry_distance_m=env.free_pick_place_max_carry_distance,
        carried=env.free_pick_place_carried, finger_open=finger_open,
        released=env.free_pick_place_released,
        basket_floor_force_n=basket_force, initial_support_force_n=support_force,
        basket_supported=basket_supported, supported=basket_supported,
        in_basket=in_basket,
        stable_steps=env.free_pick_place_stable_steps,
        failure=env.free_pick_place_failure,
        success=env.free_pick_place_stable_steps >= 30 and robot_stable
        and env.free_pick_place_failure is None,
    )
    env.free_pick_place_metrics = metrics
    env.free_pick_place_last_metrics = dict(metrics)
    env.free_pick_place_last_step = step
    env.free_pick_place_trace.append({"step": step, "time_s": step * env.step_dt,
                                      "phase": env.free_pick_place_phase, **metrics})
    return metrics


def success(env):
    return torch.tensor([update_metrics(env)["success"]], device=env.device)


def failed(env):
    return torch.tensor([update_metrics(env)["failure"] is not None], device=env.device)
