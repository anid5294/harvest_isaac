"""Native PhysX stem break reporting and independently measured task outcomes."""

import torch

from ..common.g1 import LEFT_END_EFFECTOR, LEFT_HAND_JOINT_NAMES
from .layout import settled_in_basket, stem_geometry
from .contact import FINGER_LINKS, opposing_contact
from .interaction import HarvestEvidence


def reset_orchard(env, env_ids):
    # This first gate deliberately supports one environment and one layout per process.
    # Pose resets are Isaac Lab's reset_scene_to_default, never a rollout operation.
    if env.num_envs != 1:
        raise ValueError("Orchard pick currently supports num_envs=1")
    import omni.physx
    import omni.usd
    from omni.physx.bindings._physx import SimulationEvent
    from pxr import Gf, PhysicsSchemaTools, UsdPhysics

    if not hasattr(env, "orchard_break_subscription"):
        def on_event(event):
            if event.type == int(SimulationEvent.JOINT_BREAK):
                path = str(PhysicsSchemaTools.decodeSdfPath(*event.payload["jointPath"]))
                if path in env.orchard_joint_paths:
                    env.orchard_broken.add(env.orchard_joint_paths.index(path))
                    diagnostics = getattr(env, "_orchard_physics_diagnostics", None)
                    if diagnostics is not None:
                        diagnostics.on_break(path)
        env.orchard_break_subscription = (
            omni.physx.get_physx_interface().get_simulation_event_stream_v2()
            .create_subscription_to_pop(on_event)
        )
    env.orchard_broken = set()
    env.orchard_joint_paths = []
    env.orchard_carried = False
    env.orchard_hold = 0
    env.orchard_failure = None
    env.orchard_released = False
    env.orchard_pull_started = False
    env.orchard_lift_steps = 0
    env.orchard_contact_steps = 0
    env.orchard_evidence = HarvestEvidence()
    env.orchard_last_step = -1
    env.orchard_metrics = {}
    env.orchard_trace = []
    env.orchard_phase = "passive"
    stage = omni.usd.get_context().get_stage()
    layout = env.cfg.orchard_layout
    for i in range(len(layout.apples)):
        root = env.scene.env_prim_paths[0]
        path = f"{root}/FruitJoint_{i}"
        env.orchard_joint_paths.append(path)
        # Recreate only at reset to clear PhysX's broken-constraint state.
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
        joint = UsdPhysics.FixedJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([f"{root}/Stem_{i}"])
        joint.CreateBody1Rel().SetTargets([f"{root}/Apple_{i:02d}"])
        geometry = stem_geometry(layout, i)
        joint.CreateLocalPos0Attr(Gf.Vec3f(*geometry["stem_joint_position"]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*geometry["fruit_joint_position"]))
        joint.CreateLocalRot0Attr(Gf.Quatf(1))
        w, x, y, z = geometry["fruit_joint_rotation"]
        joint.CreateLocalRot1Attr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        joint.CreateBreakForceAttr(layout.stem_break_force)
        joint.CreateBreakTorqueAttr(0.35)
        joint.CreateExcludeFromArticulationAttr(True)
    env.policy_phase = torch.zeros(1, dtype=torch.long, device=env.device)


def update_metrics(env):
    # Success/failure terms share one update, so counters advance once per control step.
    step = int(env.episode_length_buf[0])
    if env.orchard_last_step == step:
        return env.orchard_metrics
    env.orchard_last_step = step
    layout = env.cfg.orchard_layout
    robot = env.scene["robot"]
    apple = env.scene["object"]
    palm_ids, _ = robot.find_bodies([LEFT_END_EFFECTOR], preserve_order=True)
    hand_ids, _ = robot.find_joints(list(LEFT_HAND_JOINT_NAMES), preserve_order=True)
    position = (apple.data.root_pos_w[0] - env.scene.env_origins[0]).tolist()
    speed = float(torch.linalg.vector_norm(apple.data.root_lin_vel_w[0]))
    hand_distance = float(torch.linalg.vector_norm(
        robot.data.body_pos_w[0, palm_ids[0]] - apple.data.root_pos_w[0]))
    finite = bool(torch.isfinite(robot.data.joint_pos).all() and
                  torch.isfinite(apple.data.root_state_w).all())
    detached = layout.target_index in env.orchard_broken
    if env.orchard_broken - {layout.target_index}:
        env.orchard_failure = "non_target_apple_detached"
    measured_task = layout.tree_model in ("commercial", "orchardbench")
    if detached and not measured_task and not env.orchard_pull_started:
        env.orchard_failure = "premature_stem_break"
    lifted = detached and position[2] > layout.target[2] + 0.06 and hand_distance < 0.18
    forces = env.scene["apple_contact"].data.force_matrix_w
    if forces is None or forces.shape[2] != 1 + len(FINGER_LINKS):
        raise RuntimeError("Apple contact filters did not resolve floor plus seven finger links")
    finger_forces = {name: float(torch.linalg.vector_norm(forces[0, 0, i+1]))
                     for i, name in enumerate(FINGER_LINKS)}
    contact = opposing_contact(finger_forces)
    env.orchard_contact_steps = env.orchard_contact_steps + 1 if contact else 0
    grasp_contact = env.orchard_contact_steps >= 5
    if measured_task:
        # Break/carry evidence comes from sensor state, not scripted phase flags.
        # This short grace handles contact readback at the joint-break boundary;
        # it is not a claim of force closure.
        env.orchard_evidence.update(detached=detached, sustained_contact=grasp_contact,
                                    lift_candidate=lifted)
        env.orchard_lift_steps = env.orchard_evidence.lift_steps
        env.orchard_carried = env.orchard_evidence.carried
        if env.orchard_evidence.failure:
            env.orchard_failure = env.orchard_evidence.failure
    else:
        env.orchard_lift_steps = env.orchard_lift_steps + 1 if lifted else 0
        env.orchard_carried |= env.orchard_lift_steps >= 10
    finger_ids, finger_names = robot.find_bodies(list(FINGER_LINKS), preserve_order=True)
    if list(finger_names) != list(FINGER_LINKS):
        raise RuntimeError(f"Unexpected Dex3 finger links: {finger_names}")
    finger_distances = {name: float(torch.linalg.vector_norm(
        robot.data.body_pos_w[0, body_id] - apple.data.root_pos_w[0]))
        for name, body_id in zip(FINGER_LINKS, finger_ids)}
    from isaaclab.utils.math import quat_rotate_inverse
    apple_in_palm = quat_rotate_inverse(
        robot.data.body_quat_w[0, palm_ids[0]],
        apple.data.root_pos_w[0] - robot.data.body_pos_w[0, palm_ids[0]]).tolist()
    supported = forces is not None and float(forces[0, 0, 0, 2]) > layout.mass * 9.81 * 0.4
    opened = float(torch.max(torch.abs(robot.data.joint_pos[0, hand_ids]))) < 0.30
    if measured_task:
        env.orchard_released = env.orchard_evidence.released(
            opened=opened, contact=contact, hand_distance=hand_distance)
    placed = (finite and detached and env.orchard_carried and env.orchard_released
              and opened and hand_distance > 0.18 and supported
              and settled_in_basket(position, speed, layout.basket, layout.radius))
    env.orchard_hold = env.orchard_hold + 1 if placed else 0
    if not finite:
        env.orchard_failure = "non_finite_state"
    elif position[2] < 0.40:
        env.orchard_failure = "apple_dropped"
    metrics = dict(detached=detached, detached_indices=sorted(env.orchard_broken),
                   carried=env.orchard_carried, released=env.orchard_released,
                   supported=bool(supported), opened=opened, hand_distance=hand_distance,
                   position=position, speed=speed, stable_steps=env.orchard_hold,
                   finger_contact_force_n=finger_forces, finger_link_distance_m=finger_distances,
                   apple_in_palm_frame_m=apple_in_palm, grasp_contact=grasp_contact,
                   grasp_contact_steps=env.orchard_contact_steps,
                   hand_joint_position_rad=robot.data.joint_pos[0, hand_ids].tolist(),
                   failure=env.orchard_failure, success=env.orchard_hold >= 30 and env.orchard_failure is None)
    env.orchard_metrics = metrics
    env.orchard_last_metrics = dict(metrics)
    env.orchard_trace.append({"step": step, "time_s": step * env.step_dt,
                              "phase": env.orchard_phase, **metrics})
    return metrics


def success(env):
    return torch.tensor([update_metrics(env)["success"]], device=env.device)


def failed(env):
    return torch.tensor([update_metrics(env)["failure"] is not None], device=env.device)
