"""Reviewed parked-root configuration only; no simulator imports or pose writes."""
import math


def _finite_vector(value, size):
    values = tuple(float(x) for x in value)
    if len(values) != size or not all(math.isfinite(x) for x in values):
        raise ValueError('Finite reviewed stance vector required')
    return values


def bind_reviewed_stance(cfg, stance, backend_quaternion):
    """Bind a prospective reset pose, preserving the existing joint posture.

    The canonical actor frame has root yaw90deg. This only uses the existing
    world-vector frame transform; it does not rotate/move fruit or the basket.
    All validation precedes configuration mutation. No runtime state is written.
    """
    position = _finite_vector(stance['position_m'], 3)
    quaternion = _finite_vector(stance['orientation_xyzw'], 4)
    if abs(sum(x*x for x in quaternion)-1.) > 1e-8 or max(abs(x) for x in quaternion[:2]) > 1e-8:
        raise ValueError('Normalized yaw-only reviewed stance required')
    yaw = math.degrees(2*math.atan2(quaternion[2], quaternion[3]))
    frame = float(stance['policy_frame_yaw_deg'])
    if not math.isfinite(frame) or abs((frame-(90.-yaw)+180.) % 360.-180.) > 1e-6:
        raise ValueError('Reviewed policy frame does not match canonical root yaw')
    if stance['arm_reset_noise_rad'] != 0.:
        raise ValueError('Reviewed dense diagnostic requires zero reset noise')
    joints = stance['default_joint_positions']
    if (cfg.scene.robot.init_state.joint_pos != joints
            or not all(math.isfinite(float(x)) for x in joints.values())):
        raise ValueError('Initial joint posture differs from frozen review')
    rotation = _finite_vector(backend_quaternion(quaternion), 4)
    cfg.scene.robot.init_state.pos = position
    cfg.scene.robot.init_state.rot = rotation
    cfg.policy_frame_yaw_deg = frame
    cfg.arm_reset_noise = 0.
    return cfg


def verify_measured_root(position, quaternion_xyzw, stance, *, tolerance=1e-5):
    """Check measured canonical root, accepting equivalent quaternion signs.

    This is an initialization identity check, not balance/contact admission.
    """
    position = _finite_vector(position, 3)
    quaternion = _finite_vector(quaternion_xyzw, 4)
    expected = _finite_vector(stance['position_m'], 3)
    rotation = _finite_vector(stance['orientation_xyzw'], 4)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Positive finite root tolerance required')
    if abs(sum(x*x for x in quaternion)-1.) > tolerance:
        raise ValueError('Measured root quaternion is not normalized')
    if (max(abs(x-y) for x,y in zip(position, expected)) > tolerance
            or min(max(abs(x-s*y) for x,y in zip(quaternion, rotation)) for s in (-1,1)) > tolerance):
        raise ValueError('Measured root differs from frozen reviewed stance')


def initial_reset_positions(default_q, names, positions):
    """Return the initial dense posture in native joint order without changing policy home."""
    import torch

    if len(names) != 43 or len(set(names)) != 43 or set(names) != set(positions) or tuple(default_q.shape) != (1, 43):
        raise ValueError("Exact 43-joint initial reset order required")
    posture = default_q.new_tensor([[positions[name] for name in names]])
    if not torch.isfinite(posture).all():
        raise ValueError("Nonfinite initial reset posture")
    return posture
