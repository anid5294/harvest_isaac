"""CPU-only Dex3 candidate geometry; physical contact remains authoritative.

Values are derived from the supplied G1 USD joint axes and collision meshes.
They are not a validated harvest policy and do not modify the sugar-box task.
"""

import math

APPLE_CENTER_IN_PALM_M = (0.105, -0.050, 0.0)
ENTRY_DISTANCE_M = 0.12
CLOSED_HAND_RAD = {
    # Preserve inherited magnitudes (nominal targets * 1.10). Only fix the
    # negative-only finger directions; do not also retune closing strength.
    "left_hand_thumb_0_joint": 0.22,
    "left_hand_thumb_1_joint": 0.88,
    "left_hand_thumb_2_joint": 0.88,
    "left_hand_middle_0_joint": -1.32,
    "left_hand_middle_1_joint": -1.32,
    "left_hand_index_0_joint": -1.32,
    "left_hand_index_1_joint": -1.32,
}


def rotate_vector(quaternion_wxyz, vector):
    """Rotate a vector with a normalized wxyz quaternion, without torch."""
    w, x, y, z = quaternion_wxyz
    norm = (w*w + x*x + y*y + z*z) ** 0.5
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Quaternion must have nonzero finite norm")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    vx, vy, vz = vector
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return (vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz, vz+w*tz+x*ty-y*tx)


def grasp_offset_world(quaternion_wxyz):
    return tuple(-v for v in rotate_vector(quaternion_wxyz, APPLE_CENTER_IN_PALM_M))


def pregrasp_offset_world(quaternion_wxyz):
    # Withdraw the palm along its finger axis, not along arbitrary world Z.
    return rotate_vector(quaternion_wxyz, (-ENTRY_DISTANCE_M, 0.0, 0.0))
