"""Express world geometry in the fixed frame used to train the actor."""

import math
import torch


def rotate_vectors(vectors, yaw_degrees):
    angle = math.radians(yaw_degrees)
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = vectors.unbind(-1)
    return torch.stack((c*x-s*y, s*x+c*y, z), dim=-1)


def rotate_quaternions_xyzw(quaternions, yaw_degrees):
    angle = math.radians(yaw_degrees) / 2
    c, s = math.cos(angle), math.sin(angle)
    x, y, z, w = quaternions.unbind(-1)
    return torch.stack((c*x-s*y, c*y+s*x, c*z+s*w, c*w-s*z), dim=-1)
