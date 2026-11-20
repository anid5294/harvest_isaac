"""Batched apple-stem model for many apples per env (pure torch, no simulator imports).

While an apple is attached, a spring-damper holds it at its hanging rest pose every physics
step: stiff along the stem, softer sideways, a torsional spring about the stem (twist) and a
bending spring against tilting it. The stem breaks only when a **failure criterion** on the
loads actually carried by the stem holds for ``hold_steps`` consecutive physics steps:

    D = (F_axial+ / F_tensile)^2 + (M_bend / M_bend_max)^2 + (T_twist / T_twist_max)^2  >= 1

* ``F_axial+``: tension along the stem (pulling the apple away from the branch) [N]. The weight
  of the hanging apple is part of it, as it is in reality, but the thresholds are sampled
  independently of mass: a heavy apple is not assumed to need a harder pull.
* ``M_bend``: bending moment at the stem = lateral force x lever (stem + apple half height)
  plus the tilt spring's moment [N m]. Lifting/tilting the apple sideways is the cheap way
  to pick it, as in hand picking.
* ``T_twist``: torsion from twisting the apple about the stem, i.e. torque applied through
  the wrist [N m].

The hold requirement (default 3 steps = 25 ms at 120 Hz) filters single-step contact impulses,
so brushing or bumping an apple does not knock it off by itself; a sustained pull, bend or twist
does. Thresholds are per apple and per episode (domain randomization); their ranges live in
the env config and should be refined with measured detachment data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _axis_angle_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Rotation vector (axis * angle) of (x, y, z, w) quaternions, angle in [0, pi]."""
    q = torch.where(q[..., 3:4] < 0, -q, q)
    v, w = q[..., :3], q[..., 3:4].clamp(-1.0, 1.0)
    s = torch.linalg.norm(v, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, w)
    scale = torch.where(s > 1e-8, angle / s.clamp_min(1e-8), torch.full_like(s, 2.0))
    return v * scale


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ), dim=-1)


def _quat_conj_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


@dataclass
class StemParams:
    axial_stiffness: float = 1000.0
    """[N/m] along the stem."""
    lateral_stiffness: float = 300.0
    """[N/m] sideways."""
    twist_stiffness: float = 0.03
    """[N m/rad] about the stem axis."""
    tilt_stiffness: float = 0.2
    """[N m/rad] against tilting the apple off the stem axis."""
    damping_ratio: float = 0.7
    hold_steps: int = 3
    """Consecutive physics steps the failure criterion must hold before the stem breaks."""


class StemModel:
    """State and forces of ``(num_envs, num_apples)`` stems."""

    def __init__(self, num_envs: int, num_apples: int, params: StemParams, device):
        n, k = num_envs, num_apples
        self.p = params
        self.device = device
        z3 = lambda: torch.zeros(n, k, 3, device=device)  # noqa: E731
        z = lambda: torch.zeros(n, k, device=device)  # noqa: E731
        self.anchor = z3()
        """Where the stem meets the branch, env-local [m]."""
        self.center = z3()
        """Apple centre at rest (hanging, loaded by its weight), env-local [m]."""
        self.axis = z3()
        """Unit vector from the apple centre to the anchor."""
        self.rest = z3()
        """Spring rest point: the unloaded position, so that the loaded apple sits at ``center``."""
        self.rest_quat = torch.zeros(n, k, 4, device=device)
        self.rest_quat[..., 3] = 1.0
        self.mass = z()
        self.inertia = z()
        self.lever = z()
        """Stem length + apple half height: lever arm of a lateral force about the stem's base [m]."""
        self.f_tensile = z()
        self.m_bend_max = z()
        self.t_twist_max = z()
        self.attached = torch.zeros(n, k, dtype=torch.bool, device=device)
        self.over_steps = torch.zeros(n, k, dtype=torch.long, device=device)
        # last-step loads (diagnostics / observations)
        self.axial = z()
        self.bending = z()
        self.torsion = z()
        self.damage = z()
        """Failure index D (>= 1 means the criterion holds)."""

    # -- configuration --------------------------------------------------------------------

    def set_apples(self, env_ids, anchor, center, mass, half_height, stem_length, f_tensile, m_bend_max, t_twist_max, gravity):
        """(Re)attach apples of ``env_ids``: all arguments are (len(env_ids), K[, 3]) tensors."""
        axis = anchor - center
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-9)
        k_ax, k_lat = self.p.axial_stiffness, self.p.lateral_stiffness
        weight = gravity.view(1, 1, 3) * mass.unsqueeze(-1)
        w_ax = (weight * axis).sum(-1, keepdim=True) * axis
        self.anchor[env_ids] = anchor
        self.center[env_ids] = center
        self.axis[env_ids] = axis
        self.rest[env_ids] = center - (w_ax / k_ax + (weight - w_ax) / k_lat)
        self.rest_quat[env_ids] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.mass[env_ids] = mass
        # Solid sphere of the apple's mean radius (only used for damping).
        self.inertia[env_ids] = 0.4 * mass * half_height**2
        self.lever[env_ids] = stem_length + half_height
        self.f_tensile[env_ids] = f_tensile
        self.m_bend_max[env_ids] = m_bend_max
        self.t_twist_max[env_ids] = t_twist_max
        self.attached[env_ids] = True
        self.over_steps[env_ids] = 0
        for buf in (self.axial, self.bending, self.torsion, self.damage):
            buf[env_ids] = 0.0

    # -- physics step -------------------------------------------------------------------

    def step(self, pos, quat, lin_vel, ang_vel, step_index=None):
        """Forces/torques (world frame) to apply to every apple this physics step.

        Args: (N, K, 3/4) env-local position, (x, y, z, w) quaternion, world velocities.
        Returns ``(force, torque, newly_detached)``; forces are zero for detached apples.
        """
        p = self.p
        axis = self.axis
        m = self.mass.clamp_min(1e-4).unsqueeze(-1)
        c_ax = 2.0 * p.damping_ratio * torch.sqrt(p.axial_stiffness * m)
        c_lat = 2.0 * p.damping_ratio * torch.sqrt(p.lateral_stiffness * m)
        inertia = self.inertia.clamp_min(1e-7).unsqueeze(-1)
        c_tw = 2.0 * p.damping_ratio * torch.sqrt(p.twist_stiffness * inertia)
        c_tilt = 2.0 * p.damping_ratio * torch.sqrt(p.tilt_stiffness * inertia)

        disp = pos - self.rest
        d_ax = (disp * axis).sum(-1, keepdim=True)
        disp_lat = disp - d_ax * axis
        v_ax = (lin_vel * axis).sum(-1, keepdim=True)
        v_lat = lin_vel - v_ax * axis
        elastic = -(p.axial_stiffness * d_ax * axis + p.lateral_stiffness * disp_lat)
        force = elastic - (c_ax * v_ax * axis + c_lat * v_lat)

        rot = _axis_angle_from_quat_xyzw(_quat_mul_xyzw(quat, _quat_conj_xyzw(self.rest_quat)))
        twist = (rot * axis).sum(-1, keepdim=True)
        tilt = rot - twist * axis
        w_ax = (ang_vel * axis).sum(-1, keepdim=True)
        w_tilt = ang_vel - w_ax * axis
        torque = -(p.twist_stiffness * twist * axis + p.tilt_stiffness * tilt) - (c_tw * w_ax * axis + c_tilt * w_tilt)

        # Loads carried by the stem (equal and opposite to what it applies to the apple).
        self.axial = (-p.axial_stiffness * d_ax.squeeze(-1)).clamp_min(0.0)  # tension: apple pulled away from the anchor
        lateral = p.lateral_stiffness * torch.linalg.norm(disp_lat, dim=-1)
        self.bending = lateral * self.lever + p.tilt_stiffness * torch.linalg.norm(tilt, dim=-1)
        self.torsion = p.twist_stiffness * twist.squeeze(-1).abs()
        self.damage = (
            (self.axial / self.f_tensile.clamp_min(1e-6)) ** 2
            + (self.bending / self.m_bend_max.clamp_min(1e-6)) ** 2
            + (self.torsion / self.t_twist_max.clamp_min(1e-6)) ** 2
        )
        over = self.attached & (self.damage >= 1.0)
        self.over_steps = torch.where(over, self.over_steps + 1, torch.zeros_like(self.over_steps))
        newly = self.attached & (self.over_steps >= p.hold_steps)
        self.attached &= ~newly

        on = self.attached.unsqueeze(-1).float()
        for buf in (self.axial, self.bending, self.torsion, self.damage):
            buf.mul_(self.attached.float())
        return force * on, torque * on, newly
