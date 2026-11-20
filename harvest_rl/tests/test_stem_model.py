"""Stem failure criterion: holds the hanging apple, breaks only on a sustained pull/bend/twist."""

import math

import torch

from harvest_rl.g1_harvest.stem import StemModel, StemParams

G = torch.tensor([0.0, 0.0, -9.81])
DT = 1.0 / 120.0


def make(mass=0.16, f_t=17.0, m_b=0.4, t_t=0.05, n=1, k=1, **kw):
    s = StemModel(n, k, StemParams(**kw), "cpu")
    ids = torch.arange(n)
    anchor = torch.tensor([0.0, 0.0, 1.1]).expand(n, k, 3).clone()
    center = torch.tensor([0.0, 0.0, 1.04]).expand(n, k, 3).clone()
    full = lambda v: torch.full((n, k), float(v))  # noqa: E731
    s.set_apples(ids, anchor, center, full(mass), full(0.036), full(0.025), full(f_t), full(m_b), full(t_t), G)
    return s


def simulate(stem, steps, ext_force=None, ext_torque=None, mass=0.16):
    """Point-mass/rigid integration of one apple under gravity + stem (+ external load)."""
    n, k = stem.attached.shape
    pos = stem.center.clone()
    vel = torch.zeros(n, k, 3)
    quat = torch.zeros(n, k, 4)
    quat[..., 3] = 1.0
    ang = torch.zeros(n, k, 3)
    inertia = 0.4 * mass * 0.036**2
    for _ in range(steps):
        f, tq, _ = stem.step(pos, quat, vel, ang)
        f = f + mass * G
        if ext_force is not None:
            f = f + ext_force
        if ext_torque is not None:
            tq = tq + ext_torque
        vel = vel + f / mass * DT
        pos = pos + vel * DT
        ang = ang + tq / inertia * DT
        dq = torch.cat((0.5 * ang * DT, torch.ones(n, k, 1)), -1)
        x1, y1, z1, w1 = dq.unbind(-1)
        x2, y2, z2, w2 = quat.unbind(-1)
        quat = torch.stack((w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2), -1)
        quat = quat / quat.norm(dim=-1, keepdim=True)
    return pos


def test_hangs_still_and_carries_weight():
    s = make()
    pos = simulate(s, 240)
    assert (pos - s.center).norm() < 1e-3
    assert bool(s.attached.all())
    assert abs(s.axial.item() - 0.16 * 9.81) < 0.05


def test_threshold_independent_of_mass():
    # A 300 g apple at rest is far from breaking: weight is part of the load, but thresholds are not mass-derived.
    s = make(mass=0.30)
    simulate(s, 240, mass=0.30)
    assert bool(s.attached.all()) and s.damage.item() < 0.2


def test_sustained_pull_breaks_weak_pull_does_not():
    strong, weak = make(f_t=17.0), make(f_t=17.0)
    simulate(strong, 60, ext_force=torch.tensor([0.0, 0.0, -25.0]))
    simulate(weak, 60, ext_force=torch.tensor([0.0, 0.0, -10.0]))
    assert not bool(strong.attached.any())
    assert bool(weak.attached.all())


def test_sideways_bend_breaks_below_tensile_force():
    # 8 N sideways bends the stem past its moment limit although 8 N straight down would not break it.
    s = make(f_t=17.0, m_b=0.4)
    simulate(s, 60, ext_force=torch.tensor([8.0, 0.0, 0.0]))
    assert not bool(s.attached.any())


def test_wrist_twist_breaks():
    s = make(t_t=0.04)
    simulate(s, 240, ext_torque=torch.tensor([0.0, 0.0, 0.06]))
    assert not bool(s.attached.any())


def test_single_step_impulse_does_not_break():
    s = make(f_t=17.0, hold_steps=3)
    n, k = s.attached.shape
    pos = s.center.clone()
    pos[..., 2] -= 0.05  # one-step 50 N spike
    q = torch.zeros(n, k, 4)
    q[..., 3] = 1
    z = torch.zeros(n, k, 3)
    s.step(pos, q, z, z)
    assert s.damage.item() > 1.0 and bool(s.attached.all())
    s.step(s.center.clone(), q, z, z)  # contact gone
    s.step(s.center.clone(), q, z, z)
    s.step(s.center.clone(), q, z, z)
    assert bool(s.attached.all())


def test_detached_apples_get_no_force():
    s = make()
    simulate(s, 60, ext_force=torch.tensor([0.0, 0.0, -40.0]))
    n, k = s.attached.shape
    q = torch.zeros(n, k, 4)
    q[..., 3] = 1
    f, t, newly = s.step(s.center + 0.1, q, torch.zeros(n, k, 3), torch.zeros(n, k, 3))
    assert f.abs().max() == 0 and t.abs().max() == 0 and not bool(newly.any())


def test_batched_shapes():
    s = make(n=4, k=3)
    pos = simulate(s, 10)
    assert pos.shape == (4, 3, 3) and s.damage.shape == (4, 3)
    assert math.isfinite(float(s.damage.sum()))
