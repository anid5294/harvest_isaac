"""Small, deterministic box-constrained DLS solves without simulator imports."""

from __future__ import annotations

import torch


@torch.no_grad()
def bounded_dls(jacobian, error, lower, upper, *, damping=0.05,
                posture_delta=None, max_iterations=32, tolerance=None):
    """Return ``(delta, kkt_residual)`` for a single box-constrained IK solve.

    Minimize ``||J delta-error||² + damping²||delta-posture_delta||²``.
    The primal active-set method solves each free subproblem exactly, steps to
    blocking bounds, and releases bounds with incorrect multiplier signs.
    ``kkt_residual`` is the infinity norm of the bound-aware gradient of half
    this objective; fixed coordinates contribute zero. A residual above
    ``tolerance`` exposes iteration exhaustion rather than claiming convergence.
    Inputs share a floating dtype/device; supports arbitrary small joint counts.
    """
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    m, n = jacobian.shape
    if error.shape != (m,) or lower.shape != (n,) or upper.shape != (n,):
        raise ValueError("error and bounds must match jacobian dimensions")
    if posture_delta is None:
        posture_delta = torch.zeros_like(lower)
    if posture_delta.shape != (n,):
        raise ValueError("posture_delta must match joint dimensions")
    tensors = (jacobian, error, lower, upper, posture_delta)
    if not jacobian.is_floating_point() or any(
        t.dtype != jacobian.dtype or t.device != jacobian.device for t in tensors
    ):
        raise ValueError("inputs must share a floating dtype and device")
    if not all(bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError("inputs must be finite")
    if bool((lower > upper).any()) or damping < 0 or not torch.isfinite(torch.tensor(damping)):
        raise ValueError("bounds must be ordered and damping finite/nonnegative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    tol = (1e-6 if jacobian.dtype == torch.float32 else 1e-10) if tolerance is None else tolerance
    if tol <= 0:
        raise ValueError("tolerance must be positive")
    hessian = jacobian.T @ jacobian + damping**2 * torch.eye(
        n, dtype=jacobian.dtype, device=jacobian.device)
    rhs = jacobian.T @ error + damping**2 * posture_delta
    delta = torch.clamp(posture_delta, lower, upper).clone()
    fixed = lower == upper
    # -1: lower bound, +1: upper bound, 0: free; fixed bounds never release.
    active = torch.zeros(n, dtype=torch.int8, device=jacobian.device)
    active[delta == lower] = -1
    active[delta == upper] = 1

    def violations(gradient):
        return torch.where(fixed, torch.zeros_like(gradient), torch.where(
            active == -1, (-gradient).clamp(min=0), torch.where(
                active == 1, gradient.clamp(min=0), gradient.abs())))

    for _ in range(max_iterations):
        free = active == 0
        if bool(free.any()):
            indices = torch.where(free)[0]
            hff = hessian[indices][:, indices]
            reduced_rhs = rhs[free] - hessian[indices][:, ~free] @ delta[~free]
            optimum = (torch.linalg.solve(hff, reduced_rhs) if damping > 0 else
                       torch.linalg.lstsq(hff, reduced_rhs.unsqueeze(-1)).solution.squeeze(-1))
            direction = optimum - delta[free]
            ratios = torch.full_like(direction, float("inf"))
            positive, negative = direction > 0, direction < 0
            ratios[positive] = (upper[free][positive] - delta[free][positive]) / direction[positive]
            ratios[negative] = (lower[free][negative] - delta[free][negative]) / direction[negative]
            alpha = ratios.min().clamp(min=0, max=1)
            delta[free] += alpha * direction
            delta = torch.clamp(delta, lower, upper)
            if bool(alpha < 1):
                blocking = ratios == ratios.min()
                blocked_indices = indices[blocking]
                active[blocked_indices] = torch.where(direction[blocking] > 0, 1, -1).to(active.dtype)
                delta[blocked_indices] = torch.where(direction[blocking] > 0,
                                                     upper[blocked_indices], lower[blocked_indices])
                continue
        gradient = hessian @ delta - rhs
        violation = violations(gradient)
        if bool(violation.max() <= tol):
            break
        releasable = (active != 0) & ~fixed
        release_violation = torch.where(releasable, violation, torch.zeros_like(violation))
        if bool(release_violation.max() <= tol):
            break
        active[release_violation.argmax()] = 0
    return delta, violations(hessian @ delta - rhs).max()
