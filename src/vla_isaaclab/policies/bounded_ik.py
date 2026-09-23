"""Small batched, box-constrained damped least-squares IK steps."""

import torch


def bounded_dls(jacobian, error, lower, upper, damping):
    """Clamp violating joints and re-solve the residual with remaining joints."""
    free = torch.ones_like(lower, dtype=torch.bool)
    fixed = torch.zeros_like(lower)
    eye = torch.eye(jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype)
    for _ in range(jacobian.shape[2] + 1):
        reduced = jacobian * free.unsqueeze(1)
        residual = error - (jacobian @ fixed.unsqueeze(-1)).squeeze(-1)
        solved = torch.linalg.solve(
            reduced @ reduced.transpose(-1, -2) + damping**2 * eye,
            residual.unsqueeze(-1),
        )
        delta = fixed + (reduced.transpose(-1, -2) @ solved).squeeze(-1)
        violations = free & ((delta < lower) | (delta > upper))
        fixed = torch.where(violations, torch.clamp(delta, lower, upper), fixed)
        free = free & ~violations
    return torch.clamp(delta, lower, upper)
