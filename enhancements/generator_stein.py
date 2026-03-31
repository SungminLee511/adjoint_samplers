"""
enhancements/generator_stein.py

SDE generator-based Stein kernel: uses the learned drift b_theta
instead of the generic score s_p for potentially better control variates.

Uses the practical simplification (diffusion-free approximation) to avoid
computing Jacobians of b_theta:
    k_gen(x, x') = b(x)^T k(x,x') b(x')
                  + b(x)^T ∇_{x'} k(x,x')
                  + (∇_x k)^T b(x')
                  + (g(1)^2 / 2) * tr(∇_x ∇_{x'} k(x,x'))
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from adjoint_samplers.components.sde import ControlledSDE

from enhancements.stein_kernel import median_bandwidth


def generator_stein_kernel_matrix(
    samples: torch.Tensor,
    sde: "ControlledSDE",
    ell: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Stein kernel using the SDE generator instead of generic score.

    Replaces s_p(x) with b_theta(x, 1) in the Stein kernel formula,
    and uses g(1)^2/2 as the diffusion coefficient in the trace term.

    Args:
        samples: (N, D) terminal samples
        sde: the trained ControlledSDE
        ell: bandwidth (None = median heuristic)

    Returns:
        K_gen: (N, N) generator Stein kernel matrix
    """
    N, D = samples.shape

    if ell is None:
        ell = median_bandwidth(samples)

    # Get the total drift at t=1 for each sample
    t1 = torch.ones(N, 1, device=samples.device)
    with torch.no_grad():
        drift = sde.drift(t1, samples)  # (N, D) = f(x,1) + g(1)^2 * u(x,1)
    diff_sq = sde.diff(t1[0]).item() ** 2  # g(1)^2, scalar

    # Compute pairwise differences
    diff = samples.unsqueeze(1) - samples.unsqueeze(0)  # (N, N, D)
    sq_dist = (diff ** 2).sum(-1)  # (N, N)

    # RBF kernel
    ell2 = ell ** 2
    K = torch.exp(-sq_dist / (2 * ell2))

    # Term 1: drift_i^T drift_j * K
    term1 = (drift @ drift.T) * K

    # Term 2: drift_i^T ∇_{x'} k = drift_i^T (x_i - x_j) / ell^2 * K
    # diff[i,j] = x_i - x_j
    term2 = torch.einsum('id,ijd->ij', drift, diff) / ell2 * K

    # Term 3: (∇_x k)^T drift_j = -(x_i - x_j)^T drift_j / ell^2 * K
    term3 = -torch.einsum('ijd,jd->ij', diff, drift) / ell2 * K

    # Term 4: (g(1)^2 / 2) * (d/ell^2 - sq_dist/ell^4) * K
    # Note: factor of g(1)^2/2 from the diffusion part of the generator
    term4 = (diff_sq / 2) * (D / ell2 - sq_dist / ell2 ** 2) * K

    K_gen = term1 + term2 + term3 + term4
    return K_gen


def generator_stein_cv_estimate(
    samples: torch.Tensor,
    sde: "ControlledSDE",
    f_values: torch.Tensor,
    ell: Optional[torch.Tensor] = None,
    reg_lambda: float = 1e-4,
) -> dict:
    """Stein CV using the SDE generator kernel.

    Same as stein_cv_estimate but with the generator kernel
    (uses learned drift instead of generic score).

    Args:
        samples: (N, D)
        sde: trained ControlledSDE
        f_values: (N,)
        ell: bandwidth
        reg_lambda: regularization

    Returns:
        dict with 'estimate', 'naive_estimate', 'variance_naive', 'variance_gen_stein'
    """
    N = samples.shape[0]

    if ell is None:
        ell = median_bandwidth(samples)

    K_gen = generator_stein_kernel_matrix(samples, sde, ell)
    A = K_gen + reg_lambda * N * torch.eye(N, device=samples.device)

    ones = torch.ones(N, device=samples.device)
    a = torch.linalg.solve(A, f_values)
    b = torch.linalg.solve(A, ones)

    # Normalized CF estimate
    estimate = a.sum() / b.sum()
    naive_estimate = f_values.mean()

    correction = K_gen @ a
    corrected_values = f_values - correction

    return {
        'estimate': estimate.item(),
        'naive_estimate': naive_estimate.item(),
        'variance_naive': f_values.var().item() / N,
        'variance_gen_stein': corrected_values.var().item() / N,
    }
