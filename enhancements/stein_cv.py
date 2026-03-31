"""
enhancements/stein_cv.py

Stein control variate estimator for variance reduction.

Uses the Control Functionals approach (Oates et al. 2017):
  estimate = 1^T (K_p + λNI)^{-1} f / 1^T (K_p + λNI)^{-1} 1
This is a normalized weighted mean using the inverse Stein kernel.
"""

import torch
from typing import Optional
from enhancements.stein_kernel import stein_kernel_matrix, median_bandwidth


def _solve_and_estimate(
    K_p: torch.Tensor,
    A: torch.Tensor,
    f_values: torch.Tensor,
    N: int,
    L: torch.Tensor = None,
) -> dict:
    """Solve the regularized Stein system and compute the CF estimate.

    Uses the normalized CF estimator:
        a = A^{-1} f,  b = A^{-1} 1
        estimate = sum(a) / sum(b)

    Args:
        K_p: (N, N) Stein kernel matrix
        A: (N, N) regularized matrix K_p + λNI
        f_values: (N,) function values
        N: number of samples
        L: optional Cholesky factor of A

    Returns:
        dict with estimate, naive_estimate, variance_naive, variance_stein
    """
    ones = torch.ones(N, device=f_values.device)

    if L is not None:
        a = torch.cholesky_solve(f_values.unsqueeze(1), L).squeeze(1)
        b = torch.cholesky_solve(ones.unsqueeze(1), L).squeeze(1)
    else:
        a = torch.linalg.solve(A, f_values)
        b = torch.linalg.solve(A, ones)

    # Normalized CF estimate: 1^T a / 1^T b
    estimate = a.sum() / b.sum()

    naive_estimate = f_values.mean()

    # Variance estimates
    variance_naive = f_values.var() / N

    # Stein corrected values for variance estimate
    correction = K_p @ a
    corrected_values = f_values - correction
    variance_stein = corrected_values.var() / N

    return {
        'estimate': estimate.item(),
        'naive_estimate': naive_estimate.item(),
        'variance_naive': variance_naive.item(),
        'variance_stein': variance_stein.item(),
    }


def stein_cv_estimate(
    samples: torch.Tensor,
    scores: torch.Tensor,
    f_values: torch.Tensor,
    ell: Optional[torch.Tensor] = None,
    reg_lambda: float = 1e-4,
) -> dict:
    """Compute the Stein control variate estimator for E_p[f(X)].

    Uses the normalized Control Functionals estimator (Oates et al. 2017):
        a = (K_p + λNI)^{-1} f
        b = (K_p + λNI)^{-1} 1
        estimate = sum(a) / sum(b)

    Args:
        samples: (N, D) terminal samples
        scores: (N, D) scores s_p(x_i) = -grad_E(x_i)
        f_values: (N,) function evaluations f(x_i)
        ell: RBF bandwidth (None = median heuristic)
        reg_lambda: regularization for kernel ridge regression

    Returns:
        dict with keys:
            'estimate': scalar, the Stein CV estimate of E_p[f]
            'naive_estimate': scalar, the vanilla sample mean
            'variance_naive': scalar, variance of naive estimator
            'variance_stein': scalar, estimated variance of Stein estimator
            'coefficients': (N,) optimal coefficients a
    """
    N = samples.shape[0]

    if ell is None:
        ell = median_bandwidth(samples)

    K_p = stein_kernel_matrix(samples, scores, ell)
    A = K_p + reg_lambda * N * torch.eye(N, device=samples.device)

    a = torch.linalg.solve(A, f_values)
    b = torch.linalg.solve(A, torch.ones(N, device=samples.device))

    estimate = a.sum() / b.sum()
    naive_estimate = f_values.mean()

    variance_naive = f_values.var() / N
    correction = K_p @ a
    corrected_values = f_values - correction
    variance_stein = corrected_values.var() / N

    return {
        'estimate': estimate.item(),
        'naive_estimate': naive_estimate.item(),
        'variance_naive': variance_naive.item(),
        'variance_stein': variance_stein.item(),
        'coefficients': a.detach(),
    }


def multi_function_stein_cv(
    samples: torch.Tensor,
    scores: torch.Tensor,
    f_dict: dict,
    ell: Optional[torch.Tensor] = None,
    reg_lambda: float = 1e-4,
) -> dict:
    """Apply Stein CV to multiple functions simultaneously.

    The Stein kernel matrix and its factorization are computed once and reused.

    Args:
        samples: (N, D)
        scores: (N, D)
        f_dict: dict mapping function names to (N,) tensors of evaluations
        ell: bandwidth
        reg_lambda: regularization

    Returns:
        dict mapping function names to result dicts
    """
    N = samples.shape[0]

    if ell is None:
        ell = median_bandwidth(samples)

    K_p = stein_kernel_matrix(samples, scores, ell)
    A = K_p + reg_lambda * N * torch.eye(N, device=samples.device)

    # Cholesky factorize once, solve for multiple RHS
    try:
        L = torch.linalg.cholesky(A)
        use_cholesky = True
    except torch.linalg.LinAlgError:
        use_cholesky = False

    # Solve for the normalizer once (shared across all functions)
    ones = torch.ones(N, device=samples.device)
    if use_cholesky:
        b = torch.cholesky_solve(ones.unsqueeze(1), L).squeeze(1)
    else:
        b = torch.linalg.solve(A, ones)
    b_sum = b.sum()

    results = {}
    for name, f_values in f_dict.items():
        if use_cholesky:
            a = torch.cholesky_solve(f_values.unsqueeze(1), L).squeeze(1)
        else:
            a = torch.linalg.solve(A, f_values)

        estimate = a.sum() / b_sum
        naive_estimate = f_values.mean()
        correction = K_p @ a
        corrected_values = f_values - correction
        results[name] = {
            'estimate': estimate.item(),
            'naive_estimate': naive_estimate.item(),
            'variance_naive': f_values.var().item() / N,
            'variance_stein': corrected_values.var().item() / N,
        }

    return results
