"""
enhancements/stein_kernel.py

Core Stein kernel computations: RBF kernel, Stein kernel, KSD.
All functions operate on terminal samples + precomputed scores.
"""

import torch
from typing import Optional


def median_bandwidth(samples: torch.Tensor) -> torch.Tensor:
    """Compute median heuristic bandwidth for RBF kernel.

    Args:
        samples: (N, D) tensor of sample positions

    Returns:
        ell: scalar tensor, the median pairwise distance
    """
    N = samples.shape[0]

    # Subsample if N is large to avoid O(N^2) memory
    max_subsample = 2000
    if N > max_subsample:
        idx = torch.randperm(N, device=samples.device)[:max_subsample]
        samples = samples[idx]
        N = max_subsample

    # Compute pairwise squared distances (N, N)
    sq_dist = torch.cdist(samples, samples, p=2).pow(2)

    # Extract upper triangle (no diagonal, no duplicates)
    mask = torch.triu(torch.ones(N, N, device=samples.device, dtype=torch.bool), diagonal=1)
    pairwise_dists = sq_dist[mask].sqrt()

    # Median of pairwise distances
    ell = pairwise_dists.median()

    # Guard against zero bandwidth
    if ell.item() < 1e-10:
        ell = torch.tensor(1.0, device=samples.device)

    return ell


def rbf_kernel_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    ell: torch.Tensor,
) -> torch.Tensor:
    """Compute RBF kernel matrix k(x_i, y_j) = exp(-||x_i - y_j||^2 / (2*ell^2)).

    Args:
        x: (N, D) tensor
        y: (M, D) tensor
        ell: scalar bandwidth

    Returns:
        K: (N, M) kernel matrix
    """
    sq_dist = torch.cdist(x, y, p=2).pow(2)
    K = torch.exp(-sq_dist / (2 * ell**2))
    return K


def stein_kernel_matrix(
    samples: torch.Tensor,
    scores: torch.Tensor,
    ell: torch.Tensor,
) -> torch.Tensor:
    """Compute the Stein kernel matrix K_p where (K_p)_{ij} = k_p(x_i, x_j).

    Uses RBF base kernel. The Stein kernel is:
        k_p(x, x') = s(x)^T k s(x') + s(x)^T ∇_{x'} k + (∇_x k)^T s(x') + tr(∇_x ∇_{x'} k)

    For RBF k(x,x') = exp(-||x-x'||^2 / (2*ell^2)):
        ∇_{x'} k = (x - x') / ell^2 * k
        ∇_x k    = -(x - x') / ell^2 * k
        tr(∇_x ∇_{x'} k) = (d/ell^2 - ||x-x'||^2/ell^4) * k

    So: k_p = k * [s(x)^T s(x') + s(x)^T(x-x')/ell^2 - (x-x')^T s(x')/ell^2
                    + d/ell^2 - ||x-x'||^2/ell^4]

    Args:
        samples: (N, D) tensor of sample positions
        scores: (N, D) tensor of scores s_p(x_i) = -grad_E(x_i)
        ell: scalar bandwidth

    Returns:
        K_p: (N, N) Stein kernel matrix
    """
    N, D = samples.shape

    # Pairwise difference: diff[i,j] = x_i - x_j, shape (N, N, D)
    diff = samples.unsqueeze(1) - samples.unsqueeze(0)

    # Squared distances: sq_dist[i,j] = ||x_i - x_j||^2, shape (N, N)
    sq_dist = (diff**2).sum(-1)

    # RBF kernel: K[i,j] = exp(-sq_dist / (2*ell^2))
    K = torch.exp(-sq_dist / (2 * ell**2))

    ell2 = ell**2

    # Term 1: s_i^T s_j * K[i,j]
    # (scores @ scores.T) is (N, N), element [i,j] = s_i^T s_j
    term1 = (scores @ scores.T) * K

    # Term 2: s_i^T ∇_{x'} k = s_i^T (x_i - x_j) / ell^2 * K[i,j]
    # diff[i,j] = x_i - x_j, so s_i^T diff[i,j] = sum_d scores[i,d] * diff[i,j,d]
    term2 = torch.einsum('id,ijd->ij', scores, diff) / ell2 * K

    # Term 3: (∇_x k)^T s_j = -(x_i - x_j)^T s_j / ell^2 * K[i,j]
    # diff[i,j] = x_i - x_j, so -(x_i - x_j)^T s_j = -sum_d diff[i,j,d] * scores[j,d]
    term3 = -torch.einsum('ijd,jd->ij', diff, scores) / ell2 * K

    # Term 4: (d/ell^2 - ||x_i - x_j||^2/ell^4) * K[i,j]
    term4 = (D / ell2 - sq_dist / ell2**2) * K

    K_p = term1 + term2 + term3 + term4
    return K_p


def compute_ksd(
    samples: torch.Tensor,
    scores: torch.Tensor,
    ell: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the U-statistic estimator of KSD^2.

    Args:
        samples: (N, D) tensor
        scores: (N, D) tensor of s_p(x_i)
        ell: bandwidth (if None, use median heuristic)

    Returns:
        ksd_squared: scalar tensor
    """
    N = samples.shape[0]

    # 1. Compute bandwidth if not provided
    if ell is None:
        ell = median_bandwidth(samples)

    # 2. Compute Stein kernel matrix K_p
    K_p = stein_kernel_matrix(samples, scores, ell)

    # 3. KSD^2 = (1/(N*(N-1))) * (sum(K_p) - trace(K_p))
    #    The subtraction of the trace removes diagonal terms (U-statistic)
    ksd_squared = (K_p.sum() - K_p.trace()) / (N * (N - 1))

    return ksd_squared
