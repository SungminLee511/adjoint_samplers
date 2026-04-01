"""
enhancements/rbf_collocation_cv.py

RBF Collocation Stein control variate — solves the differentiated Poisson
equation ∇_x(f + A_p g) = 0 via mesh-free Gaussian RBF collocation.

Instead of training a neural network iteratively, we expand g in a Gaussian
RBF basis and solve the resulting **linear least-squares** problem in one
matrix factorization.  No epochs, no learning rate, no gradient instability.

Best for low-to-medium dimensions (d ≤ 50): DW4 (8D), LJ13 (39D).
For LJ55 (165D) the basis size is too large — use Neural Stein CV instead.

Reference: MATH_SPEC Section 7C.
"""

import torch
from typing import Optional


def select_centers(
    samples: torch.Tensor,
    n_centers: int,
) -> torch.Tensor:
    """Select RBF centers from samples via random subsampling.

    Args:
        samples: (N, d) all samples
        n_centers: number of centers M_c (capped at N//2)
    Returns:
        centers: (M_c, d)
    """
    N = samples.shape[0]
    n_centers = min(n_centers, N // 2)
    idx = torch.randperm(N, device=samples.device)[:n_centers]
    return samples[idx].detach().clone()


def compute_rbf_quantities(
    samples: torch.Tensor,
    centers: torch.Tensor,
    ell: float,
) -> tuple:
    """Compute RBF kernel values and analytical derivatives at all sample points.

    For φ_m(x) = exp(-||x - c_m||² / (2ℓ²)), with δ = x - c_m:
      φ                     : (N, M_c)
      ∂φ/∂x_l              = -(δ_l / ℓ²) · φ           : accessed via delta and phi
      ∂²φ/(∂x_l ∂x_j)      = (δ_l δ_j / ℓ⁴ - δ_{lj}/ℓ²) · φ

    Args:
        samples: (N, d)
        centers: (M_c, d)
        ell: bandwidth
    Returns:
        phi: (N, M_c) kernel evaluations
        delta: (N, M_c, d) = samples[:, None, :] - centers[None, :, :]
    """
    # delta[i, m, l] = x_{i,l} - c_{m,l}
    delta = samples.unsqueeze(1) - centers.unsqueeze(0)   # (N, M_c, d)
    sq_dist = (delta ** 2).sum(dim=-1)                     # (N, M_c)
    phi = torch.exp(-sq_dist / (2.0 * ell ** 2))          # (N, M_c)
    return phi, delta


def compute_score_jacobian(
    samples: torch.Tensor,
    energy,
) -> torch.Tensor:
    """Compute the Jacobian of the score ∂s_j/∂x_l = -H_{lj} for all samples.

    Uses d backward passes (one per dimension).

    NOTE: BaseEnergy.score() / grad_E() internally detaches and uses
    create_graph=False, so we cannot backprop through it.  Instead we
    call energy.eval(x) directly with create_graph=True to get the
    full Hessian.

    Args:
        samples: (N, d) — will be cloned with requires_grad
        energy: energy object with eval() method
    Returns:
        dscore_dx: (N, d, d) where dscore_dx[i, l, j] = ∂s_j/∂x_l at x_i
                   = -H_{lj}  (negative energy Hessian)
    """
    N, d = samples.shape
    x = samples.detach().requires_grad_(True)

    # Compute grad_E with create_graph=True so we can differentiate again
    with torch.enable_grad():
        E = energy.eval(x)  # (N,)
        grad_E = torch.autograd.grad(
            E.sum(), x, create_graph=True,
        )[0]  # (N, d), has grad_fn

    # score = -grad_E, so ∂s_j/∂x_l = -∂²E/∂x_l∂x_j
    # Jacobian column-by-column
    dscore_dx = torch.zeros(N, d, d, device=samples.device)
    for j in range(d):
        grad2_j = torch.autograd.grad(
            grad_E[:, j].sum(), x,
            create_graph=False, retain_graph=(j < d - 1),
        )[0]  # (N, d) — this is ∂²E/∂x_l∂x_j for all l
        dscore_dx[:, :, j] = -grad2_j  # ∂s_j/∂x_l = -∂²E/∂x_l∂x_j

    return dscore_dx.detach()


def build_collocation_system(
    samples: torch.Tensor,
    scores: torch.Tensor,
    dscore_dx: torch.Tensor,
    centers: torch.Tensor,
    ell: float,
    f_grad: torch.Tensor,
) -> tuple:
    """Build the collocation matrix A and RHS b for the linear system.

    For basis function ψ_{m,j} (center m, output dim j), the collocation
    matrix entry for row (i, l) is:

        A_{(i,l), (m,j)} = ∂/∂x_l [A_p ψ_{m,j}](x_i)
            = (∂s_j/∂x_l) · φ_m(x_i)
            + s_j(x_i) · (∂φ_m/∂x_l)(x_i)
            + (∂²φ_m / ∂x_l ∂x_j)(x_i)

    RHS:  b_{(i,l)} = -∂f/∂x_l(x_i)

    Args:
        samples: (N, d)
        scores: (N, d) precomputed s_p(x)
        dscore_dx: (N, d, d) Jacobian of score, [i,l,j] = ∂s_j/∂x_l
        centers: (M_c, d)
        ell: bandwidth
        f_grad: (N, d) gradient of f at each sample
    Returns:
        A: (N*d, M_c*d) collocation matrix
        b: (N*d,) right-hand side
    """
    N, d = samples.shape
    M_c = centers.shape[0]
    M = M_c * d  # total basis functions

    phi, delta = compute_rbf_quantities(samples, centers, ell)
    # phi: (N, M_c), delta: (N, M_c, d)

    ell2 = ell ** 2
    ell4 = ell ** 4

    # Precompute ∂φ_m/∂x_l = -(δ_l / ℓ²) · φ_m
    # dphi_dx[i, m, l] = -(delta[i,m,l] / ell2) * phi[i,m]
    dphi_dx = -(delta / ell2) * phi.unsqueeze(-1)  # (N, M_c, d)

    # Build A: shape (N*d, M_c*d)
    # Row index: i*d + l  (sample i, derivative dim l)
    # Col index: m*d + j  (center m, output dim j)
    #
    # For efficiency, build as (N, d, M_c, d) then reshape.
    # A[i, l, m, j] = dscore_dx[i,l,j] * phi[i,m]
    #               + scores[i,j] * dphi_dx[i,m,l]
    #               + d2phi[i,m,l,j]

    # Term 1: (∂s_j/∂x_l) · φ_m(x_i)
    # dscore_dx: (N, d_l, d_j), phi: (N, M_c)
    # -> (N, d_l, M_c, d_j)
    term1 = dscore_dx.unsqueeze(2) * phi.unsqueeze(1).unsqueeze(-1)
    # dscore_dx[:, :, None, :] is (N, d, 1, d), phi[:, None, :, None] is (N, 1, M_c, 1)

    # Term 2: s_j(x_i) · (∂φ_m/∂x_l)(x_i)
    # scores: (N, d_j), dphi_dx: (N, M_c, d_l)
    # -> (N, d_l, M_c, d_j)
    term2 = dphi_dx.permute(0, 2, 1).unsqueeze(-1) * scores.unsqueeze(1).unsqueeze(1)
    # dphi_dx permuted to (N, d_l, M_c), then (N, d_l, M_c, 1)
    # scores reshaped to (N, 1, 1, d_j)

    # Term 3: ∂²φ_m/(∂x_l ∂x_j)
    # = (δ_l δ_j / ℓ⁴ - δ_{lj}/ℓ²) · φ_m
    # delta: (N, M_c, d)
    # outer product δ_l δ_j: (N, M_c, d, d) via delta[..., :, None] * delta[..., None, :]
    delta_outer = delta.unsqueeze(-1) * delta.unsqueeze(-2)  # (N, M_c, d_l, d_j)
    eye_d = torch.eye(d, device=samples.device)  # (d, d)
    d2phi = (delta_outer / ell4 - eye_d.unsqueeze(0).unsqueeze(0) / ell2) * phi.unsqueeze(-1).unsqueeze(-1)
    # d2phi: (N, M_c, d_l, d_j)
    # Rearrange to (N, d_l, M_c, d_j)
    d2phi = d2phi.permute(0, 2, 1, 3)

    A_4d = term1 + term2 + d2phi  # (N, d_l, M_c, d_j)

    # Reshape to (N*d, M_c*d)
    A = A_4d.reshape(N * d, M_c * d)

    # RHS: b_{(i,l)} = -f_grad[i, l]
    b = -f_grad.reshape(N * d)

    return A, b


def compute_Apg_from_coefficients(
    samples: torch.Tensor,
    scores: torch.Tensor,
    centers: torch.Tensor,
    ell: float,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Compute A_p g(x_i) for all samples using the solved coefficients.

    A_p g(x_i) = Σ_{m,j} α_{m,j} · [s_j(x_i) · φ_m(x_i) + ∂φ_m/∂x_j(x_i)]

    Args:
        samples: (N, d)
        scores: (N, d)
        centers: (M_c, d)
        ell: bandwidth
        alpha: (M_c * d,) solved coefficients
    Returns:
        Apg: (N,)
    """
    N, d = samples.shape
    M_c = centers.shape[0]

    phi, delta = compute_rbf_quantities(samples, centers, ell)
    ell2 = ell ** 2

    # α reshaped to (M_c, d)
    alpha_2d = alpha.reshape(M_c, d)

    # Term 1: Σ_{m,j} α_{m,j} · s_j(x_i) · φ_m(x_i)
    # scores: (N, d), alpha_2d: (M_c, d), phi: (N, M_c)
    # = Σ_m φ_m(x_i) · Σ_j α_{m,j} · s_j(x_i)
    # inner: (N, M_c) via scores @ alpha_2d.T -> but that's (N, M_c)
    # Actually: (scores (N,d)) @ (alpha_2d (M_c, d)).T  = (N, M_c) — dot for each m
    sa = scores @ alpha_2d.T  # (N, M_c): sa[i,m] = Σ_j s_j(x_i) α_{m,j}
    term1 = (phi * sa).sum(dim=1)  # (N,)

    # Term 2: Σ_{m,j} α_{m,j} · ∂φ_m/∂x_j(x_i)
    # ∂φ_m/∂x_j = -(δ_j / ℓ²) · φ_m
    # = -Σ_{m,j} α_{m,j} · (δ_j / ℓ²) · φ_m
    # delta: (N, M_c, d), phi: (N, M_c)
    # dphi_dx_j for each j: -(delta[:, :, j] / ell2) * phi -> (N, M_c)
    # Sum over m,j with weights α_{m,j}:
    # = -Σ_m φ_m · Σ_j α_{m,j} · δ_j / ℓ²
    # delta @ alpha_2d.T is wrong shape. Need per-m dot:
    # Σ_j α_{m,j} · δ_{i,m,j} = (delta (N,M_c,d)) · (alpha_2d (M_c,d)) summed over d
    da = (delta * alpha_2d.unsqueeze(0)).sum(dim=-1)  # (N, M_c)
    term2 = -(phi * da).sum(dim=1) / ell2  # (N,)

    return term1 + term2


def rbf_collocation_cv(
    samples: torch.Tensor,
    scores: torch.Tensor,
    f_values: torch.Tensor,
    f_grad: torch.Tensor,
    energy=None,
    n_centers: int = 200,
    ell: Optional[float] = None,
    reg_lambda: float = 1e-6,
) -> dict:
    """Solve the Stein CV via RBF collocation (single least-squares solve).

    Args:
        samples: (N, d) terminal samples
        scores: (N, d) = s_p(x_i) = -grad_E(x_i)
        f_values: (N,) function evaluations f(x_i)
        f_grad: (N, d) gradient of f at each sample
        energy: energy object with score() method (needed for Hessian computation)
        n_centers: number of RBF centers M_c
        ell: RBF bandwidth (None = median heuristic)
        reg_lambda: Tikhonov regularization strength
    Returns:
        dict with 'estimate', 'variance_rbf', 'variance_naive',
        'variance_reduction', 'coefficients', 'h_values'
    """
    N, d = samples.shape
    device = samples.device

    # 1. Select centers
    centers = select_centers(samples, n_centers)
    M_c = centers.shape[0]

    # 2. Bandwidth (median heuristic if not provided)
    if ell is None:
        from enhancements.stein_kernel import median_bandwidth
        ell = median_bandwidth(samples).item()

    # 3. Compute score Jacobian (energy Hessian) via autograd
    if energy is not None:
        dscore_dx = compute_score_jacobian(samples, energy)
    else:
        # Fallback: finite differences (less accurate but doesn't need energy)
        raise ValueError("energy is required for Hessian computation")

    # 4. Build collocation system
    A, b = build_collocation_system(
        samples, scores, dscore_dx, centers, ell, f_grad
    )

    # 5. Solve regularized least-squares: min ||Aα - b||² + λ||α||²
    M = M_c * d
    AtA = A.T @ A + reg_lambda * torch.eye(M, device=device)
    Atb = A.T @ b
    alpha = torch.linalg.solve(AtA, Atb)  # (M,)

    # 6. Compute h(x_i) = f(x_i) + A_p g(x_i)
    Apg = compute_Apg_from_coefficients(samples, scores, centers, ell, alpha)
    h_values = f_values + Apg

    estimate = h_values.mean().item()
    naive_estimate = f_values.mean().item()
    var_h = h_values.var().item()
    var_f = f_values.var().item()

    return {
        'estimate': estimate,
        'naive_estimate': naive_estimate,
        'variance_rbf': var_h / N,
        'variance_naive': var_f / N,
        'variance_reduction': var_h / (var_f + 1e-20),
        'coefficients': alpha.detach(),
        'h_values': h_values.detach(),
        'n_centers': M_c,
        'ell': ell,
    }
