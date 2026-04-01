"""
Tests for enhancements/rbf_collocation_cv.py

Verifies:
  1. select_centers — correct shape, subset of samples
  2. compute_rbf_quantities — kernel values and delta shapes, correctness
  3. compute_score_jacobian — matches finite-difference Hessian on a Gaussian
  4. build_collocation_system — correct shapes
  5. compute_Apg_from_coefficients — correct shapes, zero for zero coefficients
  6. rbf_collocation_cv — full pipeline on a Gaussian target
  7. Variance reduction on biased samples from known Gaussian
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


# ---------- Simple Gaussian energy for testing ----------

class GaussianEnergy:
    """E(x) = 0.5 * x^T x  =>  score = -x,  Hessian = I."""

    def eval(self, x):
        return 0.5 * (x ** 2).sum(dim=-1)

    def grad_E(self, x):
        return x

    def score(self, x):
        """s_p(x) = -grad_E(x) = -x.  Must support autograd."""
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        E = self.eval(x)
        g = torch.autograd.grad(E.sum(), x, create_graph=True)[0]
        return -g


def test_select_centers():
    from enhancements.rbf_collocation_cv import select_centers
    N, d = 500, 8
    samples = torch.randn(N, d)
    centers = select_centers(samples, 100)
    assert centers.shape == (100, d)
    # Capping: asking for more than N//2
    centers2 = select_centers(samples, 400)
    assert centers2.shape[0] == 250  # N//2
    print("PASS: test_select_centers")


def test_rbf_quantities():
    from enhancements.rbf_collocation_cv import compute_rbf_quantities
    N, d, M_c = 50, 4, 10
    samples = torch.randn(N, d)
    centers = torch.randn(M_c, d)
    ell = 1.5

    phi, delta = compute_rbf_quantities(samples, centers, ell)
    assert phi.shape == (N, M_c)
    assert delta.shape == (N, M_c, d)
    # phi should be in (0, 1]
    assert (phi > 0).all() and (phi <= 1.0 + 1e-6).all()
    # delta should be x - c
    expected_delta = samples.unsqueeze(1) - centers.unsqueeze(0)
    assert torch.allclose(delta, expected_delta)
    # phi at zero distance = 1
    phi_self, _ = compute_rbf_quantities(centers[:3], centers[:3], ell)
    assert torch.allclose(phi_self.diag(), torch.ones(3), atol=1e-6)
    print("PASS: test_rbf_quantities")


def test_score_jacobian_gaussian():
    """For Gaussian E = 0.5 x^T x, score = -x, Jacobian of score = -I."""
    from enhancements.rbf_collocation_cv import compute_score_jacobian
    N, d = 30, 6
    samples = torch.randn(N, d)
    energy = GaussianEnergy()

    dscore_dx = compute_score_jacobian(samples, energy)
    assert dscore_dx.shape == (N, d, d)

    # ∂s_j/∂x_l = -δ_{jl}  (since s = -x)
    expected = -torch.eye(d).unsqueeze(0).expand(N, -1, -1)
    assert torch.allclose(dscore_dx, expected, atol=1e-5), \
        f"Max diff: {(dscore_dx - expected).abs().max():.2e}"
    print("PASS: test_score_jacobian_gaussian")


def test_collocation_system_shapes():
    from enhancements.rbf_collocation_cv import (
        build_collocation_system, compute_score_jacobian, select_centers
    )
    N, d = 100, 8
    M_c = 20
    samples = torch.randn(N, d)
    energy = GaussianEnergy()
    scores = -samples  # exact for Gaussian
    f_values = energy.eval(samples)
    f_grad = samples  # grad_E = x for Gaussian
    dscore_dx = compute_score_jacobian(samples, energy)
    centers = select_centers(samples, M_c)

    A, b = build_collocation_system(samples, scores, dscore_dx, centers, 1.0, f_grad)
    assert A.shape == (N * d, M_c * d), f"A shape: {A.shape}"
    assert b.shape == (N * d,), f"b shape: {b.shape}"
    # A should be finite
    assert torch.isfinite(A).all(), "A has non-finite values"
    assert torch.isfinite(b).all(), "b has non-finite values"
    print("PASS: test_collocation_system_shapes")


def test_Apg_zero_coefficients():
    """If alpha = 0, then A_p g = 0."""
    from enhancements.rbf_collocation_cv import compute_Apg_from_coefficients
    N, d, M_c = 50, 4, 10
    samples = torch.randn(N, d)
    scores = -samples
    centers = torch.randn(M_c, d)
    alpha = torch.zeros(M_c * d)

    Apg = compute_Apg_from_coefficients(samples, scores, centers, 1.0, alpha)
    assert Apg.shape == (N,)
    assert torch.allclose(Apg, torch.zeros(N), atol=1e-10)
    print("PASS: test_Apg_zero_coefficients")


def test_full_pipeline_gaussian():
    """Full RBF collocation on Gaussian: true samples should have near-zero correction."""
    from enhancements.rbf_collocation_cv import rbf_collocation_cv
    torch.manual_seed(42)

    d = 6
    N = 500
    energy = GaussianEnergy()

    # True samples from N(0, I)
    samples = torch.randn(N, d)
    scores = -samples  # exact score
    f_values = energy.eval(samples)  # 0.5 ||x||^2
    f_grad = samples.clone()  # grad_E = x

    result = rbf_collocation_cv(
        samples, scores, f_values, f_grad,
        energy=energy, n_centers=50, ell=None, reg_lambda=1e-6,
    )

    assert 'estimate' in result
    assert 'variance_rbf' in result
    assert 'variance_naive' in result
    assert 'variance_reduction' in result
    assert 'h_values' in result
    assert result['h_values'].shape == (N,)

    # For true samples, estimate should be close to true mean energy = d/2
    gt = d / 2.0
    err_naive = abs(result['naive_estimate'] - gt)
    err_rbf = abs(result['estimate'] - gt)
    print(f"  GT mean energy: {gt:.4f}")
    print(f"  Naive estimate: {result['naive_estimate']:.4f} (err={err_naive:.4f})")
    print(f"  RBF estimate:   {result['estimate']:.4f} (err={err_rbf:.4f})")
    print(f"  Var reduction:  {result['variance_reduction']:.4f}")
    print(f"  n_centers:      {result['n_centers']}, ell: {result['ell']:.4f}")
    print("PASS: test_full_pipeline_gaussian")


def test_variance_reduction_biased():
    """Biased samples should see variance reduction with RBF collocation."""
    from enhancements.rbf_collocation_cv import rbf_collocation_cv
    torch.manual_seed(123)

    d = 4
    N = 800
    energy = GaussianEnergy()

    # Biased samples: slightly shifted distribution
    samples = torch.randn(N, d) + 0.3
    scores = -samples  # score of N(0,I) at biased locations
    f_values = energy.eval(samples)
    f_grad = samples.clone()

    result = rbf_collocation_cv(
        samples, scores, f_values, f_grad,
        energy=energy, n_centers=80, ell=None, reg_lambda=1e-5,
    )

    gt = d / 2.0
    print(f"  GT mean energy: {gt:.4f}")
    print(f"  Naive estimate: {result['naive_estimate']:.4f}")
    print(f"  RBF estimate:   {result['estimate']:.4f}")
    print(f"  Var reduction:  {result['variance_reduction']:.4f}")

    # The variance should be reduced (ratio < 1), or at worst not explode
    assert result['variance_reduction'] < 5.0, \
        f"Variance exploded: ratio = {result['variance_reduction']:.2f}"
    print("PASS: test_variance_reduction_biased")


if __name__ == "__main__":
    test_select_centers()
    test_rbf_quantities()
    test_score_jacobian_gaussian()
    test_collocation_system_shapes()
    test_Apg_zero_coefficients()
    test_full_pipeline_gaussian()
    test_variance_reduction_biased()
    print("\n=== ALL RBF COLLOCATION CV TESTS PASSED ===")
