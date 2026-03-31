"""
Test stein_kernel.py with synthetic Gaussians.

For samples from p = N(0, I):
  - score s_p(x) = -x
  - KSD should be near zero for true samples
  - KSD should be positive for biased samples
"""

import sys
sys.path.insert(0, '/home/RESEARCH/adjoint_samplers')

import torch
from enhancements.stein_kernel import (
    median_bandwidth,
    rbf_kernel_matrix,
    stein_kernel_matrix,
    compute_ksd,
)


def test_median_bandwidth():
    print("=== Test: median_bandwidth ===")
    torch.manual_seed(42)
    d = 10
    N = 500
    x = torch.randn(N, d)
    ell = median_bandwidth(x)
    print(f"  Bandwidth (N={N}, d={d}): {ell.item():.4f}")
    # For N(0, I) in d dims, expected median distance ~ sqrt(d) * some_constant
    assert ell.item() > 0, "Bandwidth must be positive"
    print("  PASSED")


def test_rbf_kernel_matrix():
    print("\n=== Test: rbf_kernel_matrix ===")
    torch.manual_seed(42)
    N, M, d = 50, 30, 5
    x = torch.randn(N, d)
    y = torch.randn(M, d)
    ell = torch.tensor(1.0)

    K = rbf_kernel_matrix(x, y, ell)
    assert K.shape == (N, M), f"Wrong shape: {K.shape}"
    assert (K >= 0).all(), "RBF kernel values must be non-negative"
    assert (K <= 1).all(), "RBF kernel values must be <= 1"

    # Diagonal of K(x, x) should be 1
    K_xx = rbf_kernel_matrix(x, x, ell)
    diag = K_xx.diag()
    assert torch.allclose(diag, torch.ones(N), atol=1e-6), "Self-kernel diagonal must be 1"
    print("  Shape, bounds, diagonal checks: PASSED")


def test_stein_kernel_matrix_symmetry():
    print("\n=== Test: stein_kernel_matrix symmetry ===")
    torch.manual_seed(42)
    N, d = 100, 10
    x = torch.randn(N, d)
    scores = -x  # score of N(0, I)
    ell = median_bandwidth(x)

    K_p = stein_kernel_matrix(x, scores, ell)
    assert K_p.shape == (N, N), f"Wrong shape: {K_p.shape}"

    # Stein kernel matrix should be symmetric
    diff = (K_p - K_p.T).abs().max().item()
    print(f"  Max asymmetry: {diff:.2e}")
    assert diff < 1e-5, f"Stein kernel not symmetric: max diff = {diff}"
    print("  PASSED")


def test_ksd_true_samples():
    print("\n=== Test: KSD on true samples from N(0, I) ===")
    torch.manual_seed(42)
    d = 10
    N = 1000

    # True samples from p = N(0, I)
    x = torch.randn(N, d)
    scores = -x  # s_p(x) = -x for p = N(0, I)

    ksd = compute_ksd(x, scores)
    print(f"  KSD^2 (true samples, d={d}, N={N}): {ksd.item():.6f}")
    # For true samples, KSD should be near zero
    # The U-statistic has expectation 0 when q = p, but finite-sample variance
    assert abs(ksd.item()) < 0.1, f"KSD too large for true samples: {ksd.item()}"
    print("  PASSED")


def test_ksd_biased_samples():
    print("\n=== Test: KSD on biased samples ===")
    torch.manual_seed(42)
    d = 10
    N = 1000

    # Biased samples from N(2, I) — NOT from target N(0, I)
    x_biased = torch.randn(N, d) + 2.0
    # Score of target N(0, I) evaluated at biased locations
    scores_at_biased = -x_biased

    ksd_biased = compute_ksd(x_biased, scores_at_biased)
    print(f"  KSD^2 (biased samples, shift=2.0): {ksd_biased.item():.6f}")

    # True samples for comparison
    x_true = torch.randn(N, d)
    scores_true = -x_true
    ksd_true = compute_ksd(x_true, scores_true)
    print(f"  KSD^2 (true samples):              {ksd_true.item():.6f}")

    assert ksd_biased.item() > ksd_true.item(), \
        "Biased KSD should be larger than true KSD"
    assert ksd_biased.item() > 0.1, \
        f"Biased KSD should be clearly positive: {ksd_biased.item()}"
    print("  PASSED: biased KSD >> true KSD")


def test_ksd_scaling_with_bias():
    print("\n=== Test: KSD increases with bias magnitude ===")
    torch.manual_seed(42)
    d = 5
    N = 500

    ksds = []
    shifts = [0.0, 0.5, 1.0, 2.0]
    for shift in shifts:
        x = torch.randn(N, d) + shift
        scores = -x  # score of target N(0, I)
        ksd = compute_ksd(x, scores)
        ksds.append(ksd.item())
        print(f"  shift={shift:.1f}: KSD^2 = {ksd.item():.6f}")

    # KSD should generally increase with shift
    # (allowing some tolerance for the first pair due to noise)
    for i in range(1, len(ksds)):
        if shifts[i] >= 1.0:
            assert ksds[i] > ksds[0], \
                f"KSD at shift={shifts[i]} should exceed KSD at shift=0"
    print("  PASSED: KSD grows with bias")


if __name__ == "__main__":
    test_median_bandwidth()
    test_rbf_kernel_matrix()
    test_stein_kernel_matrix_symmetry()
    test_ksd_true_samples()
    test_ksd_biased_samples()
    test_ksd_scaling_with_bias()
    print("\n✅ ALL TESTS PASSED")
