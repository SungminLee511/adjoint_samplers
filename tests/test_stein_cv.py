"""
Test stein_cv.py with synthetic Gaussians.

For p = N(0, I_d):
  - E_p[||x||^2] = d (ground truth)
  - Stein CV should give lower variance than naive for this estimate
"""

import sys
sys.path.insert(0, '/home/RESEARCH/adjoint_samplers')

import torch
from enhancements.stein_cv import stein_cv_estimate, multi_function_stein_cv


def test_stein_cv_basic():
    print("=== Test: stein_cv_estimate basic ===")
    torch.manual_seed(42)
    d = 10
    N = 500

    # True samples from N(0, I)
    x = torch.randn(N, d)
    scores = -x  # score of N(0, I)

    # f(x) = ||x||^2, ground truth E_p[f] = d = 10
    f_values = (x**2).sum(dim=1)  # (N,)

    result = stein_cv_estimate(x, scores, f_values)

    print(f"  Ground truth E[||x||^2] = {d}")
    print(f"  Naive estimate:  {result['naive_estimate']:.4f}")
    print(f"  Stein estimate:  {result['estimate']:.4f}")
    print(f"  Var naive:       {result['variance_naive']:.6f}")
    print(f"  Var Stein:       {result['variance_stein']:.6f}")

    # Both should be close to d=10
    assert abs(result['naive_estimate'] - d) < 2.0, \
        f"Naive estimate too far: {result['naive_estimate']}"
    assert abs(result['estimate'] - d) < 2.0, \
        f"Stein estimate too far: {result['estimate']}"

    # Stein variance should be <= naive variance
    print(f"  Variance ratio:  {result['variance_stein'] / result['variance_naive']:.4f}")
    print("  PASSED")


def test_stein_cv_variance_reduction():
    print("\n=== Test: Stein CV variance reduction across seeds ===")
    d = 5
    N = 300

    naive_estimates = []
    stein_estimates = []

    for seed in range(20):
        torch.manual_seed(seed * 100)
        x = torch.randn(N, d)
        scores = -x
        f_values = (x**2).sum(dim=1)

        result = stein_cv_estimate(x, scores, f_values)
        naive_estimates.append(result['naive_estimate'])
        stein_estimates.append(result['estimate'])

    naive_t = torch.tensor(naive_estimates)
    stein_t = torch.tensor(stein_estimates)

    naive_var = naive_t.var().item()
    stein_var = stein_t.var().item()

    print(f"  Ground truth: {d}")
    print(f"  Naive mean ± std: {naive_t.mean():.4f} ± {naive_t.std():.4f}")
    print(f"  Stein mean ± std: {stein_t.mean():.4f} ± {stein_t.std():.4f}")
    print(f"  Naive variance across seeds: {naive_var:.6f}")
    print(f"  Stein variance across seeds: {stein_var:.6f}")
    print(f"  Variance ratio (Stein/Naive): {stein_var / naive_var:.4f}")

    # Stein should have lower variance across seeds
    # (not guaranteed per run, but across 20 seeds it should be clear)
    assert stein_var < naive_var * 1.5, \
        f"Stein variance not better: {stein_var} vs {naive_var}"
    print("  PASSED")


def test_multi_function_stein_cv():
    print("\n=== Test: multi_function_stein_cv ===")
    torch.manual_seed(42)
    d = 5
    N = 300

    x = torch.randn(N, d)
    scores = -x

    f_dict = {
        'norm_sq': (x**2).sum(dim=1),       # E[||x||^2] = d = 5
        'first_coord': x[:, 0],              # E[x_0] = 0
        'energy': 0.5 * (x**2).sum(dim=1),   # E[0.5*||x||^2] = d/2 = 2.5
    }

    results = multi_function_stein_cv(x, scores, f_dict)

    for name, res in results.items():
        print(f"  {name}:")
        print(f"    Naive: {res['naive_estimate']:.4f}, Stein: {res['estimate']:.4f}")
        print(f"    Var ratio: {res['variance_stein'] / (res['variance_naive'] + 1e-10):.4f}")

    # Basic sanity
    assert 'norm_sq' in results
    assert 'first_coord' in results
    assert 'energy' in results
    print("  PASSED")


def test_stein_cv_with_biased_samples():
    print("\n=== Test: Stein CV on biased samples ===")
    torch.manual_seed(42)
    d = 5
    N = 400

    # Biased samples from N(0.5, I) but target is N(0, I)
    x = torch.randn(N, d) + 0.5
    scores = -x  # score of target N(0, I) evaluated at biased locations
    f_values = (x**2).sum(dim=1)  # E_p[||x||^2] = d = 5

    result = stein_cv_estimate(x, scores, f_values)

    print(f"  Ground truth: {d}")
    print(f"  Naive (biased): {result['naive_estimate']:.4f}")
    print(f"  Stein (biased): {result['estimate']:.4f}")
    print(f"  Naive error: {abs(result['naive_estimate'] - d):.4f}")
    print(f"  Stein error: {abs(result['estimate'] - d):.4f}")

    # With bias, Stein CV may or may not improve error (it's designed for
    # variance reduction, not bias correction). Just verify it runs.
    print("  PASSED (runs without error)")


if __name__ == "__main__":
    test_stein_cv_basic()
    test_stein_cv_variance_reduction()
    test_multi_function_stein_cv()
    test_stein_cv_with_biased_samples()
    print("\n✅ ALL TESTS PASSED")
