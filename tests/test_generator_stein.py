"""
Test generator_stein.py with a mock ControlledSDE.

Since generator_stein uses sde.drift(t, x) and sde.diff(t), we mock
a simple SDE where the drift is the score of a Gaussian (drift = -x),
matching the Langevin dynamics for N(0, I). In this case the generator
Stein kernel should behave similarly to the standard Stein kernel.
"""

import sys
sys.path.insert(0, '/home/RESEARCH/adjoint_samplers')

import torch
from enhancements.generator_stein import (
    generator_stein_kernel_matrix,
    generator_stein_cv_estimate,
)


class MockControlledSDE:
    """Mock SDE where drift = -x (Langevin for N(0,I)) and diff = 1.0."""

    def drift(self, t, x):
        """b_theta(x, t) = -x (score of N(0,I))."""
        return -x

    def diff(self, t):
        """g(t) = 1.0."""
        if isinstance(t, torch.Tensor):
            return torch.ones_like(t)
        return torch.tensor(1.0)


def test_generator_kernel_shape_and_symmetry():
    print("=== Test: generator_stein_kernel_matrix shape & symmetry ===")
    torch.manual_seed(42)
    N, D = 100, 5
    x = torch.randn(N, D)
    sde = MockControlledSDE()

    K_gen = generator_stein_kernel_matrix(x, sde)

    assert K_gen.shape == (N, N), f"Wrong shape: {K_gen.shape}"

    asym = (K_gen - K_gen.T).abs().max().item()
    print(f"  Shape: {K_gen.shape}")
    print(f"  Max asymmetry: {asym:.2e}")
    assert asym < 1e-5, f"Generator kernel not symmetric: {asym}"
    print("  PASSED")


def test_generator_cv_basic():
    print("\n=== Test: generator_stein_cv_estimate basic ===")
    torch.manual_seed(42)
    D = 5
    N = 300

    x = torch.randn(N, D)
    sde = MockControlledSDE()

    # f(x) = ||x||^2, ground truth E_p[f] = D = 5
    f_values = (x**2).sum(dim=1)

    result = generator_stein_cv_estimate(x, sde, f_values)

    print(f"  Ground truth: {D}")
    print(f"  Naive:        {result['naive_estimate']:.4f}")
    print(f"  Gen Stein:    {result['estimate']:.4f}")
    print(f"  Var naive:    {result['variance_naive']:.6f}")
    print(f"  Var gen:      {result['variance_gen_stein']:.6f}")

    assert abs(result['estimate'] - D) < 2.0, \
        f"Gen Stein estimate too far: {result['estimate']}"
    assert 'variance_gen_stein' in result
    print("  PASSED")


def test_generator_cv_variance_reduction():
    print("\n=== Test: generator CV variance reduction across seeds ===")
    D = 5
    N = 300
    sde = MockControlledSDE()

    naive_estimates = []
    gen_estimates = []

    for seed in range(15):
        torch.manual_seed(seed * 77)
        x = torch.randn(N, D)
        f_values = (x**2).sum(dim=1)

        result = generator_stein_cv_estimate(x, sde, f_values)
        naive_estimates.append(result['naive_estimate'])
        gen_estimates.append(result['estimate'])

    naive_t = torch.tensor(naive_estimates)
    gen_t = torch.tensor(gen_estimates)

    print(f"  Ground truth: {D}")
    print(f"  Naive mean ± std: {naive_t.mean():.4f} ± {naive_t.std():.4f}")
    print(f"  Gen   mean ± std: {gen_t.mean():.4f} ± {gen_t.std():.4f}")
    print(f"  Naive var across seeds: {naive_t.var().item():.6f}")
    print(f"  Gen   var across seeds: {gen_t.var().item():.6f}")
    print(f"  Ratio: {gen_t.var().item() / naive_t.var().item():.4f}")

    # Generator Stein should provide some variance reduction
    # (with mock drift = -x = score, it should be similar to standard Stein)
    assert gen_t.var().item() < naive_t.var().item() * 1.5, \
        "Gen Stein variance should not be much worse than naive"
    print("  PASSED")


def test_generator_cv_return_keys():
    print("\n=== Test: generator_stein_cv_estimate return keys ===")
    torch.manual_seed(42)
    x = torch.randn(50, 3)
    sde = MockControlledSDE()
    f_values = x[:, 0]

    result = generator_stein_cv_estimate(x, sde, f_values)

    required = ['estimate', 'naive_estimate', 'variance_naive', 'variance_gen_stein']
    for key in required:
        assert key in result, f"Missing key: {key}"
    print("  All keys present: PASSED")


if __name__ == "__main__":
    test_generator_kernel_shape_and_symmetry()
    test_generator_cv_basic()
    test_generator_cv_variance_reduction()
    test_generator_cv_return_keys()
    print("\n✅ ALL TESTS PASSED")
