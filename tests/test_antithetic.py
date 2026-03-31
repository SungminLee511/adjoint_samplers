"""
Test antithetic.py using BrownianMotionSDE (zero drift).

For pure Brownian motion: x1 = x0 + σW, x1_anti = x0 - σW.
- f(x) = x_0 (first coordinate): antithetic pairs are negatively correlated
- f(x) = ||x||^2: pairs may be positively correlated (even function)
"""

import sys
sys.path.insert(0, '/home/RESEARCH/adjoint_samplers')

import torch
from adjoint_samplers.components.sde import BrownianMotionSDE
from enhancements.antithetic import sdeint_antithetic, antithetic_estimate


def test_antithetic_basic():
    print("=== Test: sdeint_antithetic basic shapes ===")
    torch.manual_seed(42)

    sde = BrownianMotionSDE(sigma=1.0)
    B, D = 100, 5
    x0 = torch.randn(B, D)
    timesteps = torch.linspace(0, 1, 51)  # 50 steps

    x0_out, x1, x1_anti = sdeint_antithetic(sde, x0, timesteps, only_boundary=True)

    assert x0_out.shape == (B, D), f"x0 shape: {x0_out.shape}"
    assert x1.shape == (B, D), f"x1 shape: {x1.shape}"
    assert x1_anti.shape == (B, D), f"x1_anti shape: {x1_anti.shape}"
    assert torch.allclose(x0_out, x0), "x0 should be unchanged"

    # x1 and x1_anti should be different
    assert not torch.allclose(x1, x1_anti), "x1 and x1_anti should differ"
    print("  Shape and basic checks: PASSED")


def test_antithetic_full_trajectory():
    print("\n=== Test: sdeint_antithetic full trajectory ===")
    torch.manual_seed(42)

    sde = BrownianMotionSDE(sigma=1.0)
    B, D = 50, 3
    x0 = torch.zeros(B, D)
    timesteps = torch.linspace(0, 1, 11)  # 10 steps

    states, states_anti = sdeint_antithetic(sde, x0, timesteps, only_boundary=False)

    assert len(states) == 11, f"Expected 11 states, got {len(states)}"
    assert len(states_anti) == 11, f"Expected 11 anti states, got {len(states_anti)}"

    # Both should start from the same x0
    assert torch.allclose(states[0], states_anti[0]), "Should start from same x0"
    print("  Full trajectory: PASSED")


def test_antithetic_brownian_mirror():
    print("\n=== Test: Brownian motion mirror property ===")
    torch.manual_seed(42)

    sde = BrownianMotionSDE(sigma=1.0)
    B, D = 500, 10
    x0 = torch.zeros(B, D)  # Start from origin for clarity
    # Use a single step so the antithetic pair is exact mirror
    timesteps = torch.tensor([0.0, 1.0])

    _, x1, x1_anti = sdeint_antithetic(sde, x0, timesteps, only_boundary=True)

    # For single-step BM with x0=0: x1 = σ*√dt*noise, x1_anti = -σ*√dt*noise
    # So x1 + x1_anti should be ~0 (exactly 0 for single step from origin)
    residual = (x1 + x1_anti).abs().max().item()
    print(f"  Max |x1 + x1_anti|: {residual:.2e}")
    assert residual < 1e-5, f"Not a perfect mirror: {residual}"
    print("  PASSED: perfect antithetic mirror for single-step BM")


def test_antithetic_negative_correlation():
    print("\n=== Test: negative correlation for linear function ===")
    torch.manual_seed(42)

    sde = BrownianMotionSDE(sigma=1.0)
    B, D = 1000, 5
    x0 = torch.zeros(B, D)
    timesteps = torch.linspace(0, 1, 21)  # 20 steps

    _, x1, x1_anti = sdeint_antithetic(sde, x0, timesteps, only_boundary=True)

    # f(x) = x_0 (first coordinate) — odd function of displacement
    f_orig = x1[:, 0]
    f_anti = x1_anti[:, 0]

    result = antithetic_estimate(f_orig, f_anti)

    print(f"  Naive estimate:  {result['naive_estimate']:.4f} (should be ~0)")
    print(f"  Anti estimate:   {result['estimate']:.4f} (should be ~0)")
    print(f"  Correlation:     {result['correlation']:.4f} (should be negative)")
    print(f"  Var naive:       {result['variance_naive']:.6f}")
    print(f"  Var anti:        {result['variance_anti']:.6f}")
    print(f"  Var reduction:   {result['variance_reduction_factor']:.4f}")

    # For BM with zero drift, antithetic correlation should be strongly negative
    assert result['correlation'] < 0, \
        f"Expected negative correlation, got {result['correlation']}"
    assert result['variance_reduction_factor'] < 1.0, \
        f"Expected variance reduction, got {result['variance_reduction_factor']}"
    print("  PASSED")


def test_antithetic_estimate_dict():
    print("\n=== Test: antithetic_estimate return values ===")
    torch.manual_seed(42)
    N = 200

    f = torch.randn(N)
    f_anti = -f + 0.1 * torch.randn(N)  # Nearly perfect antithetic

    result = antithetic_estimate(f, f_anti)

    assert 'estimate' in result
    assert 'naive_estimate' in result
    assert 'variance_naive' in result
    assert 'variance_anti' in result
    assert 'correlation' in result
    assert 'variance_reduction_factor' in result

    # With nearly perfect antithetic, variance should be very small
    print(f"  Var reduction factor: {result['variance_reduction_factor']:.4f}")
    assert result['variance_reduction_factor'] < 0.1, \
        "Nearly perfect antithetic should give >90% variance reduction"
    print("  PASSED")


if __name__ == "__main__":
    test_antithetic_basic()
    test_antithetic_full_trajectory()
    test_antithetic_brownian_mirror()
    test_antithetic_negative_correlation()
    test_antithetic_estimate_dict()
    print("\n✅ ALL TESTS PASSED")
