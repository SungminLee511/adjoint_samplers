"""
Test mcmc_correction.py with a Gaussian energy.

E(x) = 0.5 * ||x||^2 => p(x) = N(0, I)
- Start from biased samples (shifted mean)
- MCMC should push mean toward 0
- Acceptance rate should be in reasonable range
"""

import sys
sys.path.insert(0, '/home/RESEARCH/adjoint_samplers')

import torch
from enhancements.mcmc_correction import mh_correct


class GaussianEnergy:
    """Simple Gaussian energy: E(x) = 0.5 * ||x||^2, so p(x) = N(0, I)."""
    def eval(self, x):
        return 0.5 * (x**2).sum(dim=-1)


def test_mcmc_shapes():
    print("=== Test: mh_correct output shapes ===")
    torch.manual_seed(42)
    energy = GaussianEnergy()
    N, D = 200, 5
    samples = torch.randn(N, D) + 1.0  # biased

    result = mh_correct(samples, energy, n_steps=10)

    assert result['corrected_samples'].shape == (N, D)
    assert result['energies_before'].shape == (N,)
    assert result['energies_after'].shape == (N,)
    assert 0 <= result['acceptance_rate'] <= 1
    print(f"  Acceptance rate: {result['acceptance_rate']:.4f}")
    print("  PASSED")


def test_mcmc_bias_correction():
    print("\n=== Test: MCMC corrects bias ===")
    torch.manual_seed(42)
    energy = GaussianEnergy()
    N, D = 1000, 5

    # Biased samples: mean shifted to 2.0
    samples = torch.randn(N, D) + 2.0
    mean_before = samples.mean(dim=0).norm().item()

    result = mh_correct(samples, energy, n_steps=50)
    corrected = result['corrected_samples']
    mean_after = corrected.mean(dim=0).norm().item()

    print(f"  ||mean|| before: {mean_before:.4f}")
    print(f"  ||mean|| after:  {mean_after:.4f}")
    print(f"  Acceptance rate: {result['acceptance_rate']:.4f}")

    assert mean_after < mean_before, \
        f"MCMC should reduce bias: {mean_after} >= {mean_before}"
    print("  PASSED: mean moved toward origin")


def test_mcmc_energy_decreases():
    print("\n=== Test: MCMC lowers mean energy for biased samples ===")
    torch.manual_seed(42)
    energy = GaussianEnergy()
    N, D = 500, 10

    # Start far from equilibrium
    samples = torch.randn(N, D) + 3.0

    result = mh_correct(samples, energy, n_steps=20)

    mean_E_before = result['energies_before'].mean().item()
    mean_E_after = result['energies_after'].mean().item()
    # Ground truth: E[E(X)] = E[0.5*||X||^2] = D/2 = 5.0 for p = N(0,I)

    print(f"  Ground truth E[E]: {D / 2}")
    print(f"  Mean E before: {mean_E_before:.4f}")
    print(f"  Mean E after:  {mean_E_after:.4f}")
    print(f"  Acceptance:    {result['acceptance_rate']:.4f}")

    assert mean_E_after < mean_E_before, \
        f"Energy should decrease: {mean_E_after} >= {mean_E_before}"
    print("  PASSED")


def test_mcmc_acceptance_rate_range():
    print("\n=== Test: acceptance rate in reasonable range ===")
    torch.manual_seed(42)
    energy = GaussianEnergy()
    N, D = 500, 5

    # Well-initialized samples (already from target)
    samples = torch.randn(N, D)

    result = mh_correct(samples, energy, n_steps=20)
    acc = result['acceptance_rate']

    print(f"  Acceptance rate (well-initialized): {acc:.4f}")
    # For well-initialized samples with optimal step size, expect ~0.2-0.5
    assert 0.05 < acc < 0.95, f"Acceptance rate out of range: {acc}"
    print("  PASSED")


def test_mcmc_custom_step_size():
    print("\n=== Test: custom step size ===")
    torch.manual_seed(42)
    energy = GaussianEnergy()
    N, D = 300, 5
    samples = torch.randn(N, D)

    # Very small step → high acceptance, little movement
    result_small = mh_correct(samples, energy, n_steps=10, step_size=0.01)
    # Very large step → low acceptance
    result_large = mh_correct(samples, energy, n_steps=10, step_size=10.0)

    print(f"  Small step (0.01) acceptance: {result_small['acceptance_rate']:.4f}")
    print(f"  Large step (10.0) acceptance: {result_large['acceptance_rate']:.4f}")

    assert result_small['acceptance_rate'] > result_large['acceptance_rate'], \
        "Smaller step should have higher acceptance"
    print("  PASSED")


def test_mcmc_convergence_with_steps():
    print("\n=== Test: more MH steps → better convergence ===")
    torch.manual_seed(42)
    energy = GaussianEnergy()
    N, D = 500, 5
    samples = torch.randn(N, D) + 2.0  # biased

    errors = []
    for K in [0, 5, 20, 50]:
        if K == 0:
            mean_norm = samples.mean(dim=0).norm().item()
        else:
            result = mh_correct(samples, energy, n_steps=K)
            mean_norm = result['corrected_samples'].mean(dim=0).norm().item()
        errors.append(mean_norm)
        print(f"  K={K:3d}: ||mean|| = {mean_norm:.4f}")

    # Error should generally decrease with more steps
    assert errors[-1] < errors[0], \
        f"More steps should reduce bias: K=50 ({errors[-1]}) vs K=0 ({errors[0]})"
    print("  PASSED")


if __name__ == "__main__":
    test_mcmc_shapes()
    test_mcmc_bias_correction()
    test_mcmc_energy_decreases()
    test_mcmc_acceptance_rate_range()
    test_mcmc_custom_step_size()
    test_mcmc_convergence_with_steps()
    print("\n✅ ALL TESTS PASSED")
