"""
Test enhanced_evaluator.py with a synthetic Gaussian energy.

E(x) = 0.5 * ||x||^2 => p(x) = N(0, I)
Ground truth E[E(X)] = D/2
"""

import sys
sys.path.insert(0, '/home/RESEARCH/adjoint_samplers')

import torch
from enhancements.enhanced_evaluator import evaluate_enhanced


class GaussianEnergy:
    """E(x) = 0.5 * ||x||^2, so p(x) = N(0, I)."""
    def eval(self, x):
        return 0.5 * (x**2).sum(dim=-1)

    def score(self, x):
        """s_p(x) = -grad_E(x) = -x."""
        return -x


def test_enhanced_evaluator_full_pipeline():
    print("=== Test: evaluate_enhanced full pipeline ===")
    torch.manual_seed(42)
    D = 5
    N = 500

    energy = GaussianEnergy()
    gt_mean_energy = D / 2.0  # = 2.5

    # True samples from N(0, I)
    samples = torch.randn(N, D)
    ref_energies = energy.eval(torch.randn(10000, D))

    results = evaluate_enhanced(
        samples, energy,
        ref_energies=ref_energies,
        mh_steps=10,
        stein_reg_lambda=1e-4,
        max_stein_samples=500,
    )

    print(f"  Ground truth E[E]: {gt_mean_energy}")
    print(f"  KSD^2:                    {results['ksd_squared']:.6f}")
    print(f"  Naive mean energy:        {results['mean_energy_naive']:.4f}")
    print(f"  Stein CV estimate:        {results['stein_cv_energy_estimate']:.4f}")
    print(f"  MCMC mean energy:         {results['mean_energy_mcmc']:.4f}")
    print(f"  Hybrid estimate:          {results['hybrid_energy_estimate']:.4f}")
    print(f"  MH acceptance rate:       {results['mh_acceptance_rate']:.4f}")

    if 'stein_cv_energy_var_reduction' in results:
        print(f"  Stein var reduction:      {results['stein_cv_energy_var_reduction']:.4f}")

    # All required keys present
    required_keys = [
        'ksd_squared', 'mean_energy_naive', 'std_energy_naive',
        'var_energy_estimator_naive', 'stein_cv_energy_estimate',
        'mh_acceptance_rate', 'mean_energy_mcmc',
        'hybrid_energy_estimate', 'gt_mean_energy',
        'error_naive', 'error_stein', 'error_mcmc', 'error_hybrid',
    ]
    for key in required_keys:
        assert key in results, f"Missing key: {key}"

    # Sanity checks
    assert results['ksd_squared'] >= -0.5, \
        f"KSD^2 too negative: {results['ksd_squared']}"
    assert results['mh_acceptance_rate'] > 0, "No MH accepts"
    assert results['error_naive'] < 2.0, "Naive error too large"

    print("  All keys present and sane: PASSED")


def test_enhanced_evaluator_biased():
    print("\n=== Test: evaluate_enhanced with biased samples ===")
    torch.manual_seed(42)
    D = 5
    N = 400

    energy = GaussianEnergy()

    # Biased samples from N(1, I) — NOT the target
    samples = torch.randn(N, D) + 1.0
    ref_energies = energy.eval(torch.randn(5000, D))

    results = evaluate_enhanced(
        samples, energy,
        ref_energies=ref_energies,
        mh_steps=20,
        stein_reg_lambda=1e-4,
        max_stein_samples=400,
    )

    print(f"  Ground truth:    {results['gt_mean_energy']:.4f}")
    print(f"  Error naive:     {results['error_naive']:.4f}")
    print(f"  Error Stein:     {results['error_stein']:.4f}")
    print(f"  Error MCMC:      {results['error_mcmc']:.4f}")
    print(f"  Error Hybrid:    {results['error_hybrid']:.4f}")

    # MCMC should improve on naive for biased samples
    assert results['error_mcmc'] < results['error_naive'], \
        "MCMC should reduce error for biased samples"
    print("  PASSED: MCMC improves on naive for biased samples")


def test_enhanced_evaluator_no_ref():
    print("\n=== Test: evaluate_enhanced without ref_energies ===")
    torch.manual_seed(42)
    D = 3
    N = 200

    energy = GaussianEnergy()
    samples = torch.randn(N, D)

    results = evaluate_enhanced(
        samples, energy,
        ref_energies=None,
        mh_steps=5,
        max_stein_samples=200,
    )

    # Should run without error, no ground truth keys
    assert 'gt_mean_energy' not in results
    assert 'error_naive' not in results
    assert 'ksd_squared' in results
    assert 'mean_energy_naive' in results
    print(f"  Naive: {results['mean_energy_naive']:.4f}, KSD: {results['ksd_squared']:.6f}")
    print("  PASSED: runs without ref_energies")


if __name__ == "__main__":
    test_enhanced_evaluator_full_pipeline()
    test_enhanced_evaluator_biased()
    test_enhanced_evaluator_no_ref()
    print("\n✅ ALL TESTS PASSED")
