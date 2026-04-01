"""
enhancements/enhanced_evaluator.py

Enhanced evaluator that wraps vanilla ASBS evaluation with all enhancements.
Runs KSD diagnostic, Stein CV, MCMC correction, and the hybrid pipeline
on a set of terminal samples.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from adjoint_samplers.energies.base_energy import BaseEnergy

from enhancements.stein_kernel import compute_ksd, median_bandwidth
from enhancements.stein_cv import multi_function_stein_cv
from enhancements.mcmc_correction import mh_correct
from enhancements.neural_stein_cv import NeuralSteinCV, train_neural_stein_cv


def evaluate_enhanced(
    samples: torch.Tensor,
    energy: "BaseEnergy",
    ref_energies: Optional[torch.Tensor] = None,
    mh_steps: int = 10,
    stein_reg_lambda: float = 1e-4,
    max_stein_samples: int = 2000,
) -> dict:
    """Run all enhancements on terminal ASBS samples.

    Pipeline:
        1. Compute energies and scores
        2. KSD diagnostic
        3. Vanilla (naive) estimates
        4. Stein CV estimates
        5. MCMC correction
        6. MCMC + Stein CV (hybrid)
        7. Ground truth comparison (if ref_energies provided)

    Args:
        samples: (N, D) terminal samples from ASBS
        energy: energy function with eval(), score() methods
        ref_energies: (M,) reference energy values for ground truth comparison
        mh_steps: number of MH correction steps
        stein_reg_lambda: regularization for Stein CV
        max_stein_samples: max samples for Stein kernel (O(N^2) memory)

    Returns:
        dict with all metrics
    """
    N, D = samples.shape
    device = samples.device
    results = {}

    # --- 1. Compute energies and scores ---
    with torch.no_grad():
        energies = energy.eval(samples)        # (N,)
    scores = energy.score(samples)             # (N, D) = -grad_E

    # --- 2. KSD diagnostic ---
    # Subsample if N is too large for O(N^2)
    N_ksd = min(N, max_stein_samples)
    idx = torch.randperm(N, device=device)[:N_ksd]
    ksd_sq = compute_ksd(samples[idx], scores[idx])
    results['ksd_squared'] = ksd_sq.item()

    # --- 3. Vanilla estimates ---
    results['mean_energy_naive'] = energies.mean().item()
    results['std_energy_naive'] = energies.std().item()
    results['var_energy_estimator_naive'] = (energies.var() / N).item()

    # --- 4. Stein CV estimates ---
    N_stein = min(N, max_stein_samples)
    idx_stein = torch.randperm(N, device=device)[:N_stein]
    f_dict = {
        'energy': energies[idx_stein],
    }
    stein_results = multi_function_stein_cv(
        samples[idx_stein],
        scores[idx_stein],
        f_dict,
        reg_lambda=stein_reg_lambda,
    )
    for fname, fres in stein_results.items():
        results[f'stein_cv_{fname}_estimate'] = fres['estimate']
        results[f'stein_cv_{fname}_var_naive'] = fres['variance_naive']
        results[f'stein_cv_{fname}_var_stein'] = fres['variance_stein']
        if fres['variance_naive'] > 0:
            results[f'stein_cv_{fname}_var_reduction'] = (
                fres['variance_stein'] / fres['variance_naive']
            )

    # --- 5. MCMC correction ---
    mh_result = mh_correct(samples, energy, n_steps=mh_steps)
    corrected = mh_result['corrected_samples']
    results['mh_acceptance_rate'] = mh_result['acceptance_rate']

    corrected_energies = energy.eval(corrected)
    results['mean_energy_mcmc'] = corrected_energies.mean().item()
    results['var_energy_estimator_mcmc'] = (corrected_energies.var() / N).item()

    # --- 6. MCMC + Stein CV (the full pipeline) ---
    corrected_scores = energy.score(corrected)
    N_hybrid = min(N, max_stein_samples)
    idx_hybrid = torch.randperm(N, device=device)[:N_hybrid]
    hybrid_f_dict = {
        'energy': corrected_energies[idx_hybrid],
    }
    hybrid_results = multi_function_stein_cv(
        corrected[idx_hybrid],
        corrected_scores[idx_hybrid],
        hybrid_f_dict,
        reg_lambda=stein_reg_lambda,
    )
    for fname, fres in hybrid_results.items():
        results[f'hybrid_{fname}_estimate'] = fres['estimate']
        results[f'hybrid_{fname}_var_stein'] = fres['variance_stein']
        if fres['variance_naive'] > 0:
            results[f'hybrid_{fname}_var_reduction'] = (
                fres['variance_stein'] / fres['variance_naive']
            )

    # --- 7. Neural Stein CV ---
    neural_epochs = 500 if D <= 40 else 1000
    hutch = 0 if D <= 20 else 1  # exact div for low-d, Hutchinson for high-d
    g_model = NeuralSteinCV(
        dim=D, hidden_dim=min(256, max(64, D * 2)), n_layers=3,
    ).to(device)
    neural_result = train_neural_stein_cv(
        g_model,
        samples,
        energy,
        f_func=lambda x: energy.eval(x),
        n_epochs=neural_epochs,
        batch_size=min(256, N),
        lr=1e-3,
        hutchinson_samples=hutch,
        verbose=False,
    )
    results['neural_cv_estimate'] = neural_result['estimate']
    results['neural_cv_var'] = neural_result['variance_neural']
    results['neural_cv_var_reduction'] = neural_result['variance_reduction']
    results['neural_cv_final_loss'] = neural_result['losses'][-1]

    # --- 8. Ground truth comparison ---
    if ref_energies is not None:
        gt_mean = ref_energies.mean().item()
        results['gt_mean_energy'] = gt_mean
        results['error_naive'] = abs(results['mean_energy_naive'] - gt_mean)
        results['error_stein'] = abs(stein_results['energy']['estimate'] - gt_mean)
        results['error_mcmc'] = abs(results['mean_energy_mcmc'] - gt_mean)
        results['error_hybrid'] = abs(hybrid_results['energy']['estimate'] - gt_mean)
        results['error_neural_cv'] = abs(neural_result['estimate'] - gt_mean)

    return results
