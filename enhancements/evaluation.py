"""
enhancements/evaluation.py

Systematic evaluation: multiple seeds, multiple sample sizes,
all enhancements compared against baseline, with statistics.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List

import torch
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
import json

if TYPE_CHECKING:
    from adjoint_samplers.components.sde import ControlledSDE
    from adjoint_samplers.energies.base_energy import BaseEnergy

from adjoint_samplers.components.sde import sdeint

from enhancements.stein_kernel import compute_ksd, median_bandwidth
from enhancements.stein_cv import stein_cv_estimate, multi_function_stein_cv
from enhancements.antithetic import sdeint_antithetic, antithetic_estimate
from enhancements.mcmc_correction import mh_correct
from enhancements.generator_stein import generator_stein_cv_estimate


@dataclass
class EvalConfig:
    """Configuration for the systematic evaluation."""
    n_seeds: int = 10                          # Number of independent runs
    sample_sizes: List[int] = field(           # Sample sizes to test scaling
        default_factory=lambda: [100, 500, 1000, 2000]
    )
    mh_steps_list: List[int] = field(          # MH steps to ablate
        default_factory=lambda: [0, 5, 10, 20]
    )
    stein_reg_lambdas: List[float] = field(    # Stein regularization to ablate
        default_factory=lambda: [1e-6, 1e-4, 1e-2]
    )
    max_stein_samples: int = 2000              # Memory cap for Stein kernel
    eval_batch_size: int = 2000                # Batch size for generation


def generate_samples(
    sde: "ControlledSDE",
    source,
    timesteps: torch.Tensor,
    n_samples: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Generate n_samples terminal samples from the SDE."""
    x1_list = []
    n_gen = 0
    while n_gen < n_samples:
        b = min(batch_size, n_samples - n_gen)
        x0 = source.sample([b]).to(device)
        _, x1 = sdeint(sde, x0, timesteps, only_boundary=True)
        x1_list.append(x1)
        n_gen += b
    return torch.cat(x1_list, dim=0)[:n_samples]


def single_run_evaluation(
    sde: "ControlledSDE",
    source,
    energy: "BaseEnergy",
    timesteps: torch.Tensor,
    n_samples: int,
    mh_steps: int,
    stein_reg_lambda: float,
    device: str,
    gt_mean_energy: Optional[float] = None,
) -> dict:
    """Run one complete evaluation: generate samples, apply all enhancements.

    Returns a dict of scalar metrics.
    """
    results = {}

    # --- Generate samples ---
    samples = generate_samples(
        sde, source, timesteps, n_samples,
        batch_size=min(n_samples, 2000), device=device,
    )
    N, D = samples.shape

    # --- Energies and scores ---
    with torch.no_grad():
        energies = energy.eval(samples)
    scores = energy.score(samples)

    # --- 1. Naive estimate ---
    results['naive_mean_energy'] = energies.mean().item()
    results['naive_var'] = (energies.var() / N).item()

    # --- 2. KSD ---
    N_ksd = min(N, 2000)
    idx = torch.randperm(N, device=device)[:N_ksd]
    ksd_sq = compute_ksd(samples[idx], scores[idx])
    results['ksd_squared'] = ksd_sq.item()

    # --- 3. Stein CV ---
    N_s = min(N, 2000)
    idx_s = torch.randperm(N, device=device)[:N_s]
    scv = stein_cv_estimate(
        samples[idx_s], scores[idx_s], energies[idx_s],
        reg_lambda=stein_reg_lambda,
    )
    results['stein_cv_estimate'] = scv['estimate']
    results['stein_cv_var'] = scv['variance_stein']
    results['stein_var_reduction'] = (
        scv['variance_stein'] / (scv['variance_naive'] + 1e-20)
    )

    # --- 4. Antithetic ---
    N_anti = min(N, 1000)
    x0_anti = source.sample([N_anti]).to(device)
    _, x1_orig, x1_anti = sdeint_antithetic(
        sde, x0_anti, timesteps, only_boundary=True
    )
    E_orig = energy.eval(x1_orig)
    E_anti = energy.eval(x1_anti)
    anti = antithetic_estimate(E_orig, E_anti)
    results['anti_estimate'] = anti['estimate']
    results['anti_var'] = anti['variance_anti']
    results['anti_correlation'] = anti['correlation']
    results['anti_var_reduction'] = anti['variance_reduction_factor']

    # --- 5. MCMC correction ---
    if mh_steps > 0:
        mh = mh_correct(samples, energy, n_steps=mh_steps)
        corrected = mh['corrected_samples']
        corrected_energies = energy.eval(corrected)
        results['mcmc_mean_energy'] = corrected_energies.mean().item()
        results['mcmc_var'] = (corrected_energies.var() / N).item()
        results['mcmc_acceptance'] = mh['acceptance_rate']

        # --- 6. MCMC + Stein CV (hybrid) ---
        corrected_scores = energy.score(corrected)
        N_h = min(N, 2000)
        idx_h = torch.randperm(N, device=device)[:N_h]
        hybrid = stein_cv_estimate(
            corrected[idx_h], corrected_scores[idx_h],
            corrected_energies[idx_h],
            reg_lambda=stein_reg_lambda,
        )
        results['hybrid_estimate'] = hybrid['estimate']
        results['hybrid_var'] = hybrid['variance_stein']
    else:
        results['mcmc_mean_energy'] = results['naive_mean_energy']
        results['mcmc_var'] = results['naive_var']
        results['mcmc_acceptance'] = 0.0
        results['hybrid_estimate'] = results['stein_cv_estimate']
        results['hybrid_var'] = results['stein_cv_var']

    # --- 7. Generator Stein CV ---
    N_g = min(N, 1000)
    idx_g = torch.randperm(N, device=device)[:N_g]
    gen = generator_stein_cv_estimate(
        samples[idx_g], sde, energies[idx_g],
        reg_lambda=stein_reg_lambda,
    )
    results['gen_stein_estimate'] = gen['estimate']
    results['gen_stein_var'] = gen['variance_gen_stein']

    # --- 8. Errors vs ground truth ---
    if gt_mean_energy is not None:
        gt = gt_mean_energy
        results['error_naive'] = abs(results['naive_mean_energy'] - gt)
        results['error_stein'] = abs(results['stein_cv_estimate'] - gt)
        results['error_anti'] = abs(results['anti_estimate'] - gt)
        results['error_mcmc'] = abs(results['mcmc_mean_energy'] - gt)
        results['error_hybrid'] = abs(results['hybrid_estimate'] - gt)
        results['error_gen_stein'] = abs(results['gen_stein_estimate'] - gt)

    return results


def full_evaluation(
    sde: "ControlledSDE",
    source,
    energy: "BaseEnergy",
    timesteps: torch.Tensor,
    device: str,
    gt_mean_energy: Optional[float] = None,
    config: Optional[EvalConfig] = None,
) -> dict:
    """Run the full systematic evaluation across seeds and sample sizes.

    Returns a nested dict:
        results[n_samples][metric_name] = {
            'mean': float, 'std': float, 'values': list[float]
        }
    """
    if config is None:
        config = EvalConfig()

    all_results = {}

    for n_samples in config.sample_sizes:
        print(f"\n--- N = {n_samples} ---")
        seed_results = []

        for seed in range(config.n_seeds):
            torch.manual_seed(seed * 12345 + n_samples)
            np.random.seed(seed * 12345 + n_samples)

            run = single_run_evaluation(
                sde=sde,
                source=source,
                energy=energy,
                timesteps=timesteps,
                n_samples=n_samples,
                mh_steps=10,
                stein_reg_lambda=1e-4,
                device=device,
                gt_mean_energy=gt_mean_energy,
            )
            seed_results.append(run)
            print(f"  seed {seed}: naive={run['naive_mean_energy']:.4f}, "
                  f"stein={run['stein_cv_estimate']:.4f}, "
                  f"hybrid={run['hybrid_estimate']:.4f}")

        # Aggregate across seeds
        all_keys = seed_results[0].keys()
        aggregated = {}
        for key in all_keys:
            values = [r[key] for r in seed_results]
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values,
            }

        all_results[n_samples] = aggregated

    return all_results


def save_results(results: dict, path: str):
    """Save results to JSON (convert numpy/torch types)."""
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        return obj

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"Results saved to {path}")
