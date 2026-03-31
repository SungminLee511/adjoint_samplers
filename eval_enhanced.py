"""
eval_enhanced.py

Load a trained ASBS checkpoint and run all enhancements.
Usage:
    python eval_enhanced.py experiment=dw4_asbs checkpoint=checkpoints/checkpoint_latest.pt
"""

import hydra
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.utils import train_utils

# Our enhancements
from enhancements.enhanced_evaluator import evaluate_enhanced
from enhancements.antithetic import sdeint_antithetic, antithetic_estimate
from enhancements.stein_kernel import compute_ksd
from enhancements.generator_stein import generator_stein_cv_estimate


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Setup (same as train.py) ---
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    # Load checkpoint
    checkpoint_path = Path(cfg.get('checkpoint', 'checkpoints/checkpoint_latest.pt'))
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    controller.load_state_dict(checkpoint["controller"])
    print(f"Loaded checkpoint from {checkpoint_path}")

    # --- Load reference samples for ground truth ---
    evaluator = hydra.utils.instantiate(cfg.evaluator, energy=energy)
    if hasattr(evaluator, 'ref_samples'):
        ref_samples = evaluator.ref_samples.to(device)
        ref_energies = energy.eval(ref_samples)
    else:
        ref_energies = None

    # --- Generate samples ---
    N = cfg.get('num_eval_samples', 2000)
    B = cfg.get('eval_batch_size', 2000)
    timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)

    print(f"Generating {N} samples...")
    x1_list = []
    n_gen = 0
    while n_gen < N:
        b = min(B, N - n_gen)
        x0 = source.sample([b]).to(device)
        _, x1 = sdeint(sde, x0, timesteps, only_boundary=True)
        x1_list.append(x1)
        n_gen += b
    samples = torch.cat(x1_list, dim=0)
    print(f"Generated {samples.shape[0]} samples, shape {samples.shape}")

    # --- Run enhanced evaluation ---
    print("\n=== Enhanced Evaluation ===")
    results = evaluate_enhanced(
        samples, energy,
        ref_energies=ref_energies,
        mh_steps=10,
        stein_reg_lambda=1e-4,
        max_stein_samples=min(N, 2000),
    )

    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.6f}")

    # --- Antithetic sampling ---
    print("\n=== Antithetic Sampling ===")
    x0 = source.sample([min(N, 1000)]).to(device)
    _, x1_orig, x1_anti = sdeint_antithetic(sde, x0, timesteps, only_boundary=True)
    E_orig = energy.eval(x1_orig)
    E_anti = energy.eval(x1_anti)
    anti_results = antithetic_estimate(E_orig, E_anti)
    for k, v in sorted(anti_results.items()):
        print(f"  {k}: {v:.6f}")

    # --- Generator Stein CV ---
    print("\n=== Generator Stein CV ===")
    N_gen = min(N, 1000)
    idx = torch.randperm(N)[:N_gen]
    gen_results = generator_stein_cv_estimate(
        samples[idx], sde, energy.eval(samples[idx]),
        reg_lambda=1e-4,
    )
    for k, v in sorted(gen_results.items()):
        print(f"  {k}: {v:.6f}")

    # --- Summary ---
    print("\n=== SUMMARY ===")
    if ref_energies is not None:
        gt = ref_energies.mean().item()
        print(f"  Ground truth mean energy: {gt:.6f}")
        print(f"  Naive estimate:           {results['mean_energy_naive']:.6f}  (error: {results.get('error_naive', 'N/A')})")
        print(f"  Stein CV estimate:        {results.get('stein_cv_energy_estimate', 'N/A')}  (error: {results.get('error_stein', 'N/A')})")
        print(f"  MCMC corrected:           {results['mean_energy_mcmc']:.6f}  (error: {results.get('error_mcmc', 'N/A')})")
        print(f"  MCMC + Stein CV:          {results.get('hybrid_energy_estimate', 'N/A')}  (error: {results.get('error_hybrid', 'N/A')})")
        print(f"  Antithetic:               {anti_results['estimate']:.6f}")
        print(f"  Generator Stein CV:       {gen_results['estimate']:.6f}")
    print(f"\n  KSD^2:                    {results['ksd_squared']:.6f}")
    print(f"  MH acceptance rate:       {results['mh_acceptance_rate']:.4f}")
    print(f"  Antithetic correlation:   {anti_results['correlation']:.4f}")


if __name__ == "__main__":
    main()
