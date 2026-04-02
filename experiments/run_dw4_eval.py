#!/usr/bin/env python
"""
DW4 Evaluation — Phases 2-5 combined.

Runs all 9 SML methods with practical hyperparameters for DW4 (8D).
Produces results.json + all plots.

Usage:
    /root/miniconda3/envs/Sampling_env/bin/python -u experiments/run_dw4_eval.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from pathlib import Path

from adjoint_samplers.energies.double_well_energy import DoubleWellEnergy
from adjoint_samplers.components.sde import GraphVESDE, ControlledSDE, sdeint
from adjoint_samplers.components.model import EGNN_dynamics
from adjoint_samplers.utils.dist_utils import CenteredParticlesHarmonic
from adjoint_samplers.utils.graph_utils import remove_mean

from enhancements.evaluation import EvalConfig, full_evaluation, single_run_evaluation, save_results
from enhancements.visualization import generate_all_plots, plot_mcmc_ablation

import torch.nn as nn

CKPT = "/home/RESEARCH/adjoint_samplers/results/local/2026.03.31/152919/checkpoints/checkpoint_latest.pt"
OUT_DIR = "/home/RESEARCH/adjoint_samplers/eval_results"
EXP_NAME = "dw4_asbs"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    n_particles, spatial_dim = 4, 2
    dim = n_particles * spatial_dim

    # Build energy
    energy = DoubleWellEnergy(dim=dim, n_particles=n_particles, device=device)
    source = CenteredParticlesHarmonic(
        n_particles=n_particles, spatial_dim=spatial_dim,
        scale=2.0, device=device,
    )
    ref_sde = GraphVESDE(
        n_particles=n_particles, spatial_dim=spatial_dim,
        sigma_max=1.0, sigma_min=0.001,
    ).to(device)
    controller = EGNN_dynamics(
        n_particles=n_particles, spatial_dim=spatial_dim,
        hidden_nf=128, n_layers=5,
        act_fn=nn.SiLU(), recurrent=True, tanh=True,
        attention=True, condition_time=True, agg="sum",
    ).to(device)

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt["controller"])
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint (epoch {epoch})")

    sde = ControlledSDE(ref_sde, controller).to(device)

    timesteps = torch.linspace(0, 1, 200, device=device)

    # Ground truth
    root = Path(__file__).parent.parent
    ref = np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)
    ref_t = remove_mean(torch.tensor(ref, dtype=torch.float32), n_particles, spatial_dim).to(device)
    gt_mean_energy = energy.eval(ref_t).mean().item()
    print(f"Ground truth mean energy: {gt_mean_energy:.6f}")

    # Eval config — practical sizes for DW4
    eval_config = EvalConfig(
        n_seeds=10,
        sample_sizes=[100, 500, 1000, 2000],
        max_stein_samples=2000,
        eval_batch_size=2000,
        # Neural CV (MLP) — moderate
        neural_cv_epochs=500,
        neural_cv_hidden_dim=128,
        neural_cv_n_layers=3,
        neural_cv_batch_size=512,
        neural_cv_lr=1e-3,
        # EGNN CV — moderate
        egnn_cv_epochs=500,
        egnn_cv_hidden_nf=64,
        egnn_cv_n_layers=4,
        egnn_cv_batch_size=512,
        egnn_cv_lr=1e-3,
        # RBF
        rbf_n_centers=200,
        rbf_reg_lambda=1e-6,
        # Particle structure
        n_particles=n_particles,
        spatial_dim=spatial_dim,
    )

    # Main evaluation
    print("\n========== MAIN EVALUATION ==========")
    results = full_evaluation(
        sde, source, energy, timesteps, device,
        gt_mean_energy=gt_mean_energy,
        config=eval_config,
    )

    out_path = f'{OUT_DIR}/{EXP_NAME}'
    save_results(results, f'{out_path}/results.json')

    # MCMC ablation
    print("\n========== MCMC ABLATION ==========")
    mcmc_results = {}
    for K in [0, 5, 10, 20, 50]:
        torch.manual_seed(0)
        run = single_run_evaluation(
            sde, source, energy, timesteps,
            n_samples=2000, mh_steps=K,
            stein_reg_lambda=1e-4,
            device=device,
            gt_mean_energy=gt_mean_energy,
            config=eval_config,
        )
        mcmc_results[K] = run
        err_mcmc = run.get('error_mcmc', float('nan'))
        acc = run.get('mcmc_acceptance', 0)
        print(f"  K={K}: error_mcmc={err_mcmc:.6f}, acceptance={acc:.4f}")

    # Generate plots
    print("\n========== GENERATING PLOTS ==========")
    generate_all_plots(
        results, OUT_DIR, EXP_NAME,
        gt_mean_energy=gt_mean_energy,
    )

    if gt_mean_energy is not None:
        plot_mcmc_ablation(
            mcmc_results,
            f'{out_path}/mcmc_ablation.png',
            gt_mean_energy=gt_mean_energy,
            title=f'{EXP_NAME}: MCMC Ablation',
        )

    # Final summary
    N_max = max(results.keys())
    r = results[N_max]
    print(f"\n========== FINAL SUMMARY (DW4, N={N_max}) ==========")
    print(f"  Ground truth:       {gt_mean_energy:.6f}")
    print(f"  KSD^2:              {r['ksd_squared']['mean']:.6f} +/- {r['ksd_squared']['std']:.6f}")
    print(f"  MH acceptance:      {r['mcmc_acceptance']['mean']:.4f}")
    print(f"  Anti correlation:   {r['anti_correlation']['mean']:.4f}")
    print()
    print(f"  {'Method':<22} {'Estimate':>12} {'|Error|':>10} {'Var':>12} {'VarRatio':>10}")
    print(f"  {'-'*66}")

    rows = [
        ('Vanilla',       'naive_mean_energy', 'error_naive', 'naive_var',     None),
        ('Stein CV',      'stein_cv_estimate', 'error_stein', 'stein_cv_var',  'stein_var_reduction'),
        ('Antithetic',    'anti_estimate',     'error_anti',  'anti_var',      'anti_var_reduction'),
        ('MCMC',          'mcmc_mean_energy',  'error_mcmc',  'mcmc_var',      None),
        ('MCMC+Stein',    'hybrid_estimate',   'error_hybrid','hybrid_var',    None),
        ('Gen Stein CV',  'gen_stein_estimate','error_gen_stein','gen_stein_var', None),
        ('Neural CV',     'neural_cv_estimate','error_neural_cv','neural_cv_var','neural_cv_var_reduction'),
        ('EGNN CV',       'egnn_cv_estimate',  'error_egnn_cv','egnn_cv_var',  'egnn_cv_var_reduction'),
        ('RBF Colloc CV', 'rbf_cv_estimate',   'error_rbf_cv', 'rbf_cv_var',  'rbf_cv_var_reduction'),
    ]

    for label, est_key, err_key, var_key, vr_key in rows:
        est = r[est_key]['mean']
        err = r.get(err_key, {}).get('mean', float('nan'))
        var = r[var_key]['mean']
        vr = r.get(vr_key, {}).get('mean', float('nan')) if vr_key else float('nan')
        vr_str = f"{vr:.3f}" if not np.isnan(vr) else "---"
        print(f"  {label:<22} {est:>12.6f} {err:>10.6f} {var:>12.2e} {vr_str:>10}")

    print("\nDone.")


if __name__ == "__main__":
    main()
