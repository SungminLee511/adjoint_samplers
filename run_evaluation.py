"""
run_evaluation.py

Complete evaluation pipeline: load checkpoint, run all enhancements,
generate plots and summary.

Usage:
    python run_evaluation.py experiment=dw4_asbs \
        checkpoint=checkpoints/checkpoint_4999.pt \
        output_dir=eval_results
"""

import hydra
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from adjoint_samplers.components.sde import ControlledSDE
from adjoint_samplers.utils import train_utils

from enhancements.evaluation import (
    EvalConfig, full_evaluation, single_run_evaluation, save_results,
)
from enhancements.visualization import generate_all_plots, plot_mcmc_ablation


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = cfg.get('output_dir', 'eval_results')

    # --- Setup ---
    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    # Load checkpoint
    ckpt_path = Path(cfg.get('checkpoint', 'checkpoints/checkpoint_latest.pt'))
    assert ckpt_path.exists(), f"Not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt["controller"])
    print(f"Loaded: {ckpt_path}")

    timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)

    # --- Ground truth from reference samples ---
    evaluator = hydra.utils.instantiate(cfg.evaluator, energy=energy)
    gt_mean_energy = None
    if hasattr(evaluator, 'ref_samples'):
        ref = evaluator.ref_samples.to(device)
        ref_E = energy.eval(ref)
        gt_mean_energy = ref_E.mean().item()
        print(f"Ground truth mean energy: {gt_mean_energy:.6f}")

    # --- Main evaluation ---
    print("\n========== MAIN EVALUATION ==========")
    eval_config = EvalConfig(
        n_seeds=10,
        sample_sizes=[100, 500, 1000, 2000],
        max_stein_samples=2000,
    )
    results = full_evaluation(
        sde, source, energy, timesteps, device,
        gt_mean_energy=gt_mean_energy,
        config=eval_config,
    )

    # Save raw results
    exp_name = cfg.get('exp_name', 'unknown')
    save_results(results, f'{output_dir}/{exp_name}/results.json')

    # --- MCMC ablation ---
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
        )
        mcmc_results[K] = run
        err_mcmc = run.get('error_mcmc', float('nan'))
        acc = run.get('mcmc_acceptance', 0)
        print(f"  K={K}: error_mcmc={err_mcmc:.6f}, acceptance={acc:.4f}")

    # --- Generate all plots ---
    print("\n========== GENERATING PLOTS ==========")
    generate_all_plots(
        results, output_dir, exp_name,
        gt_mean_energy=gt_mean_energy,
    )

    if gt_mean_energy is not None:
        plot_mcmc_ablation(
            mcmc_results,
            f'{output_dir}/{exp_name}/mcmc_ablation.png',
            gt_mean_energy=gt_mean_energy,
            title=f'{exp_name}: MCMC Ablation',
        )

    # --- Print final summary ---
    N_max = max(results.keys())
    r = results[N_max]
    print(f"\n========== FINAL SUMMARY ({exp_name}, N={N_max}) ==========")
    if gt_mean_energy is not None:
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
    ]

    for label, est_key, err_key, var_key, vr_key in rows:
        est = r[est_key]['mean']
        err = r.get(err_key, {}).get('mean', float('nan'))
        var = r[var_key]['mean']
        vr = r.get(vr_key, {}).get('mean', float('nan')) if vr_key else float('nan')
        vr_str = f"{vr:.3f}" if not np.isnan(vr) else "---"
        print(f"  {label:<22} {est:>12.6f} {err:>10.6f} {var:>12.2e} {vr_str:>10}")


if __name__ == "__main__":
    main()
