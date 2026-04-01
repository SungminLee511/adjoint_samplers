"""
enhancements/visualization.py

Generate publication-quality plots comparing all enhancement methods.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from typing import Optional


# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'naive': '#1f77b4',
    'stein_cv': '#ff7f0e',
    'antithetic': '#2ca02c',
    'mcmc': '#d62728',
    'hybrid': '#9467bd',
    'gen_stein': '#8c564b',
    'neural_cv': '#e377c2',
}

LABELS = {
    'naive': 'Vanilla ASBS',
    'stein_cv': 'Stein CV (RKHS)',
    'antithetic': 'Antithetic',
    'mcmc': 'MCMC Corrected',
    'hybrid': 'MCMC + Stein CV',
    'gen_stein': 'Generator Stein CV',
    'neural_cv': 'Neural Stein CV',
}


def plot_estimation_error_vs_samples(
    results: dict,
    save_path: str,
    gt_mean_energy: Optional[float] = None,
    title: str = "Mean Energy Estimation Error vs Sample Size",
):
    """Plot |estimated - ground truth| for each method vs N."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sample_sizes = sorted(results.keys())

    methods = [
        ('error_naive', 'naive'),
        ('error_stein', 'stein_cv'),
        ('error_anti', 'antithetic'),
        ('error_mcmc', 'mcmc'),
        ('error_hybrid', 'hybrid'),
        ('error_gen_stein', 'gen_stein'),
        ('error_neural_cv', 'neural_cv'),
    ]

    for metric_key, method_key in methods:
        if metric_key not in results[sample_sizes[0]]:
            continue

        means = [results[n][metric_key]['mean'] for n in sample_sizes]
        stds = [results[n][metric_key]['std'] for n in sample_sizes]

        ax.errorbar(
            sample_sizes, means, yerr=stds,
            label=LABELS[method_key],
            color=COLORS[method_key],
            marker='o', markersize=5,
            linewidth=2, capsize=3,
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Samples (N)')
    ax.set_ylabel('|Estimated - Ground Truth|')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_variance_comparison(
    results: dict,
    save_path: str,
    title: str = "Estimator Variance vs Sample Size",
):
    """Plot variance of each estimator vs N."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sample_sizes = sorted(results.keys())

    variance_methods = [
        ('naive_var', 'naive'),
        ('stein_cv_var', 'stein_cv'),
        ('anti_var', 'antithetic'),
        ('mcmc_var', 'mcmc'),
        ('hybrid_var', 'hybrid'),
        ('gen_stein_var', 'gen_stein'),
        ('neural_cv_var', 'neural_cv'),
    ]

    for var_key, method_key in variance_methods:
        if var_key not in results[sample_sizes[0]]:
            continue

        means = [results[n][var_key]['mean'] for n in sample_sizes]
        stds = [results[n][var_key]['std'] for n in sample_sizes]

        ax.errorbar(
            sample_sizes, means, yerr=stds,
            label=LABELS[method_key],
            color=COLORS[method_key],
            marker='o', markersize=5,
            linewidth=2, capsize=3,
        )

    # Reference line: O(1/N) scaling
    N0 = sample_sizes[0]
    var0 = results[N0]['naive_var']['mean']
    ref_line = [var0 * N0 / n for n in sample_sizes]
    ax.plot(sample_sizes, ref_line, 'k--', alpha=0.3, label='$O(1/N)$ reference')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Samples (N)')
    ax.set_ylabel('Estimator Variance')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_variance_reduction_factors(
    results: dict,
    save_path: str,
    title: str = "Variance Reduction Factor (lower = better)",
):
    """Bar chart of variance reduction factors at fixed N."""
    sample_sizes = sorted(results.keys())
    N = sample_sizes[-1]  # Use largest N

    methods = [
        ('stein_var_reduction', 'Stein CV (RKHS)'),
        ('anti_var_reduction', 'Antithetic'),
        ('neural_cv_var_reduction', 'Neural Stein CV'),
    ]

    # Also compute hybrid / naive ratio
    hybrid_ratio_mean = None
    if 'hybrid_var' in results[N] and 'naive_var' in results[N]:
        hybrid_ratio_mean = results[N]['hybrid_var']['mean'] / (
            results[N]['naive_var']['mean'] + 1e-20
        )
        methods.append(('_hybrid_ratio', 'MCMC + Stein CV'))

    fig, ax = plt.subplots(figsize=(8, 4))

    names = []
    means = []
    stds = []

    for key, label in methods:
        if key.startswith('_'):
            names.append(label)
            means.append(hybrid_ratio_mean)
            stds.append(0)
        elif key in results[N]:
            names.append(label)
            means.append(results[N][key]['mean'])
            stds.append(results[N][key]['std'])

    colors = ['#ff7f0e', '#2ca02c', '#e377c2', '#9467bd'][:len(names)]
    bars = ax.bar(names, means, yerr=stds, color=colors,
                  capsize=5, edgecolor='black', linewidth=0.5)

    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5,
               label='No improvement (ratio = 1)')
    ax.set_ylabel('Var(enhanced) / Var(naive)')
    ax.set_title(f'{title}\n(N = {N}, averaged over seeds)')
    ax.legend()

    # Annotate bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_ksd_across_samples(
    results: dict,
    save_path: str,
    title: str = "KSD² vs Sample Size",
):
    """Plot KSD² (should be roughly constant in N, measures distributional gap)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    sample_sizes = sorted(results.keys())
    means = [results[n]['ksd_squared']['mean'] for n in sample_sizes]
    stds = [results[n]['ksd_squared']['std'] for n in sample_sizes]

    ax.errorbar(sample_sizes, means, yerr=stds,
                color='#e377c2', marker='s', markersize=6,
                linewidth=2, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel('Number of Samples (N)')
    ax.set_ylabel('KSD²')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_mcmc_ablation(
    results_by_mh_steps: dict,
    save_path: str,
    gt_mean_energy: float,
    title: str = "Effect of MCMC Steps on Estimation Error",
):
    """Plot estimation error vs number of MH correction steps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    steps = sorted(results_by_mh_steps.keys())

    # Left: error vs MH steps
    errors_mcmc = [abs(results_by_mh_steps[k]['mcmc_mean_energy'] - gt_mean_energy)
                   for k in steps]
    errors_hybrid = [abs(results_by_mh_steps[k]['hybrid_estimate'] - gt_mean_energy)
                     for k in steps]

    ax1.plot(steps, errors_mcmc, 'o-', color=COLORS['mcmc'],
             label='MCMC only', linewidth=2, markersize=6)
    ax1.plot(steps, errors_hybrid, 's-', color=COLORS['hybrid'],
             label='MCMC + Stein CV', linewidth=2, markersize=6)
    ax1.axhline(y=abs(results_by_mh_steps[0]['naive_mean_energy'] - gt_mean_energy),
                color=COLORS['naive'], linestyle='--', label='Vanilla (no MCMC)')
    ax1.set_xlabel('MH Steps (K)')
    ax1.set_ylabel('|Estimated - Ground Truth|')
    ax1.set_title('Estimation Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: acceptance rate vs MH steps
    acc_rates = [results_by_mh_steps[k].get('mcmc_acceptance', 0) for k in steps]
    ax2.plot(steps, acc_rates, 'o-', color='#17becf', linewidth=2, markersize=6)
    ax2.set_xlabel('MH Steps (K)')
    ax2.set_ylabel('Acceptance Rate')
    ax2.set_title('MH Acceptance Rate')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_antithetic_correlation(
    results: dict,
    save_path: str,
    title: str = "Antithetic Correlation vs Sample Size",
):
    """Plot the correlation between original and antithetic trajectories."""
    fig, ax = plt.subplots(figsize=(6, 4))

    sample_sizes = sorted(results.keys())
    means = [results[n]['anti_correlation']['mean'] for n in sample_sizes]
    stds = [results[n]['anti_correlation']['std'] for n in sample_sizes]

    ax.errorbar(sample_sizes, means, yerr=stds,
                color=COLORS['antithetic'], marker='o', markersize=6,
                linewidth=2, capsize=3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlabel('Number of Samples (N)')
    ax.set_ylabel("Correlation(f(X), f(X'))")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax.text(0.95, 0.95, 'Negative = variance reduced',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, style='italic', alpha=0.6)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_summary_table(
    results: dict,
    save_path: str,
    gt_mean_energy: Optional[float] = None,
    title: str = "Summary",
):
    """Generate a summary table as an image."""
    sample_sizes = sorted(results.keys())
    N = sample_sizes[-1]
    r = results[N]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    def fmt_mean_std(key):
        if key in r:
            return f"{r[key]['mean']:.4f} +/- {r[key]['std']:.4f}"
        return "N/A"

    def fmt_val(key, fmt='.2e'):
        if key in r:
            return f"{r[key]['mean']:{fmt}}"
        return "N/A"

    def fmt_err(key):
        if key in r:
            m = r[key]['mean']
            if isinstance(m, float):
                return f"{m:.4f}"
        return "N/A"

    rows = [
        ['Vanilla ASBS', fmt_mean_std('naive_mean_energy'),
         fmt_val('naive_var'), fmt_err('error_naive'), "1.000"],
        ['Stein CV (RKHS)', fmt_mean_std('stein_cv_estimate'),
         fmt_val('stein_cv_var'), fmt_err('error_stein'),
         fmt_val('stein_var_reduction', '.3f')],
        ['Antithetic', fmt_mean_std('anti_estimate'),
         fmt_val('anti_var'), fmt_err('error_anti'),
         fmt_val('anti_var_reduction', '.3f')],
        ['MCMC (K=10)', fmt_mean_std('mcmc_mean_energy'),
         fmt_val('mcmc_var'), fmt_err('error_mcmc'), "---"],
        ['MCMC + Stein CV', fmt_mean_std('hybrid_estimate'),
         fmt_val('hybrid_var'), fmt_err('error_hybrid'), "---"],
        ['Generator Stein CV', fmt_mean_std('gen_stein_estimate'),
         fmt_val('gen_stein_var'), fmt_err('error_gen_stein'), "---"],
        ['Neural Stein CV', fmt_mean_std('neural_cv_estimate'),
         fmt_val('neural_cv_var'), fmt_err('error_neural_cv'),
         fmt_val('neural_cv_var_reduction', '.3f')],
    ]

    if gt_mean_energy is not None:
        rows.insert(0, ['Ground Truth', f"{gt_mean_energy:.4f}", '---', '0', '---'])

    col_labels = ['Method', 'Mean Energy', 'Variance', '|Error|', 'Var Ratio']

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color the header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    n_seeds = len(r['naive_mean_energy']['values']) if 'naive_mean_energy' in r else '?'
    ax.set_title(f'{title} (N = {N}, {n_seeds} seeds)',
                 fontsize=14, pad=20)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_all_plots(
    results: dict,
    output_dir: str,
    experiment_name: str,
    gt_mean_energy: Optional[float] = None,
):
    """Generate all plots for a given experiment."""
    out = Path(output_dir) / experiment_name
    out.mkdir(parents=True, exist_ok=True)

    plot_estimation_error_vs_samples(
        results, str(out / 'error_vs_N.png'),
        gt_mean_energy=gt_mean_energy,
        title=f'{experiment_name}: Estimation Error vs N',
    )

    plot_variance_comparison(
        results, str(out / 'variance_vs_N.png'),
        title=f'{experiment_name}: Estimator Variance vs N',
    )

    plot_variance_reduction_factors(
        results, str(out / 'variance_reduction_bars.png'),
        title=f'{experiment_name}: Variance Reduction',
    )

    plot_ksd_across_samples(
        results, str(out / 'ksd_vs_N.png'),
        title=f'{experiment_name}: KSD^2',
    )

    plot_antithetic_correlation(
        results, str(out / 'antithetic_correlation.png'),
        title=f'{experiment_name}: Antithetic Correlation',
    )

    plot_summary_table(
        results, str(out / 'summary_table.png'),
        gt_mean_energy=gt_mean_energy,
        title=experiment_name,
    )

    print(f"\nAll plots saved to {out}/")
