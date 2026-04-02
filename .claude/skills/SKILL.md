# adjoint_samplers — Adjoint-Based Diffusion Sampling + Stein Enhancements

## Overview

Implements **Adjoint Sampling for Boltzmann-like distributions (ASBS)** with 7 Stein-based variance reduction and diagnostic enhancements. The original ASBS codebase trains a controlled SDE to sample from energy-based distributions. The `enhancements/` module adds post-hoc analysis methods.

## File Tree

```
adjoint_samplers/                    # ORIGINAL — DO NOT MODIFY
    adjoint_samplers/
        components/
            sde.py                   # BaseSDE, ControlledSDE, sdeint, Graph mixin
            matcher.py               # AdjointMatcher, AdjointVEMatcher
            evaluator.py             # SyntheticEnergyEvaluator
            model.py                 # FourierMLP, EGNN_dynamics
            term_cost.py             # GradEnergy
            buffer.py                # BatchBuffer
            state_cost.py
        energies/
            base_energy.py           # BaseEnergy (eval, grad_E, score)
            double_well_energy.py    # DoubleWellEnergy (needs bgflow)
            lennard_jones_energy.py  # LennardJonesEnergy (needs bgflow)
            dist_energy.py
        utils/
            train_utils.py           # get_timesteps(), training helpers
            eval_utils.py
            graph_utils.py           # remove_mean, COM-free utilities
            dist_utils.py
    train.py                         # Original training entry point

enhancements/                        # NEW — all Stein enhancement code
    __init__.py
    stein_kernel.py                  # RBF kernel, Stein kernel, KSD
    stein_cv.py                      # Stein control variate estimator (RKHS)
    antithetic.py                    # Antithetic SDE integration
    mcmc_correction.py               # MH post-correction
    enhanced_evaluator.py            # Unified evaluation pipeline
    generator_stein.py               # SDE generator Stein operator
    neural_stein_cv.py               # Neural Stein CV via PDE loss (SML-style)
    egnn_stein_cv.py                 # EGNN-based g-networks: EGNNSteinCV, SteinEGNN, SteinEGNN_LN
    variance_stein_cv.py             # KSH-style: SteinBiasCorrector (var loss), ScoreInformedSteinCV
    score_matching.py                # KSH-style: ScoreEGNN, ImplicitScoreModel (ISM)
    observables.py                   # Observable functions (energy, interatomic dist, histogram)
    rbf_collocation_cv.py            # RBF collocation CV
    evaluation.py                    # Multi-seed systematic evaluation
    visualization.py                 # Publication-quality plots

experiments/                         # KSH-style evaluation scripts
    dw4_ksh_steincv.py               # DW4: 3 seeds × 2 obs, SteinEGNN_LN + variance loss
    lj13_ksh_steincv.py              # LJ13: ISM + basic CV + score-informed CV

eval_enhanced.py                     # Single-run evaluation script (hydra)
run_evaluation.py                    # Full evaluation runner (hydra)

tests/
    test_stein_kernel.py
    test_stein_cv.py
    test_antithetic.py
    test_mcmc_correction.py
    test_enhanced_evaluator.py
    test_generator_stein.py
    test_eval_imports.py

configs/                             # Hydra configs
    train.yaml                       # Base config
    experiment/                      # dw4_asbs, lj13_asbs, lj55_asbs, demo_asbs, etc.
```

## Key Interfaces

### BaseEnergy (`adjoint_samplers/energies/base_energy.py`)
- `eval(x: (N,D)) -> (N,)` — energy values
- `grad_E(x: (N,D)) -> (N,D)` — energy gradients via autograd
- `score(x: (N,D)) -> (N,D)` — returns `-grad_E(x)`

### BaseSDE (`adjoint_samplers/components/sde.py`)
- `drift(t, x) -> (B,D)` — drift f(t,x)
- `diff(t) -> scalar` — diffusion coefficient g(t)
- `randn_like(x)` — noise (COM-free for Graph mixin)
- `propagate(x, dx)` — state update (COM-free for Graph mixin)

### ControlledSDE
- Wraps `ref_sde` + learned `controller`
- `drift(t, x) = ref_sde.drift(t,x) + g(t)^2 * controller(t,x)`

### sdeint(sde, state0, timesteps, only_boundary=False)
- Euler forward integration, `@torch.no_grad()`
- Returns `[x0, ..., xT]` or `(x0, xT)` if `only_boundary=True`

## Enhancement Modules

### stein_kernel.py
- `median_bandwidth(samples)` → scalar bandwidth (median heuristic)
- `rbf_kernel_matrix(x, y, ell)` → (N,M) RBF kernel
- `stein_kernel_matrix(samples, scores, ell)` → (N,N) Stein kernel
- `compute_ksd(samples, scores, ell=None)` → scalar KSD²

**Sign convention:** term2 = `s^T(x-x')/ℓ²·k`, term3 = `-(x-x')^T s'/ℓ²·k`
(Note: the math spec's "corrected" formula has wrong signs — use the derivation from ∇_x k and ∇_{x'} k)

### stein_cv.py
- `stein_cv_estimate(samples, scores, f_values, ell, reg_lambda)` → dict
- `multi_function_stein_cv(samples, scores, f_dict, ell, reg_lambda)` → dict
- Uses **normalized CF estimator**: `sum(a) / sum(b)` where `a = A^{-1}f`, `b = A^{-1}1`

### antithetic.py
- `sdeint_antithetic(sde, state0, timesteps, only_boundary)` → (states, states_anti) or (x0, x1, x1_anti)
- `antithetic_estimate(f_values, f_values_anti)` → dict with correlation, variance reduction

### mcmc_correction.py
- `mh_correct(samples, energy, n_steps, step_size)` → dict with corrected_samples, acceptance_rate
- Auto step size: `2.38/sqrt(D) * marginal_std`
- Uses `TYPE_CHECKING` guard for BaseEnergy (avoids bgflow import)

### generator_stein.py
- `generator_stein_kernel_matrix(samples, sde, ell)` → (N,N) generator Stein kernel
- `generator_stein_cv_estimate(samples, sde, f_values, ell, reg_lambda)` → dict
- Replaces `s_p(x)` with `b_theta(x,1)` (learned drift), uses `g(1)²/2` in trace term

### neural_stein_cv.py (NEW — v2)
- `NeuralSteinCV(dim, hidden_dim, n_layers, activation)` — MLP g_φ: R^d → R^d
  - Twice-differentiable activations only (SiLU/GELU/Tanh, NOT ReLU)
  - Last layer initialized small (zero bias, 1e-3 std weights)
- `compute_stein_operator(g_values, x, scores, g_func, hutchinson_samples)` → (B,) A_p g values
  - `hutchinson_samples=0` → exact divergence (d backward passes, good for d≤20)
  - `hutchinson_samples≥1` → Hutchinson estimator (good for d>20)
- `neural_stein_cv_loss(g_model, x, scores, f_values, ...)` → scalar PDE loss ||∇_x h||²
  - h(x) = f(x) + A_p g_φ(x) should be constant → gradient should be zero
  - Eliminates unknown E_p[f] via differentiation (Poisson eq trick)
- `train_neural_stein_cv(g_model, samples, energy, f_func, ...)` → dict with estimate, losses, variance_reduction
  - Adam optimizer, cosine annealing, gradient clipping (max_norm=10.0)
  - Returns h_values for all samples after training

**Scaling guidance:**
- DW4 (12D): 500 epochs, exact divergence, hidden_dim=64
- LJ13 (39D): 500 epochs, exact or Hutchinson, hidden_dim=128
- LJ55 (165D): 1000 epochs, Hutchinson (1-5 probes), hidden_dim=256

### egnn_stein_cv.py (expanded)
- `EGNNSteinCV(n_particles, spatial_dim, hidden_nf, n_layers, ...)` — wraps EGNN_dynamics, learnable output_scale
- `SteinEGNN(n_particles, spatial_dim, hidden_nf, n_layers)` — lightweight EGNN wrapper (KSH-style, no LN)
- `E_GCL_LN(hidden_nf, ...)` — E(n)-equivariant GCL with LayerNorm in all MLPs
- `SteinEGNN_LN(n_particles, spatial_dim, hidden_nf, n_layers, tanh, condition_time)` — standalone EGNN with LayerNorm
  - Preferred g-network for variance-loss Stein CV (better training stability)
  - Coord MLP last layer init: xavier_uniform_(gain=0.001)
  - Coord diff normalized: coord_diff / (norm + 1) for stable gradients
  - Edge attributes computed once (not per-layer), edge indices cached per (B, device)
  - Output: COM-free via remove_mean

### variance_stein_cv.py (KSH-style, NEW)
- `stein_operator_on_net(g_net, x_flat, score_flat, use_hutchinson, n_probes)` → (B,) Stein operator values
  - Flat-space API: detaches x, reattaches grad, computes score·g + div(g)
- `SteinBiasCorrector(g_net, use_hutchinson, n_probes)` — variance-loss Stein CV trainer
  - `fit(samples, grad_E, f_vals, lr, n_iters, batch_size, val_fraction, patience, bias_penalty, val_sampler)`
  - Loss: Var[f + T_ν g] + λ·E[T_ν g]²
  - Validation: 3 modes (fresh sampler / fixed split / train-only)
  - Early stopping (patience=6), cosine LR, grad clip=5.0
  - `estimate(samples, grad_E, f_vals)` → (mean_float, Tg_tensor)
- `ScoreInformedSteinCV(g_residual_net, score_model, use_hutchinson, n_probes)` — score-decomposition CV
  - g = α·g_init + g_res where g_init = -(∇E + s_ϕ)
  - α: log-parameterized scalar (init=0.01)
  - T_ν g_init: precomputed once (frozen), only g_res and α trained
  - Loss: mean((h - running_mean)²) with EMA decay=0.99
  - `fit(samples, energy, f_vals, ...)`, `eval_all(samples, grad_E, f_vals, Tg_init)`

### score_matching.py (KSH-style, NEW)
- `_build_edges(n_particles, n_systems, device)` → (edge_index, batch) for fully-connected graphs
- `ScoreEGNN(n_particles, spatial_dim, hidden_nf, n_layers, coord_init_gain)` — EGNN score model
  - Wraps EGNN_dynamics, condition_time=True, evaluates at t=1.0
  - Coord layer reinit with small gain (0.1) for stable ISM training
  - Forward: (x, batch, edge_index) particle format
- `ImplicitScoreModel(s_net, dim, use_hutchinson, n_probes, grad_clip, device)` — ISM trainer
  - ISM loss: E[½||s_ϕ||² + div(s_ϕ)] (Hyvärinen score matching)
  - `fit(samples, lr=3e-4, n_iters=5000, batch_size=256)`
  - `diagnose(samples, grad_E)` — prints cosine_sim(s_ϕ, -∇E) and ||s_ϕ + ∇E||

### observables.py (NEW)
- `mean_energy_observable(x, energy_fn)` → (B,) energies
- `mean_interatomic_distance(x, n_particles, spatial_dim)` → (B,) mean pairwise distances
- `interatomic_dist_histogram(x, n_particles, spatial_dim, bins, range_min, range_max)` → (bins,)
- `observable_gradient(x, observable_fn)` → (B, dim) via autograd
- All accept flat (B, dim) input (SML convention)

### enhanced_evaluator.py
- `evaluate_enhanced(samples, energy, ref_energies, mh_steps, stein_reg_lambda, max_stein_samples)` → dict
- Runs: KSD → naive → Stein CV → MCMC → hybrid (MCMC+Stein CV) → Neural CV → ground truth comparison

### evaluation.py
- `EvalConfig` dataclass: n_seeds, sample_sizes, mh_steps_list, etc.
- `single_run_evaluation(...)` → dict of scalar metrics (includes neural_cv_*)
- `full_evaluation(...)` → nested dict with mean/std across seeds
- `save_results(results, path)` → JSON export

### visualization.py
- 7 plot functions: error_vs_N, variance_vs_N, variance_reduction_bars, ksd_vs_N, mcmc_ablation, antithetic_correlation, summary_table
- `generate_all_plots(results, output_dir, experiment_name, gt_mean_energy)`
- All plots now include Neural CV as a 7th method (color: `#e377c2`)

## Environment
- **Conda env:** `Sampling_env` (created from `environment.yml`)
- **NOT** `SML_env` — this project has its own heavy deps (bgflow, rdkit, openmm, mace-torch)
- Run pattern: `/root/miniconda3/envs/Sampling_env/bin/python -u <command>`
  - Note: `conda run` + nohup swallows stdout. Use direct python binary instead.

## Running

### Train baseline
```bash
bash scripts/download.sh  # download reference samples to data/
/root/miniconda3/envs/Sampling_env/bin/python -u train.py experiment=dw4_asbs seed=0 use_wandb=false > logs/train_dw4.log 2>&1 &
```

### Run enhanced evaluation
```bash
/root/miniconda3/envs/Sampling_env/bin/python -u run_evaluation.py experiment=dw4_asbs checkpoint=results/local/2026.03.31/152919/checkpoints/checkpoint_latest.pt output_dir=eval_results > logs/eval_dw4.log 2>&1 &
```

### Log files
- Training logs: `logs/train_dw4.log`, `logs/train_lj13.log`, `logs/train_lj55.log`
- Eval logs: `logs/eval_dw4.log`, `logs/eval_lj13.log`, `logs/eval_lj55.log`
- See `.claude/TODO/PLAN.md` for full execution plan

### Checkpoint locations
- DW4: `results/local/2026.03.31/152919/checkpoints/checkpoint_latest.pt`
- LJ13: `results/local/2026.03.31/212242/checkpoints/checkpoint_latest.pt`

## Theoretical Background (v2)

### Bias-Variance Coupling (Key Result)
- Stein CV variance reduction **simultaneously reduces bias**: |Bias| ≤ √(C · Var_{q_θ}[h_g])
- A 4× variance reduction → up to 2× bias reduction (square root)
- No separate bias-correction mechanism needed — RKHS and Neural CV both benefit

### Two Regimes for MCMC + Stein CV
- **Regime 1 (CV only):** Cheaper, bias reduction depends on RKHS/neural approximation quality
- **Regime 2 (MCMC + CV):** Expensive but exact — MCMC ensures samples from p, CV provides pure variance reduction

### Neural vs RKHS Stein CV
| Property | RKHS (Task 2) | Neural PDE (SML) | Neural Var (KSH) |
|----------|---------------|------------------|------------------|
| Compute | O(N³ + N²d) | O(epochs × B × d) | O(iters × B × d) |
| Memory | O(N²) | O(net params) | O(net params) |
| High-d | Poor (kernel degrades) | Good | Good |
| g-network | N/A (kernel) | MLP | EGNN (equivariant) |
| Loss | Closed-form | ‖∇h‖² (PDE) | Var[h] |
| Needs ∇f | No | Yes | No |
| Validation | No | No | Yes (early stopping) |
| Best for | DW4 (12D) | LJ55 (165D) | Particle systems |

### KSH-Style Experiment Scripts
- `experiments/dw4_ksh_steincv.py`: 3 seeds × 2 observables, SteinEGNN_LN + SteinBiasCorrector
- `experiments/lj13_ksh_steincv.py`: ISM → basic Stein CV → score-informed CV, 200 bootstrap trials

## Gotchas

1. **bgflow dependency**: `double_well_energy.py` and `lennard_jones_energy.py` import bgflow. Use `TYPE_CHECKING` guards when importing BaseEnergy at module level.
2. **Stein kernel signs**: The math spec's "corrected" formula (Section 1.4 Note) has BOTH term2 and term3 signs wrong. Use: term2 = `+s^T(x-x')/ℓ²·k`, term3 = `-(x-x')^T s'/ℓ²·k`.
3. **Normalized CF estimator**: Raw `1^T a` diverges because the Stein RKHS can't represent constants. Must use normalized form: `sum(a)/sum(b)`.
4. **COM-free particles**: Graph SDEs use `sde.randn_like()` and `sde.propagate()` which project to zero center-of-mass. Antithetic sampling must use these, not `torch.randn_like`.
5. **O(N²) memory**: Stein kernel matrix is N×N. Cap at `max_stein_samples=2000` and subsample.
6. **sdeint is @torch.no_grad()**: No gradients flow through forward integration.
7. **Neural CV activations**: Must be **twice differentiable** (SiLU, GELU, Tanh). ReLU will crash because PDE loss involves second derivatives of g_φ.
8. **Neural CV gradient clipping**: Essential — PDE loss involves second derivatives that can explode early in training. Use `max_norm=10.0`.
9. **conda run swallows stdout**: Use direct python binary `/root/miniconda3/envs/Sampling_env/bin/python -u` instead of `conda run -n Sampling_env`.
10. **Hydra changes CWD**: Checkpoints saved by Hydra go to `results/local/<date>/<time>/checkpoints/`, not the working directory.
11. **SteinEGNN_LN edge_attr**: Computed once before all layers (not recomputed per layer). This is a design choice for the LN variant vs base EGNN.
12. **ScoreInformedSteinCV running mean**: Loss is `mean((h - running_mean)²)`, NOT `h.var()`. The EMA (0.99 decay) provides more stable centering than per-batch mean.
13. **ISM uses condition_time=True**: ScoreEGNN evaluates at t=1.0 (terminal time). It uses EGNN_dynamics with time conditioning — different from SteinEGNN_LN which has condition_time=False.
14. **_build_edges duplication**: Defined in both `score_matching.py` and used by `ScoreInformedSteinCV._precompute_Tg_init`. Import from `score_matching.py`.
15. **Variance loss grad clip = 5.0**: KSH-style modules use max_norm=5.0 (stricter than SML's 10.0 for PDE loss). This is intentional — variance loss has different gradient dynamics.
