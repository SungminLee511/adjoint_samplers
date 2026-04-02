# DW4 Evaluation & Analysis Plan

**System:** DW4 (4 particles × 2D = 8D), Double Well Energy
**Checkpoint:** `results/local/2026.03.31/152919/checkpoints/checkpoint_latest.pt`
**Output:** `RESULT/` (plots + filled RESULT.md)
**GPU note:** LJ13 training uses ~4GB/80GB — DW4 eval can run concurrently.

---

## Phase 1: Ground Truth Computation

Compute ground truth mean energy from 10,000 reference samples (`data/test_split_DW4.npy`).

**Script:** standalone, ~10 seconds, CPU-only.
- Load reference samples
- Instantiate DoubleWellEnergy
- Compute `energy.eval(ref_samples).mean()` → ground truth value
- Also compute std, median, percentiles for context

**Output:** printed value → manually recorded for all subsequent phases.

---

## Phase 2: Main Evaluation (10 seeds × 4 sample sizes × 7 methods)

Run `run_evaluation.py` which calls `full_evaluation()`:
- Seeds: 0–9
- Sample sizes: N ∈ {100, 500, 1000, 2000}
- Methods per run: naive, stein_cv, antithetic, mcmc, hybrid, gen_stein, neural_cv
- Total: 10 × 4 = 40 single_run_evaluations

**Estimated time:** ~12 min total (benchmarked: 16s/run × 45 runs). Neural CV 500-epoch training = 11s on GPU. All 7 methods together = 16s/run at N=2000.

**Script:** `run_evaluation.py` with hydra overrides.
**Output:** `eval_results/dw4_asbs/results.json` + 6 plot PNGs.

**Important:** The run_evaluation.py also runs MCMC ablation (K=0,5,10,20,50) — 5 additional single runs.

---

## Phase 3: Copy Plots & Fill RESULT.md

1. Copy generated PNGs from `eval_results/dw4_asbs/` → `RESULT/` with `dw4_` prefix
2. Parse `results.json` to extract all numbers
3. Fill DW4 sections (4.2–4.5) of `RESULT/RESULT.md`:
   - Summary table (N=2000)
   - Diagnostics (KSD², MH acceptance, antithetic correlation)
   - Uncomment image links
   - Write observations

---

## Phase 4: Cross-System Partial Fill

With DW4 done, fill the DW4 column in Section 7 cross-system tables:
- 7.1 Variance Reduction Factor
- 7.2 Absolute Error
- 7.4 KSD²
- 7.5 Computational Cost (time the eval)

---

## Execution Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4
  (~10s)    (~12m)    (~5m)     (~5m)
```

Phase 2 is the bottleneck. Runs on GPU alongside LJ13 training (safe — DW4 is tiny).

---

## Phase 5: Advanced Methods — EGNN Stein CV & RBF Collocation

Two new Stein CV variants to implement and evaluate on DW4.

### 5A. EGNN Stein CV (Task 12)

**What:** Replace the MLP in Neural Stein CV with an EGNN backbone that respects E(3) equivariance of the particle system. The EGNN architecture is already in the codebase (`EGNN_dynamics`).

**New file:** `enhancements/egnn_stein_cv.py`
- `EGNNSteinCV` class wrapping `EGNN_dynamics` with `condition_time=False`
- Small output scale (0.01) initialization
- Same training via `train_neural_stein_cv()` — just swap the model

**DW4 config:** `n_particles=4, spatial_dim=2, hidden_nf=64, n_layers=4`
**Epochs:** 1000 (more than MLP due to equivariance constraints, but should converge better)
**Divergence:** Hutchinson (1-2 probes) — EGNN forward is expensive, avoid d backward passes

**Why it should help:** MLP ignores particle structure. EGNN enforces translation/rotation equivariance by construction, so the solution space is smaller → faster convergence, less overfitting, more stable training. The DW4 MLP Neural CV had 309× variance explosion — EGNN should prevent this by constraining $g_\phi$ to physically meaningful vector fields.

### 5B. RBF Collocation CV (Task 13)

**What:** Solve the differentiated Poisson equation via mesh-free RBF collocation instead of neural net training. Expand $g(x) = \sum_k \alpha_k \psi_k(x)$ in Gaussian RBF basis, then solve the resulting **linear least-squares** in one shot.

**New file:** `enhancements/rbf_collocation_cv.py`
- `select_centers()` — pick M_c centers from samples (random or k-means)
- `build_collocation_matrix()` — analytical Gaussian derivatives for the PDE
- `rbf_collocation_cv()` — main function: build matrix → solve → estimate

**DW4 config:** `n_centers=200, ell=median_heuristic, reg_lambda=1e-6`
- Basis size: 200 centers × 8 dims = 1,600 basis functions
- Matrix: 16,000 × 1,600 — modest, solves in milliseconds

**Requires:** Energy Hessian $\partial^2 E / \partial x_l \partial x_j$ at each sample point. DoubleWellEnergy has cheap closed-form Hessian. Use autograd for LJ.

**Why it should help:** No training loop, no learning rate, no gradient clipping, no 3rd-order autograd instability. A single deterministic matrix solve. Uses the *differentiated* PDE form (eliminates unknown constant $c$), avoiding the bias problem that plagued RKHS Stein CV. Should give the stability of RKHS with the correctness of the differentiated formulation.

### 5C. Evaluation

Run both new methods on the same DW4 checkpoint with 10 seeds × 4 sample sizes. Add as methods #8 and #9 in the summary table:

| Method | Expected Behavior |
|--------|-------------------|
| EGNN Stein CV | More stable than MLP Neural CV, better variance reduction |
| RBF Collocation | Most stable of all (no training), competitive with RKHS on variance |

Update RESULT.md Section 4.2 table with two new rows.

### 5D. Execution

```
Phase 5A: Implement egnn_stein_cv.py  (~30 min coding)
Phase 5B: Implement rbf_collocation_cv.py (~45 min coding)
Phase 5C: Re-run DW4 eval with 9 methods (~15 min GPU)
Phase 5D: Update RESULT.md with new results
```

---

## Phase 6: KSH-Style Enhancements — Variance-Loss Stein CV

KSH-style methods ported from `KSH_ASBS/stein_cv/`. These use a **different loss function** (Var[h] instead of ‖∇h‖²), an **equivariant g-network** (SteinEGNN_LN), and **validation-based early stopping**.

### 6A. SteinEGNN_LN + SteinBiasCorrector (Variance Loss)

**What:** Train `SteinEGNN_LN` (EGNN with LayerNorm) by minimizing `Var[f + T_ν g]` directly, with train/val split and early stopping.

**Files:** `enhancements/egnn_stein_cv.py` (SteinEGNN_LN), `enhancements/variance_stein_cv.py` (SteinBiasCorrector)

**DW4 config:**
- g-network: `SteinEGNN_LN(n_particles=4, spatial_dim=2, hidden_nf=64, n_layers=4, tanh=False)`
- Training: `n_iters=10000, batch_size=2500, lr=1e-3, cosine_lr=True`
- Validation: `val_fraction=0.2, patience=6`
- Divergence: exact (dim=8, cheap)
- Grad clip: 5.0

**Script:** `experiments/dw4_ksh_steincv.py`
- 3 seeds × 2 observables (energy, interatomic distance)
- Bootstrap evaluation (2000 resamples)
- Reports: bias, MSE, reduction ratios

**Why this matters:** Compares two different optimization strategies for the same problem:
- SML PDE loss: minimizes ‖∇h‖², needs ∇f, no validation
- KSH variance loss: minimizes Var[h], doesn't need ∇f, has early stopping
- Same architecture (EGNN) → pure loss-function comparison

**Estimated time:** ~15 min (3 seeds × 2 obs × 10k iters each, but DW4 is small)

### 6B. Results to Capture

Add to RESULT.md:
- New method row: "SteinEGNN_LN (Var loss)" in Section 4.2 summary table
- Compare variance reduction: PDE loss vs variance loss on same EGNN architecture
- Compare bias reduction: does early stopping help?
- Observable-specific results: energy AND interatomic distance (SML only tracks energy)

### 6C. Execution

```
Phase 6A: Run dw4_ksh_steincv.py  (~15 min GPU)
Phase 6B: Extract results, add to RESULT.md  (~5 min)
```

**Note:** No ScoreInformedSteinCV for DW4 — ISM score matching is overkill for 8D. KSH also only uses basic SteinBiasCorrector for DW4.
