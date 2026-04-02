# LJ13 Evaluation & Analysis Plan

**System:** LJ13 (13 particles × 3D = 39D), Lennard-Jones Energy
**Checkpoint:** `results/local/2026.03.31/212242/checkpoints/checkpoint_latest.pt`
**Output:** `RESULT/` (plots + filled RESULT.md)
**GPU note:** LJ55 training uses significant VRAM — LJ13 eval is moderate-size, check `nvidia-smi` before launching.

---

## Phase 1: Ground Truth Computation

Compute ground truth mean energy from reference samples (`data/test_split_LJ13-1000.npy`).

**Script:** standalone, ~10 seconds, GPU (LJ energy needs autograd).
- Load reference samples (1000 configs of 13 particles in 3D)
- Instantiate LennardJonesEnergy (requires bgflow)
- Compute `energy.eval(ref_samples).mean()` → ground truth value
- Also compute std, median, percentiles for context

**Output:** printed value → manually recorded for all subsequent phases.

**Note:** LJ energy evaluation involves pairwise distance computation (13×12/2 = 78 pairs) — fast but needs GPU for autograd gradients.

---

## Phase 2: Main Evaluation (10 seeds × 4 sample sizes × 7 methods)

Run `run_evaluation.py` which calls `full_evaluation()`:
- Seeds: 0–9
- Sample sizes: N ∈ {100, 500, 1000, 2000}
- Methods per run: naive, stein_cv, antithetic, mcmc, hybrid, gen_stein, neural_cv
- Total: 10 × 4 = 40 single_run_evaluations

**Estimated time:** ~30–60 min total. LJ13 is 39D so:
- Stein kernel matrix (O(N²d)) is ~5× more expensive than DW4 (8D)
- Neural CV: 500 epochs, exact divergence or Hutchinson, hidden_dim=128
- Neural CV training per run: ~30–60s on GPU (39 backward passes for exact div)
- MCMC: MH steps with LJ energy + autograd grad_E — slower than DW4

**Script:** `run_evaluation.py` with hydra overrides.
**Output:** `eval_results/lj13_asbs/results.json` + 6 plot PNGs.

**Important:** The run_evaluation.py also runs MCMC ablation (K=0,5,10,20,50) — 5 additional single runs.

---

## Phase 3: Copy Plots & Fill RESULT.md

1. Copy generated PNGs from `eval_results/lj13_asbs/` → `RESULT/` with `lj13_` prefix
2. Parse `results.json` to extract all numbers
3. Fill LJ13 sections of `RESULT/RESULT.md`:
   - Summary table (N=2000)
   - Diagnostics (KSD², MH acceptance, antithetic correlation)
   - Uncomment image links
   - Write observations

**LJ13-specific observations to look for:**
- Does Stein CV variance reduction degrade vs DW4? (39D kernel may be less effective)
- Does Neural CV outperform RKHS Stein CV? (expected — 39D is in the crossover regime)
- MH acceptance rate — LJ13 has a rugged energy landscape, expect lower acceptance than DW4
- Antithetic correlation — may be weaker in higher dimensions

---

## Phase 4: Cross-System Partial Fill

With LJ13 done, fill the LJ13 column in Section 7 cross-system tables:
- 7.1 Variance Reduction Factor
- 7.2 Absolute Error
- 7.4 KSD²
- 7.5 Computational Cost (time the eval)

Compare with DW4 column to begin identifying dimension-dependent trends.

---

## Execution Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4
  (~10s)   (~30-60m)  (~5m)     (~5m)
```

Phase 2 is the bottleneck. **Must verify GPU memory** — if LJ55 training is consuming most VRAM, LJ13 eval may need to wait or use smaller batch sizes.

---

## Phase 5: Advanced Methods — EGNN Stein CV & RBF Collocation

Two new Stein CV variants (same as DW4 Phase 5) adapted for the 39D particle system.

### 5A. EGNN Stein CV (Task 12)

**What:** Replace the MLP in Neural Stein CV with an EGNN backbone that respects E(3) equivariance.

**LJ13 config:** `n_particles=13, spatial_dim=3, hidden_nf=128, n_layers=5`
**Epochs:** 1000 (more than MLP — equivariance constraints slow convergence but improve generalization)
**Divergence:** Hutchinson (2–3 probes) — EGNN forward is expensive with 13 particles, avoid 39 backward passes

**Why LJ13 is the key test case:** 39D is where MLP Neural CV should start struggling (curse of dimensionality) while EGNN exploits the physical symmetry of the 13-particle system. If EGNN doesn't clearly beat MLP here, it won't matter for LJ55 either.

### 5B. RBF Collocation CV (Task 13)

**What:** Mesh-free RBF collocation solve for the differentiated Poisson equation.

**LJ13 config:** `n_centers=300, ell=median_heuristic, reg_lambda=1e-6`
- Basis size: 300 centers × 39 dims = 11,700 basis functions
- Matrix: ~78,000 × 11,700 — larger than DW4 but still tractable (seconds, not minutes)

**Requires:** LJ energy Hessian — must use autograd (`torch.autograd.functional.hessian` or double backward). 39D Hessian is 39×39 = 1,521 entries per sample. May need batched computation.

**Scaling concern:** RBF collocation matrix grows as (N×d) × (M_c×d). At 39D this is on the boundary of tractability. May need to reduce N_collocation or use iterative solver instead of direct `lstsq`.

### 5C. Evaluation

Run both new methods on the same LJ13 checkpoint with 10 seeds × 4 sample sizes. Add as methods #8 and #9.

| Method | Expected Behavior (LJ13) |
|--------|--------------------------|
| EGNN Stein CV | Should clearly beat MLP Neural CV — 39D is where equivariance matters |
| RBF Collocation | May struggle — 39D is high for RBF basis, expect weaker variance reduction than DW4 |

### 5D. Execution

```
Phase 5A: Implement/adapt egnn_stein_cv.py for LJ13  (~15 min if DW4 version exists)
Phase 5B: Implement/adapt rbf_collocation_cv.py for LJ13  (~20 min, Hessian is the work)
Phase 5C: Re-run LJ13 eval with 9 methods  (~45 min GPU)
Phase 5D: Update RESULT.md with new results
```

---

## Phase 6: KSH-Style Enhancements — Full Pipeline (ISM + Stein CV + Score-Informed)

LJ13 is the **primary testbed** for KSH-style methods. At 39D, this is where:
- ISM score matching becomes meaningful (q_θ ≠ π is detectable)
- Score-informed CV decomposition provides real benefit over basic CV
- Hutchinson divergence is necessary (exact = 39 backward passes)

### 6A. Phase 1 — Implicit Score Matching (ISM)

**What:** Train `ScoreEGNN` via ISM loss to learn s_ϕ ≈ ∇log q_θ.

**Files:** `enhancements/score_matching.py` (ScoreEGNN, ImplicitScoreModel)

**LJ13 config:**
- Score model: `ScoreEGNN(n_particles=13, spatial_dim=3, hidden_nf=128, n_layers=4, coord_init_gain=0.1)`
- ISM training: `lr=3e-4, n_iters=5000, batch_size=256`
- Divergence: Hutchinson with `n_probes=8` (more probes for accurate score)
- Grad clip: 5.0

**Diagnostic:** After training, `diagnose()` should show:
- cos_sim(s_ϕ, -∇E) > 0.8 (good ASBS → s_ϕ ≈ -∇E)
- ||s_ϕ + ∇E|| small (residual measures how far q_θ is from π)

**Estimated time:** ~10 min (5000 iters, LJ energy is moderate cost)

### 6B. Phase 2a — Basic SteinBiasCorrector (Variance Loss)

**What:** Train `SteinEGNN_LN` by minimizing `Var[f + T_ν g]`.

**LJ13 config:**
- g-network: `SteinEGNN_LN(n_particles=13, spatial_dim=3, hidden_nf=128, n_layers=5)`
- Training: `n_iters=5000, batch_size=256, lr=1e-3`
- Validation: `val_fraction=0.2, patience=6`
- Divergence: Hutchinson with `n_probes=4`
- Grad clip: 5.0

**Per-observable training:** Separate g-network trained for each observable (energy, interatomic_dist).

### 6C. Phase 2b — ScoreInformedSteinCV (Score Decomposition)

**What:** Decompose g = α·g_init + g_res where g_init = -(∇E + s_ϕ).

**Files:** `enhancements/variance_stein_cv.py` (ScoreInformedSteinCV)

**LJ13 config:**
- Residual network: `SteinEGNN_LN(n_particles=13, spatial_dim=3, hidden_nf=128, n_layers=5)`
- Uses pre-trained ISM from Phase 1
- Training: `n_iters=5000, batch_size=256, lr=1e-3`
- Loss: `mean((h - running_mean)²)` with EMA decay=0.99
- α initialized at 0.01 (log-parameterized)
- Divergence: Hutchinson with `n_probes=4`

**Key insight:** T_ν g_init is precomputed once and frozen. This is cheap at evaluation but expensive to compute (∇E + ISM + divergence for all N samples). Budget ~5 min for precomputation.

### 6D. Evaluation Protocol

**Script:** `experiments/lj13_ksh_steincv.py`

**Protocol:** 200 bootstrap trials with subsample_size=200
- For each trial: random permutation of N samples → take first 200
- Compute naive, basic_cv, score_cv estimates on each subsample
- Report: mean, |bias|, var, MSE per method

**Comparison targets:**
| Method | Expected MSE Reduction vs Naive |
|--------|-------------------------------|
| Basic Stein CV (var loss) | 2–5× |
| Score+Stein CV | 5–15× (if ISM is good) |
| vs SML Neural CV (PDE loss) | Comparable or better (equivariance helps at 39D) |

### 6E. Results to Capture

Add to RESULT.md:
- New method rows: "SteinBiasCorrector (Var loss)", "Score-Informed CV"
- ISM diagnostics: cos_sim, ||s_ϕ + ∇E||
- Per-observable tables (energy AND interatomic distance)
- α convergence value (how much the score component contributes)
- Comparison: PDE loss (SML) vs Variance loss (KSH) on same architecture

### 6F. Execution Order

```
Phase 6A: ISM training                         (~10 min GPU)
Phase 6B: Basic Stein CV × 2 observables       (~15 min GPU)
Phase 6C: Score-informed CV × 2 observables     (~15 min GPU)
Phase 6D: Evaluation (200 trials)               (~5 min)
Phase 6E: Extract results, update RESULT.md     (~5 min)
Total: ~50 min
```

**Dependencies:** Phase 6A must complete before 6C (ISM needed for score-informed). Phases 6B and 6A can run sequentially. Phase 6D needs both 6B and 6C.

**GPU note:** ISM + 2×basic CV + 2×score-informed = 5 separate training runs. Each is moderate (5000 iters, LJ13 energy). Total GPU usage: ~2–4 GB — can coexist with LJ55 training if needed.
