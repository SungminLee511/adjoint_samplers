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
