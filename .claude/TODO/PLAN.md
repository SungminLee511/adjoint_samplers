# Execution Plan: ASBS Training + Stein Enhancement Evaluation

## Environment
- **Conda env:** `Sampling_env` (created from `environment.yml`)
- **GPU:** NVIDIA A100 80GB PCIe
- **Working dir:** `/home/RESEARCH/adjoint_samplers`

## Phase 1: Setup
1. Create `Sampling_env` from `environment.yml`
2. Download reference samples: `bash scripts/download.sh` → `data/`

## Phase 2: Training (Sequential — one GPU)

All training uses `use_wandb=false` and runs via nohup.

| Experiment | Dims | NFE | Epochs | Log File | Checkpoint Dir |
|-----------|------|-----|--------|----------|----------------|
| DW4 ASBS | 12D (4×3) | 200 | 5000 | `logs/train_dw4.log` | `results/local/2026.03.31/152919/checkpoints/` |
| LJ13 ASBS | 39D (13×3) | 1000 | 5000 | `logs/train_lj13.log` | `results/local/2026.03.31/212242/checkpoints/` |
| LJ55 ASBS | 165D (55×3) | 1000 | 5000 | `logs/train_lj55.log` | TBD |

### Commands
```bash
# DW4 (fastest — start here)
/root/miniconda3/envs/Sampling_env/bin/python -u train.py experiment=dw4_asbs seed=0 use_wandb=false > logs/train_dw4.log 2>&1 &

# LJ13 (run after DW4 finishes)
/root/miniconda3/envs/Sampling_env/bin/python -u train.py experiment=lj13_asbs seed=0 use_wandb=false > logs/train_lj13.log 2>&1 &

# LJ55 (run after LJ13 finishes)
/root/miniconda3/envs/Sampling_env/bin/python -u train.py experiment=lj55_asbs seed=0 use_wandb=false > logs/train_lj55.log 2>&1 &
```

## Phase 3: Enhanced Evaluation (after each training completes)

Run `run_evaluation.py` on each checkpoint. Now includes **7 enhancement methods** (added Neural Stein CV in v2).

| Experiment | Log File | Output Dir |
|-----------|----------|------------|
| DW4 | `logs/eval_dw4.log` | `eval_results/dw4_asbs/` |
| LJ13 | `logs/eval_lj13.log` | `eval_results/lj13_asbs/` |
| LJ55 | `logs/eval_lj55.log` | `eval_results/lj55_asbs/` |

### Commands
```bash
# DW4 eval (checkpoint from Hydra output dir)
/root/miniconda3/envs/Sampling_env/bin/python -u run_evaluation.py experiment=dw4_asbs checkpoint=results/local/2026.03.31/152919/checkpoints/checkpoint_latest.pt output_dir=eval_results > logs/eval_dw4.log 2>&1 &

# LJ13 eval
/root/miniconda3/envs/Sampling_env/bin/python -u run_evaluation.py experiment=lj13_asbs checkpoint=results/local/2026.03.31/212242/checkpoints/checkpoint_latest.pt output_dir=eval_results > logs/eval_lj13.log 2>&1 &
```

### What the evaluation produces
- `eval_results/<exp>/results.json` — raw metrics across 10 seeds × 4 sample sizes
- `eval_results/<exp>/error_vs_N.png` — estimation error scaling
- `eval_results/<exp>/variance_vs_N.png` — variance scaling
- `eval_results/<exp>/variance_reduction_bars.png` — bar chart of var ratios (now includes Neural CV)
- `eval_results/<exp>/ksd_vs_N.png` — KSD diagnostic
- `eval_results/<exp>/antithetic_correlation.png` — antithetic correlation
- `eval_results/<exp>/mcmc_ablation.png` — MH steps ablation
- `eval_results/<exp>/summary_table.png` — summary table image (now includes Neural CV row)

## Phase 4: Analysis

### Key questions per experiment
| Enhancement | Question | Metric |
|-------------|----------|--------|
| KSD | Does KSD track W2 metrics? | ksd_squared |
| Stein CV (RKHS) | How much variance + bias reduction? | stein_var_reduction, error_stein |
| Antithetic | Is correlation negative? | anti_correlation |
| MCMC | Does mean energy shift toward GT? | error_mcmc vs error_naive |
| Hybrid | Best of both worlds? | error_hybrid, hybrid_var |
| Generator CV | Better than standard Stein? | variance_gen_stein vs stein_cv_var |
| Neural Stein CV | Scales to high d? Better than RKHS? | neural_cv_var vs stein_cv_var |

### Expected scaling behavior
- **DW4 (12D):** All enhancements should work well. Kernel methods effective in low d. Neural CV and RKHS CV should perform similarly.
- **LJ13 (39D):** Moderate. RKHS Stein CV may degrade slightly. Neural CV should start showing advantage.
- **LJ55 (165D):** Stress test. RBF kernels degrade in high d. Neural CV strongly preferred — should outperform RKHS.

### Key theoretical result to validate: Bias-Variance Coupling
- Compare `error_stein < error_naive` consistently — confirms Stein CVs reduce bias simultaneously with variance
- A 4× variance reduction should give up to 2× bias reduction (square root relationship)
- Compare Regime 1 (Stein CV alone) vs Regime 2 (MCMC + Stein CV) tradeoff

## Status
- [x] Phase 1: Setup
- [x] Phase 2: Training — DW4 (5000 epochs, checkpoint at `results/local/2026.03.31/152919/`)
- [x] Phase 2: Training — LJ13 (5000 epochs done, in corrector phase, checkpoint at `results/local/2026.03.31/212242/`)
- [ ] Phase 2: Training — LJ55
- [ ] Phase 3: Eval — DW4
- [ ] Phase 3: Eval — LJ13
- [ ] Phase 3: Eval — LJ55
- [ ] Phase 4: Analysis

---

## Session Notes (2026-03-31 KST)

### Environment Setup
- **Rebuilt `Sampling_env` from scratch** (Python 3.10, not 3.11 as in environment.yml — works fine)
- Installed via **pip** (not conda) to avoid solver hangs:
  - `torch 2.5.1+cu121` (from pytorch whl index)
  - `torchvision 0.20.1+cu121`, `torchaudio 2.5.1+cu121`
  - `numpy 1.26.4`, `matplotlib 3.8.0`, `rdkit 2025.9.6`, `openmm 8.5.0`
  - `torch_geometric 2.7.0`, `bgflow` (git+atong01), `mace-torch 0.3.8`
  - `hydra-core`, `wandb`, `scikit-learn`, `mdtraj`, `einops`, `POT`, `torchquad`, `submitit`, etc.
- **openmmtools: NOT INSTALLED** — only available via conda-forge, solver kept hanging (99% CPU for 5+ min each attempt). Skipped for now.
  - Not needed for DW4. Likely needed for LJ13/LJ55 — revisit before those runs.
  - Possible workaround: install from source or use a minimal conda env just for openmmtools deps.

### DW4 Training — COMPLETED
- **Checkpoint:** `results/local/2026.03.31/152919/checkpoints/checkpoint_latest.pt`
- **Final loss:** ~4.69 (stable, converged)

### LJ13 Training — COMPLETED (corrector phase finishing)
- **Checkpoint:** `results/local/2026.03.31/212242/checkpoints/checkpoint_latest.pt`
- **Training phase:** 5000 adjoint epochs complete, corrector phase at epoch ~2227

### How to Resume After Timeout
1. **Check if training finished:**
   ```bash
   ps aux | grep "train.py" | grep -v grep
   tail -20 /home/RESEARCH/adjoint_samplers/logs/train_dw4.log
   tail -20 /home/RESEARCH/adjoint_samplers/logs/train_lj13.log
   ```
2. **Find checkpoints:**
   ```bash
   ls /home/RESEARCH/adjoint_samplers/results/local/2026.03.31/*/checkpoints/checkpoint_latest.pt
   ```
3. **Run eval (DW4):**
   ```bash
   cd /home/RESEARCH/adjoint_samplers
   /root/miniconda3/envs/Sampling_env/bin/python -u run_evaluation.py experiment=dw4_asbs checkpoint=results/local/2026.03.31/152919/checkpoints/checkpoint_latest.pt output_dir=eval_results > logs/eval_dw4.log 2>&1 &
   ```

---

## Session Notes (2026-04-01 KST)

### v2 Spec Integration
- Added `enhancements/neural_stein_cv.py` — Task 11: Neural Stein CV via differentiated Poisson equation
- Updated all eval pipelines (`enhanced_evaluator.py`, `evaluation.py`, `eval_enhanced.py`, `run_evaluation.py`) to include neural CV
- Updated `visualization.py` — plots and summary tables now include neural CV method
- Key v2 additions:
  - **Neural Stein CV**: MLP-based CV that scales linearly in N (no N×N matrices), uses PDE loss ||∇h||² where h = f + A_p g_φ
  - **Bias-Variance Coupling theorem**: Variance reduction simultaneously reduces bias via |Bias| ≤ √(C · Var[h_g])
  - **Two regimes**: Regime 1 (Stein CV alone) vs Regime 2 (MCMC + Stein CV) — user chooses based on energy cost vs accuracy
