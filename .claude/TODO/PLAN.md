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

| Experiment | Dims | NFE | Epochs | Log File | Checkpoint |
|-----------|------|-----|--------|----------|------------|
| DW4 ASBS | 12D (4×3) | 200 | 5000 | `logs/train_dw4.log` | `checkpoints/dw4_asbs/` |
| LJ13 ASBS | 39D (13×3) | 1000 | 5000 | `logs/train_lj13.log` | `checkpoints/lj13_asbs/` |
| LJ55 ASBS | 165D (55×3) | 1000 | 5000 | `logs/train_lj55.log` | `checkpoints/lj55_asbs/` |

### Commands
```bash
# DW4 (fastest — start here)
conda run -n Sampling_env nohup python -u train.py experiment=dw4_asbs seed=0 use_wandb=false > logs/train_dw4.log 2>&1 &

# LJ13 (run after DW4 finishes)
conda run -n Sampling_env nohup python -u train.py experiment=lj13_asbs seed=0 use_wandb=false > logs/train_lj13.log 2>&1 &

# LJ55 (run after LJ13 finishes)
conda run -n Sampling_env nohup python -u train.py experiment=lj55_asbs seed=0 use_wandb=false > logs/train_lj55.log 2>&1 &
```

## Phase 3: Enhanced Evaluation (after each training completes)

Run `run_evaluation.py` on each checkpoint. Produces JSON results + publication plots.

| Experiment | Log File | Output Dir |
|-----------|----------|------------|
| DW4 | `logs/eval_dw4.log` | `eval_results/dw4_asbs/` |
| LJ13 | `logs/eval_lj13.log` | `eval_results/lj13_asbs/` |
| LJ55 | `logs/eval_lj55.log` | `eval_results/lj55_asbs/` |

### Commands
```bash
conda run -n Sampling_env nohup python -u run_evaluation.py experiment=dw4_asbs checkpoint=checkpoints/checkpoint_latest.pt output_dir=eval_results > logs/eval_dw4.log 2>&1 &
```

### What the evaluation produces
- `eval_results/<exp>/results.json` — raw metrics across 10 seeds × 4 sample sizes
- `eval_results/<exp>/error_vs_N.png` — estimation error scaling
- `eval_results/<exp>/variance_vs_N.png` — variance scaling
- `eval_results/<exp>/variance_reduction_bars.png` — bar chart of var ratios
- `eval_results/<exp>/ksd_vs_N.png` — KSD diagnostic
- `eval_results/<exp>/antithetic_correlation.png` — antithetic correlation
- `eval_results/<exp>/mcmc_ablation.png` — MH steps ablation
- `eval_results/<exp>/summary_table.png` — summary table image

## Phase 4: Analysis

### Key questions per experiment
| Enhancement | Question | Metric |
|-------------|----------|--------|
| KSD | Does KSD track W2 metrics? | ksd_squared |
| Stein CV | How much variance reduction? | stein_var_reduction |
| Antithetic | Is correlation negative? | anti_correlation |
| MCMC | Does mean energy shift toward GT? | error_mcmc vs error_naive |
| Hybrid | Best of both worlds? | error_hybrid, hybrid_var |
| Generator CV | Better than standard Stein? | variance_gen_stein vs stein_cv_var |

### Expected scaling behavior
- **DW4 (12D):** All enhancements should work well. Kernel methods effective in low d.
- **LJ13 (39D):** Moderate. Stein CV may degrade slightly.
- **LJ55 (165D):** Stress test. RBF kernels degrade in high d. If Stein CV fails here but works on DW4, it's a dimension effect (informative negative result).

## Status
- [x] Phase 1: Setup
- [ ] Phase 2: Training — DW4
- [ ] Phase 2: Training — LJ13
- [ ] Phase 2: Training — LJ55
- [ ] Phase 3: Eval — DW4
- [ ] Phase 3: Eval — LJ13
- [ ] Phase 3: Eval — LJ55
- [ ] Phase 4: Analysis
