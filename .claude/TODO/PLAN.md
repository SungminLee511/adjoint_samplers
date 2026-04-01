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
- [🔄] Phase 2: Training — DW4 ← **IN PROGRESS** (see notes below)
- [ ] Phase 2: Training — LJ13
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

### DW4 Training — Currently Running
- **PID:** 1566356 (launched directly via `/root/miniconda3/envs/Sampling_env/bin/python`)
- **Command:** `/root/miniconda3/envs/Sampling_env/bin/python -u train.py experiment=dw4_asbs seed=0 use_wandb=false`
- **Working dir:** `/home/RESEARCH/adjoint_samplers`
- **Log file:** `logs/train_dw4.log`
- **Checkpoint dir:** `checkpoints/dw4_asbs/` (expected — check Hydra output dir for actual path)
- **Started:** ~15:29 UTC (2026-03-31 00:29 KST)
- **Rate:** ~18 epochs/min (55 epochs in ~3 min)
- **Estimated finish:** ~20:00 UTC (~05:00 KST Apr 1) — **~4.5 hours total**
- **Note:** Used direct python binary, not `conda run`, because `conda run` + nohup was swallowing stdout.

### How to Resume After Timeout
1. **Check if DW4 finished:**
   ```bash
   ps aux | grep "train.py" | grep -v grep
   tail -20 /home/RESEARCH/adjoint_samplers/logs/train_dw4.log
   ```
2. **Find checkpoint:**
   ```bash
   find /home/RESEARCH/adjoint_samplers -name "*.pt" -newer logs/train_dw4.log
   ls /home/RESEARCH/adjoint_samplers/checkpoints/
   ```
3. **If DW4 done → run eval:**
   ```bash
   cd /home/RESEARCH/adjoint_samplers
   /root/miniconda3/envs/Sampling_env/bin/python -u run_evaluation.py experiment=dw4_asbs checkpoint=checkpoints/checkpoint_latest.pt output_dir=eval_results > logs/eval_dw4.log 2>&1 &
   ```
4. **If DW4 done → start LJ13 training:**
   - First check if openmmtools is needed (try import in Sampling_env)
   - If needed, attempt: `pip install openmmtools` from source or conda with longer timeout
   ```bash
   /root/miniconda3/envs/Sampling_env/bin/python -u train.py experiment=lj13_asbs seed=0 use_wandb=false > logs/train_lj13.log 2>&1 &
   ```
