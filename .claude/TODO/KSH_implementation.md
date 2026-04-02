# KSH-Style Enhancements: Implementation Plan

> **Goal:** Port the 3 KSH-exclusive enhancements into `adjoint_samplers/enhancements/` so that the SML repo has a superset of both codebases' capabilities.
>
> **Source:** `/home/RESEARCH/KSH_ASBS/stein_cv/` → Target: `/home/RESEARCH/adjoint_samplers/enhancements/`

---

## What KSH Has That SML Doesn't

| # | Component | KSH File | Exists in SML? | Action |
|---|-----------|----------|----------------|--------|
| 1 | **SteinEGNN_LN** (LayerNorm EGNN for g-networks) | `neural_cv.py` | ❌ `egnn_stein_cv.py` has SteinEGNN but no LayerNorm | Add `E_GCL_LN` + `SteinEGNN_LN` |
| 2 | **SteinBiasCorrector** (Variance-loss CV trainer) | `neural_cv.py` | ❌ SML only has PDE-loss trainer | Add as new class |
| 3 | **ScoreInformedSteinCV** (Score decomposition g = α·g_init + g_res) | `neural_cv.py` | ❌ | Add as new class |
| 4 | **ImplicitScoreModel + ScoreEGNN** (ISM score matching) | `score_matching.py` | ❌ | Add new file |
| 5 | **Observables module** (energy, interatomic dist, histograms) | `observables.py` | ❌ (only energy observable in evaluator) | Add new file |
| 6 | **Experiment scripts** (DW4 + LJ13 per-observable eval) | `experiments/` | ❌ (SML has different eval pipeline) | Add experiment scripts |

---

## Implementation Tasks

### Task 1: SteinEGNN_LN — LayerNorm EGNN g-network

**File:** `enhancements/egnn_stein_cv.py` (extend existing)

**What to add:**
- `E_GCL_LN` class — identical to `E_GCL` but with `nn.LayerNorm(hidden_nf)` in edge/node/coord MLPs
- `SteinEGNN_LN` class — same API as existing `SteinEGNN` but uses `E_GCL_LN` layers
- Helper `_unsorted_segment_sum()` (already exists in KSH, may exist in SML's model.py)

**Key implementation details from KSH:**
```python
class E_GCL_LN(nn.Module):
    def __init__(self, hidden_nf, edges_in_d=1, act_fn=nn.SiLU(),
                 recurrent=True, attention=False, tanh=False,
                 coords_range=1.0, agg="sum"):
        # edge_mlp: Linear → LayerNorm → Act → Linear → LayerNorm → Act
        # node_mlp: Linear → LayerNorm → Act → Linear (no final act)
        # coord_mlp: Linear → LayerNorm → Act → Linear(bias=False, gain=0.001)
        # coord_diff normalized: coord_diff / (norm + 1) where norm = sqrt(radial + 1e-8)

class SteinEGNN_LN(nn.Module):
    def __init__(self, n_particles=4, spatial_dim=2, hidden_nf=128,
                 n_layers=5, tanh=False, condition_time=False):
        # embedding: Linear(1, hidden_nf) — input is ones (or zeros if condition_time)
        # layers: n_layers × E_GCL_LN with attention=True, recurrent=True
        # edge_attr: squared distances (computed once, not updated per layer)
        # edges: cached per (n_batch, device) key
        # output: velocity = coord_final - coord_init, then remove_mean
```

**Differences from existing SteinEGNN in SML:**
- SML's `SteinEGNN` wraps `EGNN_dynamics` (reuses base code)
- KSH's `SteinEGNN_LN` is a standalone implementation with LayerNorm
- KSH normalizes coord_diff: `coord_diff / (norm + 1)` — SML's base EGNN may not
- KSH caches edges per `(n_batch, device)` key
- KSH's coord MLP last layer init: `xavier_uniform_(gain=0.001)`

**Test:** Instantiate with DW4 config (n=4, s=2, h=64, L=4), forward pass on random (B, 8) tensor.

---

### Task 2: SteinBiasCorrector — Variance-Loss CV Trainer

**File:** `enhancements/variance_stein_cv.py` (new file)

**What to add:**
- `stein_operator_on_net(g_net, x_flat, score_flat, use_hutchinson, n_probes)` — flat-space Stein operator
- `SteinBiasCorrector` class with `fit()` and `estimate()` methods

**Key implementation details from KSH:**
```python
def stein_operator_on_net(g_net, x_flat, score_flat, use_hutchinson=False, n_probes=1):
    # x = x_flat.detach().requires_grad_(True)
    # g = g_net(x)
    # score_dot_g = (score_flat * g).sum(-1)
    # if hutchinson: div_g via v^T J_g v with n_probes random vectors
    # else: exact divergence via d backward passes
    # return score_dot_g + div_g  → (B,)

class SteinBiasCorrector:
    def __init__(self, g_net, use_hutchinson=False, n_probes=4):

    def fit(self, samples, grad_E, f_vals,
            lr=1e-3, n_iters=5000, batch_size=512, verbose=True,
            cosine_lr=True, weight_decay=0.0, val_fraction=0.2,
            patience=6, bias_penalty=0.0, val_sampler=None):
        # Loss: h.var() + bias_penalty * (Tg.mean()²)
        # where h = f + T_ν g
        # Optimizer: Adam with optional cosine annealing
        # Grad clip: max_norm=5.0
        # Validation: 3 modes
        #   1. val_sampler (fresh samples each eval)
        #   2. val_fraction > 0 (fixed split)
        #   3. val_fraction == 0 (train variance only)
        # Early stopping: patience-based on check_var
        # Eval every 500 steps
        # Best model: deepcopy state_dict

    def estimate(self, samples, grad_E, f_vals):
        # Returns (h.mean().item(), Tg.detach())
```

**Differences from SML's neural_stein_cv.py:**
| Aspect | SML (PDE Loss) | KSH (Variance Loss) |
|--------|---------------|---------------------|
| Loss | ‖∇_x h‖² | Var[h] |
| Needs ∇f | Yes | No |
| Validation | None | train/val split + early stopping |
| Grad clip | 10.0 | 5.0 |
| g-network | MLP (NeuralSteinCV) | Any (passed in) |

**Test:** Train on DW4 samples with SteinEGNN_LN, compare variance reduction to SML's PDE-loss method.

---

### Task 3: ScoreInformedSteinCV — Score-Decomposition CV

**File:** `enhancements/variance_stein_cv.py` (same file as Task 2)

**What to add:**
- `ScoreInformedSteinCV` class with `fit()` and `eval_all()` methods

**Key implementation details from KSH:**
```python
class ScoreInformedSteinCV:
    def __init__(self, g_residual_net, score_model, use_hutchinson=False, n_probes=4):
        # g_res: trainable residual network
        # score_model: pre-trained ISM model (frozen)
        # log_alpha: learnable scalar (log-parameterized), initialized to log(0.01)

    def _precompute_Tg_init(self, samples, energy, score_net, n_particles, spatial_dim, batch_size=256):
        # g_init(x) = -(∇E(x) + s_ϕ(x))
        # Compute T_ν g_init for ALL samples (batched)
        # ∇E via autograd (create_graph=True for divergence)
        # s_ϕ via ScoreEGNN in particle format
        # Divergence: exact or Hutchinson
        # Returns (N,) tensor of T_ν g_init values — FROZEN during training

    def fit(self, samples, energy, f_vals, lr=1e-3, n_iters=5000, batch_size=512):
        # 1. Precompute T_ν g_init (frozen)
        # 2. Initialize log_alpha = log(0.01)
        # 3. Optimizer: Adam over [g_res.parameters(), log_alpha]
        # 4. For each step:
        #    - Tg_res = stein_operator_on_net(g_res, x, score)
        #    - α = exp(log_alpha)
        #    - h = f + α * Tg_init + Tg_res
        #    - running_mean = 0.99 * running_mean + 0.01 * h.mean()
        #    - loss = ((h - running_mean)²).mean()     ← NOTE: not h.var(), uses running mean!
        #    - Grad clip: 5.0 on [g_res.parameters(), log_alpha]

    def eval_all(self, samples, grad_E, f_vals, Tg_init):
        # Returns f + α * Tg_init + Tg_res for all samples (batched eval)
```

**Mathematical decomposition:**
```
g(x) = α · g_init(x) + g_res(x)
g_init(x) = -(∇E(x) + s_ϕ(x))

When q_θ ≈ π: s_ϕ ≈ ∇log q_θ ≈ -∇E → g_init ≈ 0
So T_ν g_init ≈ 0, and the residual g_res handles the correction.

h(x) = f(x) + α · T_ν g_init(x) + T_ν g_res(x)
```

**Dependencies:** Requires Task 2 (stein_operator_on_net) and Task 4 (ScoreEGNN).

**Test:** Train on LJ13 with pre-trained ISM, compare bias/MSE reduction to basic Stein CV.

---

### Task 4: Implicit Score Matching (ISM)

**File:** `enhancements/score_matching.py` (new file)

**What to add:**
- `_build_edges(n_particles, n_systems, device)` — fully connected graph construction
- `ScoreEGNN` class — EGNN-based score model
- `ImplicitScoreModel` class — ISM trainer + diagnostics

**Key implementation details from KSH:**
```python
class ScoreEGNN(nn.Module):
    def __init__(self, n_particles=4, spatial_dim=2, hidden_nf=128, n_layers=4,
                 coord_init_gain=0.1):
        # Wraps EGNN_dynamics internally
        # condition_time=True, t=1.0 always (terminal time)
        # _reinit_coord_weights: xavier_uniform_ with small gain on coord MLP last layers

    def forward(self, x: Tensor, batch: Tensor, edge_index: Tensor) -> Tensor:
        # x: (n_atoms, spatial_dim) → flat → EGNN_dynamics(t=1.0, x_flat) → (n_atoms, spatial_dim)

class ImplicitScoreModel:
    def __init__(self, s_net, dim=None, use_hutchinson=False,
                 n_probes=8, grad_clip=5.0, device="cuda"):

    def _ism_loss_flat(self, x_flat_batch):
        # ISM loss: E[½||s_ϕ(x)||² + div(s_ϕ)(x)]
        # x_req = x.detach().requires_grad_(True)
        # s_flat = s_net.model(t=1.0, x_req)  — NOTE: calls .model directly for flat API
        # sq_norm = 0.5 * s_flat.pow(2).sum(-1)
        # Divergence: exact or Hutchinson (n_probes=8 default for ISM)
        # return (sq_norm + div_est).mean()

    def fit(self, samples, lr=3e-4, n_iters=5000, batch_size=256, verbose=True):
        # Adam optimizer
        # Grad clip: 5.0
        # Log every 500 steps
        # Sets s_net.eval() at end

    def diagnose(self, samples, grad_E, batch_size=256):
        # Computes cosine_similarity(s_ϕ, -∇E) — measures how close q_θ is to π
        # Computes ||s_ϕ + ∇E|| — ideal is 0 when q_θ = π
```

**Hyperparameters:**
| Param | Value | Notes |
|-------|-------|-------|
| LR | 3e-4 | Lower than CV training (1e-3) |
| Iters | 5000 | |
| Batch size | 256 | |
| Grad clip | 5.0 | |
| Hutchinson probes | 8 | More probes than CV (4) for score accuracy |
| Coord init gain | 0.1 | Small init for stability |

**Test:** Train on LJ13 samples, diagnose cosine_sim > 0.9 with -∇E.

---

### Task 5: Observables Module

**File:** `enhancements/observables.py` (new file)

**What to add:**
```python
def mean_energy_observable(x: Tensor, energy_fn) -> Tensor:
    """x: (B, n*d) flat → energy (B,)"""

def mean_interatomic_distance(x: Tensor, n_particles: int, spatial_dim: int) -> Tensor:
    """x: (B, n*d) flat → mean interatomic distance per sample (B,)"""

def interatomic_dist_histogram(x: Tensor, n_particles: int, spatial_dim: int,
                                bins: int = 100, range_min: float = 0.0,
                                range_max: float = 5.0) -> Tensor:
    """Returns normalized histogram (bins,)"""

def observable_gradient(x: Tensor, observable_fn) -> Tensor:
    """∇f(x) w.r.t. flat x (B, dim)"""
```

**Note:** KSH uses `(B, n_particles, spatial_dim)` shaped input; SML uses `(B, dim)` flat. We should support flat input (consistent with SML convention) and reshape internally.

**Test:** Compute on DW4 reference samples, verify values match KSH output.

---

### Task 6: Experiment Scripts

**File:** `experiments/dw4_ksh_steincv.py`, `experiments/lj13_ksh_steincv.py`

**What to add:**
Replicate KSH's evaluation protocol using the new SML-hosted modules.

**DW4 script:**
```
1. Load ASBS checkpoint
2. For each seed in [0, 1, 2]:
   a. Generate fresh ASBS samples (5000)
   b. Compute grad_E, energy, interatomic_dist observables
   c. For each observable:
      - Train SteinEGNN_LN + SteinBiasCorrector (10000 iters, batch=2500)
      - Evaluate: naive vs CV (bootstrap 2000 trials)
      - Report: mean, bias, var, MSE, reduction ratios
3. Summary: mean ± std over seeds
```

**LJ13 script:**
```
1. Load ASBS checkpoint
2. Generate 2000 samples
3. Phase 1: Train ISM ScoreEGNN (5000 iters, lr=3e-4, hutchinson=8 probes)
4. For each observable (energy, interatomic_dist):
   a. Phase 2a: Train SteinBiasCorrector (5000 iters, hutchinson=4 probes)
   b. Phase 2b: Train ScoreInformedSteinCV (5000 iters, hutchinson=4 probes)
   c. Evaluate: 200 trials, subsample_size=200
   d. Report: naive vs basic_cv vs score_cv
5. Save results JSON
```

---

## Execution Order

```
Task 1 (SteinEGNN_LN)          ← standalone, no dependencies
Task 4 (Score Matching)         ← standalone, no dependencies
Task 5 (Observables)            ← standalone, no dependencies
Task 2 (SteinBiasCorrector)     ← depends on Task 1 (uses SteinEGNN_LN)
Task 3 (ScoreInformedSteinCV)   ← depends on Tasks 2 + 4
Task 6 (Experiment scripts)     ← depends on all above
```

**Parallelizable:** Tasks 1, 4, 5 can be done simultaneously.

---

## File Summary

| New/Modified File | What Goes In |
|-------------------|-------------|
| `enhancements/egnn_stein_cv.py` (modify) | Add `E_GCL_LN`, `SteinEGNN_LN` |
| `enhancements/variance_stein_cv.py` (new) | `stein_operator_on_net`, `SteinBiasCorrector`, `ScoreInformedSteinCV` |
| `enhancements/score_matching.py` (new) | `ScoreEGNN`, `ImplicitScoreModel`, `_build_edges` |
| `enhancements/observables.py` (new) | Observable functions (energy, dist, histogram, gradient) |
| `experiments/dw4_ksh_steincv.py` (new) | DW4 evaluation (3 seeds, 2 observables) |
| `experiments/lj13_ksh_steincv.py` (new) | LJ13 evaluation (ISM + basic + score-informed) |

---

## Gotchas & Porting Notes

1. **Import paths differ:** KSH uses `from adjoint_samplers.components.model import EGNN_dynamics` — same in SML, no change needed.

2. **`remove_mean` import:** KSH imports from `adjoint_samplers.utils.graph_utils` — same path in SML.

3. **`_build_edges` is duplicated** in KSH between `neural_cv.py` and `score_matching.py`. In SML, define it once (in `score_matching.py` or a shared util) and import.

4. **KSH's `stein_operator_on_net`** detaches x and requires_grad again — this is important for correct autograd behavior. SML's `compute_stein_operator` in `neural_stein_cv.py` does similar but takes `g_values` as pre-computed. The KSH version is cleaner (takes `g_net` directly).

5. **`SteinBiasCorrector.fit()` logs every 500 steps** and does full-dataset eval for validation. This can be slow for large N. Consider making eval frequency configurable.

6. **`ScoreInformedSteinCV._precompute_Tg_init`** uses `energy.eval()` + autograd for ∇E, then score_net in particle format. The score_net forward pass needs `(x_parts, batch_idx, edge_index)` — not flat API. This is a potential API mismatch; ensure ScoreEGNN's forward signature is preserved.

7. **Running mean in ScoreInformedSteinCV:** The loss is `((h - running_mean)²).mean()`, NOT `h.var()`. The running mean provides a more stable centering signal than per-batch mean. This is a subtle but important difference.

8. **KSH evaluates via bootstrap subsampling** (200 trials, subsample_size=200), while SML evaluates on full samples. Both protocols should be available.

9. **KSH's coord_diff normalization:** `coord_diff / (norm + 1)` where `norm = sqrt(radial + 1e-8)`. This is NOT in SML's base EGNN. Only needed in `E_GCL_LN`.

10. **`SteinEGNN_LN` edge_attr is computed once** (before layers) and not updated. SML's base EGNN recomputes edge features per layer. This is a design choice for the LN variant.
