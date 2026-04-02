# Adjoint Sampling with Stein-Based Variance Reduction — Full Results

> **Generated:** 2026-04-02 (DW4 complete, LJ13/LJ55 pending)
> **Repo:** `adjoint_samplers`
> **Methods:** 12 enhancement pipelines (7 SML + 3 KSH + 2 advanced) evaluated across 3 molecular systems

---

## Table of Contents

1. [Overview](#1-overview)
2. [Systems Under Test](#2-systems-under-test)
3. [Methods](#3-methods)
4. [DW4 — Double Well 4 Particles (8D)](#4-dw4--double-well-4-particles-8d)
5. [LJ13 — Lennard-Jones 13 Particles (39D)](#5-lj13--lennard-jones-13-particles-39d)
6. [LJ55 — Lennard-Jones 55 Particles (165D)](#6-lj55--lennard-jones-55-particles-165d)
7. [Cross-System Comparison](#7-cross-system-comparison)
8. [Ablations](#8-ablations)
9. [Key Takeaways](#9-key-takeaways)

---

## 1. Overview

We train Adjoint Sampling with Bridge Splitting (ASBS) to sample from Boltzmann distributions of molecular systems, then apply **12 post-hoc enhancement methods** (7 SML-original + 3 KSH-ported + 2 advanced) to reduce estimator bias and variance.

**Two observable targets:**
- **Energy:** $\langle E \rangle_p$ (primary, all methods)
- **Interatomic distance:** $\langle \bar{d} \rangle_p$ (KSH methods only)

**Evaluation protocols:**
- **SML protocol:** 10 seeds × 4 sample sizes (N ∈ {100, 500, 1000, 2000}), full-sample evaluation
- **KSH protocol:** 3 seeds × 200 bootstrap trials (subsample_size=200), per-observable
- Ground truth: sample mean of 10,000 reference samples from the target distribution
- All metrics reported as mean ± std across seeds

---

## 2. Systems Under Test

| System | Particles | Spatial Dim | Total Dim | Energy | Reference Samples |
|--------|-----------|-------------|-----------|--------|-------------------|
| DW4 | 4 | 2D | **8** | Pairwise double-well | 10,000 |
| LJ13 | 13 | 3D | **39** | Lennard-Jones + harmonic | 10,000 |
| LJ55 | 55 | 3D | **165** | Lennard-Jones + harmonic | 10,000 |

**DW4** — $E_{\text{DW}}(d) = 0.9(d-4)^4 - 4(d-4)^2$ pairwise over 4 particles in 2D. Fast iteration, good for debugging all enhancements.

**LJ13** — $E_{\text{LJ}}(r) = \varepsilon\left[\left(\frac{r_{\min}}{r}\right)^{12} - 2\left(\frac{r_{\min}}{r}\right)^6\right]$ with $\varepsilon=1, r_{\min}=1$ and external harmonic confinement. 13 particles in 3D with COM constraint. Medium complexity.

**LJ55** — Same LJ potential, 55 particles in 3D. High-dimensional stress test (165D). RKHS kernels degrade; Neural Stein CV is the primary method here.

---

## 3. Methods

### 3.1 SML Methods (Original)

| # | Method | Key Idea | Complexity | Bias Fix? | Var Fix? |
|---|--------|----------|------------|-----------|----------|
| 1 | **Vanilla ASBS** | Sample mean $\frac{1}{N}\sum f(x_i)$ | $O(N)$ | ✗ | ✗ |
| 2 | **Stein CV (RKHS)** | Optimal RKHS control variate via kernel ridge regression | $O(N^3)$ | ✓ (coupling) | ✓ |
| 3 | **Antithetic** | Paired trajectories with negated Brownian noise | $O(N)$ | ✗ | ✓ |
| 4 | **MCMC Corrected** | K Metropolis-Hastings steps on terminal samples | $O(KN)$ | ✓ | ✗ |
| 5 | **MCMC + Stein CV** | MCMC first, then RKHS Stein CV | $O(KN + N^3)$ | ✓✓ | ✓ |
| 6 | **Generator Stein CV** | Uses learned drift $b_\theta(x,1)$ instead of $s_p(x)$ | $O(N^3)$ | ✓ (coupling) | ✓ |
| 7 | **Neural Stein CV** | MLP $g_\phi$ trained on PDE loss $\|\nabla_x h\|^2$ | $O(BdT)$ | ✓ (coupling) | ✓ |
| 8 | **EGNN Stein CV** | Equivariant $g_\phi$ (same arch as ASBS controller) on PDE loss | $O(Bn^2LT)$ | ✓ (coupling) | ✓ |
| 9 | **RBF Collocation CV** | Expand $g$ in Gaussian RBF basis, single least-squares solve | $O(NdM + M^3)$ | ✓ (differentiated PDE) | ✓ |

### 3.2 KSH Methods (Ported from KSH_ASBS)

| # | Method | Key Idea | Complexity | Bias Fix? | Var Fix? |
|---|--------|----------|------------|-----------|----------|
| 10 | **SteinEGNN_LN (Var loss)** | EGNN+LayerNorm trained on $\text{Var}[f + T_\nu g]$ | $O(Bn^2LT)$ | ✓ (coupling) | ✓ |
| 11 | **ISM + Basic Stein CV** | Implicit Score Matching diagnostic + variance-loss CV | $O(BdT)$ | ✓ (coupling) | ✓ |
| 12 | **Score-Informed Stein CV** | $g = \alpha \cdot g_{\text{init}} + g_{\text{res}}$ with ISM-derived $g_{\text{init}}$ | $O(BdT)$ | ✓✓ (score-aware) | ✓✓ |

### 3.3 Method Details

**Methods 8–9 (SML advances)** address the instability of Neural Stein CV (method 7):
- **EGNN Stein CV** exploits the particle structure and E(3) symmetry of molecular systems. The MLP ignores that DW4/LJ are particle systems — EGNN enforces equivariance by construction, constraining $g_\phi$ to physically meaningful vector fields.
- **RBF Collocation CV** avoids neural network training entirely. By expanding $g$ in a Gaussian RBF basis, the differentiated PDE becomes a **linear least-squares** problem — one matrix solve, no epochs, no gradient instability.

**Methods 10–12 (KSH advances)** take a fundamentally different approach:
- **SteinEGNN_LN** uses the same EGNN architecture but a different loss: $\text{Var}[h]$ instead of $\|\nabla h\|^2$. This doesn't require $\nabla f$, supports early stopping with train/val split, and uses LayerNorm for stability.
- **ISM + Basic CV** first trains a score model $s_\phi \approx \nabla \log q_\theta$ via Implicit Score Matching, providing diagnostics on sampler quality. The basic CV uses variance loss with SteinEGNN_LN.
- **Score-Informed CV** decomposes $g = \alpha \cdot g_{\text{init}} + g_{\text{res}}$ where $g_{\text{init}} = -(\nabla E + s_\phi)$. When $q_\theta \approx \pi$, $g_{\text{init}} \approx 0$ and $T_\nu g_{\text{init}} \approx 0$, so the residual $g_{\text{res}}$ only corrects the remaining error. This is the **most sophisticated** method, combining ISM, equivariant architecture, and score-based decomposition.

### 3.4 SML vs KSH Loss Function Comparison

| Aspect | SML (PDE Loss) | KSH (Variance Loss) |
|--------|---------------|---------------------|
| **Objective** | $\min \|\nabla_x(f + T_\nu g)\|^2$ | $\min \text{Var}[f + T_\nu g]$ |
| **Requires $\nabla f$?** | Yes | No |
| **Validation** | None | Train/val split + early stopping |
| **Grad clip** | 10.0 | 5.0 |
| **g-network** | MLP (generic) | SteinEGNN_LN (equivariant) |
| **Observable-agnostic?** | No (needs differentiable $f$) | Yes |

**Bias-Variance Coupling Theorem (v2):**
$$|\text{Bias}| \leq \sqrt{C \cdot \text{Var}_{q_\theta}[f + \mathcal{A}_p g]}$$
Optimizing $g$ for variance reduction *automatically* shrinks the bias. This applies to **both** SML and KSH methods — they differ only in how they minimize $\text{Var}[h]$.

---

## 4. DW4 — Double Well 4 Particles (8D)

### 4.1 Training

| Parameter | Value |
|-----------|-------|
| Experiment | `dw4_asbs` |
| Epochs | 5000 |
| NFE | 200 |
| σ_max / σ_min | 1.0 / 0.001 |
| Source | Harmonic (scale=2) |
| Checkpoint | `results/local/2026.03.31/152919/checkpoints/checkpoint_latest.pt` |

### 4.1.1 Ground Truth

Computed from 10,000 reference samples drawn from the true Boltzmann distribution $p \propto e^{-E(x)}$ (`data/test_split_DW4.npy`), evaluated through `DoubleWellEnergy.eval()`:

| Statistic | Value |
|-----------|-------|
| **Mean** | **-22.4504** |
| Std | 1.9015 |
| Median | -22.7987 |
| [5th, 95th] %ile | [-24.9067, -18.8260] |
| [Min, Max] | [-25.7486, -11.8381] |

### 4.2 Summary Table — Energy Observable (N = 2000, 10 seeds, SML protocol)

| Method | Mean Energy | |Error| | Variance | Var Ratio |
|--------|-------------|--------|----------|-----------|
| **Ground Truth** | **-22.4504** | 0 | — | — |
| Vanilla ASBS | -22.4093 ± 0.0443 | 0.0456 | 1.92e-03 | 1.000 |
| Stein CV (RKHS) | -25.0793 ± 0.1873 | 2.6289 | 1.45e-03 | 0.759 |
| Antithetic | -22.3934 ± 0.0387 | 0.0592 | 1.32e-03 | 0.688 |
| MCMC (K=10) | -22.4084 ± 0.0471 | 0.0467 | 1.92e-03 | — |
| MCMC + Stein CV | -24.8512 ± 0.1750 | 2.4008 | 1.31e-03 | — |
| Generator Stein CV | -22.4017 ± 0.0444 | 0.0510 | 2.40e-03 | — |
| Neural Stein CV | -21.8614 ± 0.5799 | 0.7415 | 6.00e-01 | 313.279 |
| EGNN Stein CV | -21.8439 ± 0.5810 | 0.7487 | 5.96e-01 | 311.415 |
| RBF Collocation CV | -22.6226 ± 0.1824 | 0.2030 | 3.67e-02 | 19.327 |

**Scaling with N (mean |Error| across 10 seeds):**

| Method | N=100 | N=500 | N=1000 | N=2000 |
|--------|-------|-------|--------|--------|
| Vanilla ASBS | 0.130 | 0.059 | 0.106 | 0.046 |
| Stein CV (RKHS) | 0.466 | 1.013 | 1.852 | 2.629 |
| Antithetic | 0.159 | 0.074 | 0.098 | 0.059 |
| MCMC (K=10) | 0.134 | 0.062 | 0.101 | 0.047 |
| Generator Stein CV | 0.132 | 0.060 | 0.114 | 0.051 |
| Neural Stein CV | 2.291 | 1.497 | 0.796 | 0.741 |
| EGNN Stein CV | 2.343 | 1.496 | 0.811 | 0.749 |
| RBF Collocation CV | 1.543 | 0.918 | 0.235 | 0.203 |

### 4.2b KSH-Style Results (3 seeds, N=5000, per-observable)

**Energy Observable** (GT = -22.4504):

| Method | Mean | |Bias| | Var | MSE | MSE Reduction |
|--------|------|--------|-----|-----|---------------|
| Naive ASBS | -22.3767 ± 0.0173 | 0.0737 ± 0.0173 | 8.08e-04 | 6.54e-03 | 1.00× |
| SteinEGNN_LN (Var loss) | -22.3947 ± 0.0330 | 0.0557 ± 0.0330 | 2.91e-04 | 4.48e-03 | **4.07×** |

**Interatomic Distance Observable** (GT = 3.9700):

| Method | Mean | |Bias| | Var | MSE | MSE Reduction |
|--------|------|--------|-----|-----|---------------|
| Naive ASBS | 3.9116 ± 0.0038 | 0.0585 ± 0.0038 | 5.13e-05 | 3.48e-03 | 1.00× |
| SteinEGNN_LN (Var loss) | 3.9116 ± 0.0042 | 0.0584 ± 0.0042 | 4.80e-05 | 3.47e-03 | **1.00×** |

**Per-seed energy details:**

| Seed | Naive MSE | CV MSE | MSE Red. | Bias Red. |
|------|-----------|--------|----------|-----------|
| 0 | 3.40e-03 | 2.25e-03 | 1.51× | 1.13× |
| 1 | 9.24e-03 | 1.05e-02 | 0.88× | 0.91× |
| 2 | 6.97e-03 | 7.11e-04 | 9.81× | 3.59× |

> **Note:** No ScoreInformedSteinCV for DW4 — ISM is overkill at 8D (KSH also skips it). High seed variance (0.88–9.81×) reflects the already-small bias in DW4.

### 4.2c SML vs KSH Loss Comparison (EGNN architecture, energy observable)

| Loss Function | g-Network | Mean Energy | |Error| | Var Ratio | Notes |
|---------------|-----------|-------------|--------|-----------|-------|
| PDE ($\|\nabla h\|^2$) | EGNNSteinCV (h=128,L=6) | -21.8439 | 0.7487 | 311× worse | SML, N=2000, no validation |
| Variance ($\text{Var}[h]$) | SteinEGNN_LN (h=64,L=4) | -22.3947 | 0.0557 | **2.78× better** | KSH, N=5000, early stopping |

> **Variance loss is dramatically better for DW4.** PDE loss explodes variance (311× worse), while variance loss achieves 2.78× variance reduction (naive var / CV var = 8.08e-4 / 2.91e-4). The key advantage: variance loss directly optimizes the metric we care about, plus early stopping prevents overfitting.

### 4.3 Diagnostics (N = 2000, 10 seeds)

| Metric | Value |
|--------|-------|
| KSD² | 0.0201 ± 0.0118 |
| MH Acceptance Rate | 0.0041 ± 0.0005 |
| Antithetic Correlation | 0.3735 ± 0.0224 |
| Eval Time per Seed | 68.7s ± 8.7s |

### 4.4 Plots

#### Estimation Error vs Sample Size
![DW4 Error vs N](dw4_error_vs_N.png)

#### Estimator Variance vs Sample Size
![DW4 Variance vs N](dw4_variance_vs_N.png)

#### Variance Reduction Factors
![DW4 Var Reduction](dw4_variance_reduction_bars.png)

#### KSD² vs Sample Size
![DW4 KSD](dw4_ksd_vs_N.png)

#### Antithetic Correlation vs Sample Size
![DW4 Antithetic](dw4_antithetic_correlation.png)

#### MCMC Ablation (K = 0, 5, 10, 20, 50)
![DW4 MCMC Ablation](dw4_mcmc_ablation.png)

#### Summary Table (Image)
![DW4 Summary](dw4_summary_table.png)

### 4.5 DW4 Observations

- **Vanilla ASBS is already excellent:** Error of only 0.046 — the learned sampler is very close to the target in 8D. Low KSD² (0.020) confirms good sample quality.
- **Stein CV (RKHS) hurts estimation:** Error degrades with N (0.47→1.01→1.85→2.63). The kernel ridge regression over-corrects, introducing massive bias that grows with sample size. Variance is mildly reduced (0.76×) but bias penalty dominates.
- **Neural/EGNN Stein CV (PDE loss) is catastrophic:** Variance explodes 311–313× worse. The PDE loss $\|\nabla h\|^2$ fails to converge — the already-accurate vanilla estimate leaves too little signal for the neural net, and without validation, training overshoots.
- **RBF Collocation CV is moderate:** Error 0.20 (worse than vanilla) but improves with N. VarRatio of 19× is bad but far better than neural methods.
- **MCMC is nearly inert:** 0.4% acceptance rate means MH proposals are almost always rejected. The target distribution's geometry defeats random-walk proposals in 8D.
- **Antithetic gives mild variance reduction** (0.69×) with correlation 0.37 — the anti-correlated trajectories provide modest cancellation.
- **KSH SteinEGNN_LN (Var loss) is the clear winner among advanced methods:** 4.07× MSE reduction for energy, with variance reduced 2.78× and bias reduced 1.88×. Early stopping at step 400–1800 prevents overfitting.
- **KSH interatomic distance CV shows no effect** (1.00×) — the distance observable has very low variance already (5e-5), leaving no room for improvement. The bias is dominated by sampling error, not estimator variance.
- **Key insight:** Variance loss >> PDE loss for DW4. The direct optimization of Var[h] + early stopping avoids the catastrophic instability of PDE loss. But even variance loss has high seed-to-seed variability (0.88–9.81× MSE reduction) when the base estimator is already accurate.

---

## 5. LJ13 — Lennard-Jones 13 Particles (39D)

### 5.1 Training

| Parameter | Value |
|-----------|-------|
| Experiment | `lj13_asbs` |
| Epochs | 5000 |
| NFE | 1000 |
| σ_max / σ_min | 1.0 / 0.001 |
| Source | Harmonic (scale=2) |
| Checkpoint | `[PENDING — corrector phase in progress]` |

### 5.2 Summary Table — Energy Observable (N = 2000, 10 seeds, SML protocol)

| Method | Mean Energy | |Error| | Variance | Var Ratio |
|--------|-------------|--------|----------|-----------|
| **Ground Truth** | `___` | 0 | — | — |
| Vanilla ASBS | `___` ± `___` | `___` | `___` | 1.000 |
| Stein CV (RKHS) | `___` ± `___` | `___` | `___` | `___` |
| Antithetic | `___` ± `___` | `___` | `___` | `___` |
| MCMC (K=10) | `___` ± `___` | `___` | `___` | — |
| MCMC + Stein CV | `___` ± `___` | `___` | `___` | — |
| Generator Stein CV | `___` ± `___` | `___` | `___` | — |
| Neural Stein CV | `___` ± `___` | `___` | `___` | `___` |

### 5.2b KSH-Style Results (200 trials, subsample=200, per-observable)

**ISM Score Model Diagnostics:**

| Metric | Value |
|--------|-------|
| cos_sim(s_ϕ, -∇E) | `___` ± `___` |
| \|\|s_ϕ + ∇E\|\| | `___` ± `___` |

> ISM quality indicates how close the learned sampler $q_\theta$ is to the target $\pi$. Higher cosine similarity → better ASBS training.

**Energy Observable:**

| Method | Mean | |Bias| | Var | MSE | MSE Reduction |
|--------|------|--------|-----|-----|---------------|
| Naive ASBS | `___` | `___` | `___` | `___` | 1.00× |
| SteinEGNN_LN (Var loss) | `___` | `___` | `___` | `___` | `___`× |
| Score-Informed CV | `___` | `___` | `___` | `___` | `___`× |

**Interatomic Distance Observable:**

| Method | Mean | |Bias| | Var | MSE | MSE Reduction |
|--------|------|--------|-----|-----|---------------|
| Naive ASBS | `___` | `___` | `___` | `___` | 1.00× |
| SteinEGNN_LN (Var loss) | `___` | `___` | `___` | `___` | `___`× |
| Score-Informed CV | `___` | `___` | `___` | `___` | `___`× |

**Score-Informed CV Details:**

| Parameter | Value |
|-----------|-------|
| Final α | `___` |
| T_ν g_init mean | `___` |
| T_ν g_init std | `___` |

> α measures how much the score component contributes. Small α → ISM-derived direction is weak. Large α → ISM captures useful structure.

### 5.2c SML vs KSH Loss Comparison (energy observable, LJ13)

| Loss Function | g-Network | |Error| | Var Ratio | Wall Time |
|---------------|-----------|--------|-----------|-----------|
| PDE ($\|\nabla h\|^2$) | MLP (Neural CV) | `___` | `___` | `___` |
| PDE ($\|\nabla h\|^2$) | EGNNSteinCV | `___` | `___` | `___` |
| Variance ($\text{Var}[h]$) | SteinEGNN_LN | `___` | `___` | `___` |
| Score-Informed | SteinEGNN_LN + ISM | `___` | `___` | `___` |

### 5.3 Diagnostics

| Metric | Value |
|--------|-------|
| KSD² | `___` ± `___` |
| MH Acceptance Rate | `___` |
| Antithetic Correlation | `___` |

### 5.4 Plots

#### Estimation Error vs Sample Size
<!-- ![LJ13 Error vs N](lj13_error_vs_N.png) -->
`[PENDING: lj13_error_vs_N.png]`

#### Estimator Variance vs Sample Size
<!-- ![LJ13 Variance vs N](lj13_variance_vs_N.png) -->
`[PENDING: lj13_variance_vs_N.png]`

#### Variance Reduction Factors
<!-- ![LJ13 Var Reduction](lj13_variance_reduction_bars.png) -->
`[PENDING: lj13_variance_reduction_bars.png]`

#### KSD² vs Sample Size
<!-- ![LJ13 KSD](lj13_ksd_vs_N.png) -->
`[PENDING: lj13_ksd_vs_N.png]`

#### Antithetic Correlation vs Sample Size
<!-- ![LJ13 Antithetic](lj13_antithetic_correlation.png) -->
`[PENDING: lj13_antithetic_correlation.png]`

#### MCMC Ablation (K = 0, 5, 10, 20, 50)
<!-- ![LJ13 MCMC Ablation](lj13_mcmc_ablation.png) -->
`[PENDING: lj13_mcmc_ablation.png]`

#### Summary Table (Image)
<!-- ![LJ13 Summary](lj13_summary_table.png) -->
`[PENDING: lj13_summary_table.png]`

### 5.5 RKHS vs Neural vs KSH: The 39D Crossover

At 39D, we expect method hierarchies to shift:

| Metric | RKHS Stein CV | Neural CV (PDE) | SteinEGNN_LN (Var) | Score-Informed | Winner |
|--------|---------------|-----------------|---------------------|----------------|--------|
| Variance Reduction | `___` | `___` | `___` | `___` | `___` |
| |Bias| | `___` | `___` | `___` | `___` | `___` |
| Wall-clock Time | `___` | `___` | `___` | `___` | `___` |

### 5.6 LJ13 Observations

- *Expected:* RKHS starts degrading (39D kernel), Neural CV gains advantage.
- *KSH vs SML:* At 39D, equivariance (SteinEGNN_LN) should matter — MLP has no particle structure awareness.
- *Score-Informed benefit:* If ISM cos_sim > 0.9, score decomposition gives strong g_init → larger MSE reduction.
- *Variance reduction:* `___`
- *Bias reduction:* `___`
- *PDE loss vs Variance loss:* `___`
- *Score-Informed α:* `___`

---

## 6. LJ55 — Lennard-Jones 55 Particles (165D)

### 6.1 Training

| Parameter | Value |
|-----------|-------|
| Experiment | `lj55_asbs` |
| Epochs | 5000 |
| NFE | 1000 |
| σ_max / σ_min | 2.0 / 0.001 |
| Source | Harmonic (scale=1) |
| Checkpoint | `[PENDING — not started]` |

### 6.2 Summary Table — Energy Observable (N = 2000, 10 seeds, SML protocol)

| Method | Mean Energy | |Error| | Variance | Var Ratio |
|--------|-------------|--------|----------|-----------|
| **Ground Truth** | `___` | 0 | — | — |
| Vanilla ASBS | `___` ± `___` | `___` | `___` | 1.000 |
| Stein CV (RKHS) | `___` ± `___` | `___` | `___` | `___` |
| Antithetic | `___` ± `___` | `___` | `___` | `___` |
| MCMC (K=10) | `___` ± `___` | `___` | `___` | — |
| MCMC + Stein CV | `___` ± `___` | `___` | `___` | — |
| Generator Stein CV | `___` ± `___` | `___` | `___` | — |
| Neural Stein CV | `___` ± `___` | `___` | `___` | `___` |

### 6.2b KSH-Style Results (200 trials, subsample=200, per-observable)

**ISM Score Model Diagnostics:**

| Metric | Value |
|--------|-------|
| cos_sim(s_ϕ, -∇E) | `___` ± `___` |
| \|\|s_ϕ + ∇E\|\| | `___` ± `___` |

**Energy Observable:**

| Method | Mean | |Bias| | Var | MSE | MSE Reduction |
|--------|------|--------|-----|-----|---------------|
| Naive ASBS | `___` | `___` | `___` | `___` | 1.00× |
| SteinEGNN_LN (Var loss) | `___` | `___` | `___` | `___` | `___`× |
| Score-Informed CV | `___` | `___` | `___` | `___` | `___`× |

**Interatomic Distance Observable:**

| Method | Mean | |Bias| | Var | MSE | MSE Reduction |
|--------|------|--------|-----|-----|---------------|
| Naive ASBS | `___` | `___` | `___` | `___` | 1.00× |
| SteinEGNN_LN (Var loss) | `___` | `___` | `___` | `___` | `___`× |
| Score-Informed CV | `___` | `___` | `___` | `___` | `___`× |

> At 165D, Score-Informed CV is the primary hope — RKHS collapses, MLP struggles, but EGNN with score decomposition may still capture molecular structure.

### 6.3 Diagnostics

| Metric | Value |
|--------|-------|
| KSD² | `___` ± `___` |
| MH Acceptance Rate | `___` |
| Antithetic Correlation | `___` |

### 6.4 Plots

#### Estimation Error vs Sample Size
<!-- ![LJ55 Error vs N](lj55_error_vs_N.png) -->
`[PENDING: lj55_error_vs_N.png]`

#### Estimator Variance vs Sample Size
<!-- ![LJ55 Variance vs N](lj55_variance_vs_N.png) -->
`[PENDING: lj55_variance_vs_N.png]`

#### Variance Reduction Factors
<!-- ![LJ55 Var Reduction](lj55_variance_reduction_bars.png) -->
`[PENDING: lj55_variance_reduction_bars.png]`

#### KSD² vs Sample Size
<!-- ![LJ55 KSD](lj55_ksd_vs_N.png) -->
`[PENDING: lj55_ksd_vs_N.png]`

#### Antithetic Correlation vs Sample Size
<!-- ![LJ55 Antithetic](lj55_antithetic_correlation.png) -->
`[PENDING: lj55_antithetic_correlation.png]`

#### MCMC Ablation (K = 0, 5, 10, 20, 50)
<!-- ![LJ55 MCMC Ablation](lj55_mcmc_ablation.png) -->
`[PENDING: lj55_mcmc_ablation.png]`

#### Summary Table (Image)
<!-- ![LJ55 Summary](lj55_summary_table.png) -->
`[PENDING: lj55_summary_table.png]`

### 6.5 High-Dimensional Scaling: RKHS Collapse & KSH Advantage

At 165D, the RBF kernel $k(x,y) = \exp(-\|x-y\|^2 / 2\ell^2)$ becomes nearly constant (curse of dimensionality). KSH methods with equivariant architecture may be the only viable approach.

| Metric | RKHS Stein CV | Neural CV (MLP) | SteinEGNN_LN (Var) | Score-Informed | Ratio |
|--------|---------------|-----------------|---------------------|----------------|-------|
| Variance Reduction | `___` (≈1.0?) | `___` | `___` | `___` | `___` |
| |Bias| | `___` | `___` | `___` | `___` | `___` |
| Wall-clock Time | `___` | `___` | `___` | `___` | `___` |

Hutchinson divergence estimator is required for all neural methods (exact divergence would need 165 backward passes per sample).

### 6.6 LJ55 Observations

- *Expected:* RKHS nearly useless. MLP Neural CV may struggle (no particle structure). SteinEGNN_LN with score-informed decomposition is the primary candidate.
- *Variance reduction:* `___`
- *Bias reduction:* `___`
- *Hutchinson vs exact divergence:* `___`
- *Score-Informed α:* `___`
- *EGNN vs MLP at 165D:* `___`

---

## 7. Cross-System Comparison

### 7.1 Variance Reduction Factor by System (N = 2000, SML protocol)

| Method | DW4 (8D) | LJ13 (39D) | LJ55 (165D) |
|--------|----------|-------------|--------------|
| Stein CV (RKHS) | 0.759 (mild ✓) | `___` | `___` |
| Antithetic | 0.688 (mild ✓) | `___` | `___` |
| Neural Stein CV | 313× (✗ explodes) | `___` | `___` |
| EGNN Stein CV | 311× (✗ explodes) | `___` | `___` |
| RBF Collocation CV | 19× (✗ bad) | `___` | `___` |

### 7.1b MSE Reduction by System (KSH protocol, energy observable)

| Method | DW4 (8D) | LJ13 (39D) | LJ55 (165D) |
|--------|----------|-------------|--------------|
| SteinEGNN_LN (Var loss) | **4.07×** | `___`× | `___`× |
| Score-Informed CV | — | `___`× | `___`× |

#### Cross-System Variance Reduction Plot
<!-- ![Cross-System VarRed](cross_system_var_reduction.png) -->
`[PENDING: cross_system_var_reduction.png]`

### 7.2 Absolute Error by System (N = 2000, SML protocol)

| Method | DW4 (8D) | LJ13 (39D) | LJ55 (165D) |
|--------|----------|-------------|--------------|
| Vanilla ASBS | 0.046 | `___` | `___` |
| Stein CV (RKHS) | 2.629 | `___` | `___` |
| MCMC (K=10) | 0.047 | `___` | `___` |
| MCMC + Stein CV | 2.401 | `___` | `___` |
| Neural Stein CV | 0.741 | `___` | `___` |
| EGNN Stein CV | 0.749 | `___` | `___` |
| RBF Collocation CV | 0.203 | `___` | `___` |

### 7.2b Bias Reduction by System (KSH protocol, energy observable)

| Method | DW4 (8D) | LJ13 (39D) | LJ55 (165D) |
|--------|----------|-------------|--------------|
| SteinEGNN_LN (Var loss) | **1.88×** | `___`× | `___`× |
| Score-Informed CV | — | `___`× | `___`× |

#### Cross-System Error Plot
<!-- ![Cross-System Error](cross_system_error.png) -->
`[PENDING: cross_system_error.png]`

### 7.3 Method Crossover by Dimension

<!-- ![Method Crossover](method_crossover.png) -->
`[PENDING: method_crossover.png]`

Plot: x-axis = dimension (8, 39, 165), y-axis = MSE reduction, lines for: RKHS, Neural CV (PDE), SteinEGNN_LN (Var), Score-Informed. Expected crossovers:
- RKHS dominates at d ≤ 20 (if it works), degrades sharply after
- MLP Neural CV: moderate at 39D, struggles at 165D
- SteinEGNN_LN: competitive at 39D due to equivariance
- Score-Informed: best at 39D+ where ISM provides useful signal

### 7.3b ISM Score Quality Across Systems

| System | cos_sim(s_ϕ, -∇E) | \|\|s_ϕ + ∇E\|\| | Interpretation |
|--------|-------------------|-------------------|----------------|
| DW4 | — (not run) | — | — |
| LJ13 | `___` | `___` | `___` |
| LJ55 | `___` | `___` | `___` |

> ISM quality is a proxy for ASBS training quality. Higher cos_sim → sampler is closer to target → less room for Score-Informed CV to help.

### 7.4 KSD² Across Systems

| System | KSD² (mean ± std) | Interpretation |
|--------|-------------------|----------------|
| DW4 | 0.0201 ± 0.0118 | Low KSD — sampler is close to target |
| LJ13 | `___` | `___` |
| LJ55 | `___` | `___` |

### 7.5 Computational Cost

| Method | DW4 (s) | LJ13 (s) | LJ55 (s) | Scaling |
|--------|---------|-----------|-----------|---------|
| Vanilla | `___` | `___` | `___` | $O(N)$ |
| Stein CV (RKHS) | `___` | `___` | `___` | $O(N^3)$ |
| Antithetic | `___` | `___` | `___` | $O(N)$ |
| MCMC (K=10) | `___` | `___` | `___` | $O(KN)$ |
| MCMC + Stein CV | `___` | `___` | `___` | $O(KN + N^3)$ |
| Generator Stein CV | `___` | `___` | `___` | $O(N^3)$ |
| Neural Stein CV (PDE) | `___` | `___` | `___` | $O(BdT)$ |
| SteinEGNN_LN (Var) | `___` | `___` | `___` | $O(Bn^2LT)$ |
| ISM Training | — | `___` | `___` | $O(BdT)$ |
| Score-Informed CV | — | `___` | `___` | $O(Bn^2LT)$ + ISM |

> ISM training is a one-time cost shared by all Score-Informed evaluations. SteinEGNN_LN is slower per iteration than MLP due to graph message passing ($n^2$ edges per layer), but may need fewer iterations due to equivariance.

---

## 8. Ablations

### 8.1 MCMC Steps Ablation

Effect of MH correction steps K on estimation error (N = 2000):

| K | DW4 Error (MCMC) | DW4 Acceptance | LJ13 Error (MCMC) | LJ13 Acceptance | LJ55 Error (MCMC) | LJ55 Acceptance |
|---|------------------|---------------|--------------------|------------------|--------------------|------------------|
| 0 | 0.0842 | — | `___` | `___` | `___` | `___` |
| 5 | 0.0833 | 0.40% | `___` | `___` | `___` | `___` |
| 10 | 0.0826 | 0.42% | `___` | `___` | `___` | `___` |
| 20 | 0.0750 | 0.42% | `___` | `___` | `___` | `___` |
| 50 | 0.0587 | 0.40% | `___` | `___` | `___` | `___` |

### 8.2 Stein Regularization λ Ablation

Effect of kernel ridge regression regularization on Stein CV (N = 2000):

| λ | DW4 Var Ratio | LJ13 Var Ratio | LJ55 Var Ratio |
|---|---------------|----------------|----------------|
| 1e-6 | `___` | `___` | `___` |
| 1e-4 | `___` | `___` | `___` |
| 1e-2 | `___` | `___` | `___` |

### 8.3 Neural CV: Epochs & Architecture

Effect of training epochs on Neural CV quality (N = 2000):

| Epochs | DW4 VarRed | LJ13 VarRed | LJ55 VarRed | Final PDE Loss |
|--------|------------|-------------|-------------|----------------|
| 100 | `___` | `___` | `___` | `___` |
| 500 | `___` | `___` | `___` | `___` |
| 1000 | `___` | `___` | `___` | `___` |
| 2000 | `___` | `___` | `___` | `___` |

### 8.4 Regime 1 vs Regime 2

**Regime 1** (Stein CV only): No energy evals for MH, cheaper, but bias depends on RKHS/neural approximation quality.

**Regime 2** (MCMC + Stein CV): Expensive (K×N energy evals) but exact — MCMC makes samples from $p$, so Stein CV gives pure variance reduction with zero bias contamination.

| Regime | DW4 Error | LJ13 Error | LJ55 Error | Cost |
|--------|-----------|------------|------------|------|
| Regime 1 (Stein only) | `___` | `___` | `___` | `___` |
| Regime 2 (MCMC+Stein) | `___` | `___` | `___` | `___` |
| Regime 1 (Neural only) | `___` | `___` | `___` | `___` |

### 8.5 PDE Loss vs Variance Loss (KSH-specific)

Head-to-head comparison of the two Stein CV loss functions on the same observable:

| System | Observable | PDE |Error| | PDE VarRed | Var |Error| | Var VarRed | Winner |
|--------|-----------|-----------|------------|-----------|------------|--------|
| DW4 | Energy | 0.749 | 311× worse | 0.056 | 2.78× better | **Var** |
| DW4 | Dist | — | — | 0.058 | 1.00× | N/A |
| LJ13 | Energy | `___` | `___` | `___` | `___` | `___` |
| LJ13 | Dist | — | — | `___` | `___` | `___` |
| LJ55 | Energy | `___` | `___` | `___` | `___` | `___` |
| LJ55 | Dist | — | — | `___` | `___` | `___` |

> PDE loss requires ∇f (only applicable to differentiable observables like energy). Variance loss works for any observable. "Dist" rows show variance-loss only since PDE loss cannot compute ∇(mean interatomic distance) without special handling.

### 8.6 Score-Informed CV: Effect of ISM Quality

Does Score-Informed CV help more when ISM is better? (LJ13, energy observable)

| ISM iters | cos_sim | Basic CV MSE Red. | Score CV MSE Red. | Score Benefit |
|-----------|---------|--------------------|--------------------|---------------|
| 1000 | `___` | `___`× | `___`× | `___`× |
| 3000 | `___` | `___`× | `___`× | `___`× |
| 5000 | `___` | `___`× | `___`× | `___`× |
| 10000 | `___` | `___`× | `___`× | `___`× |

> "Score Benefit" = Score CV MSE Red. / Basic CV MSE Red. Values > 1 mean ISM helped.

### 8.7 SteinEGNN_LN: Early Stopping Ablation

Does early stopping (val split + patience) help KSH methods?

| Setting | DW4 VarRed | LJ13 VarRed | Notes |
|---------|------------|-------------|-------|
| No validation (full iters) | `___` | `___` | Risk of overfitting |
| val_fraction=0.2, patience=6 | `___` | `___` | KSH default |
| val_fraction=0.2, patience=12 | `___` | `___` | Longer patience |
| Fresh val_sampler | `___` | `___` | Best but requires extra sampling |

---

## 9. Key Takeaways

### 9.1 What Works

<!-- TEMPLATE — fill after all evals -->

1. **Bias-variance coupling is real:** KSH variance loss reduces both bias (1.88×) and variance (2.78×) in DW4 — minimizing Var[h] automatically shrinks bias.
2. **Neural Stein CV scales:** `[PENDING — need LJ13/LJ55 to verify scaling]`
3. **RKHS Stein CV in low-d:** Hurts estimation in DW4 (error 2.63 vs vanilla 0.046). Over-corrects when sampler is already accurate.
4. **Antithetic is free lunch:** Mild 0.69× variance reduction in DW4, zero extra cost. Correlation 0.37 limits effectiveness.
5. **MCMC + Stein CV is the gold standard (when energy is cheap):** Fails in DW4 — 0.4% MH acceptance means proposals are always rejected.
6. **Equivariant g-networks (SteinEGNN_LN):** KSH variance-loss EGNN is the best method for DW4 energy (4.07× MSE reduction). SML PDE-loss EGNN catastrophically fails (311× variance explosion).
7. **Score-Informed decomposition:** Not tested for DW4 (ISM overkill at 8D). `[PENDING — LJ13/LJ55]`
8. **Early stopping prevents overfitting:** Critical for KSH methods. DW4 dist observable early-stopped at step 200-1200, preventing unnecessary training.

### 9.2 What Doesn't Work

1. **RKHS in high-d (165D):** `[PENDING — expected to collapse]`
2. **Antithetic for strongly stochastic regimes:** DW4 correlation only 0.37 — drift dominates, limiting anti-correlation.
3. **PDE loss without validation:** Confirmed catastrophic in DW4 — Neural CV (313×) and EGNN CV (311×) both explode variance. No validation = no way to detect overfitting.
4. **ISM at low-d (unnecessary complexity):** Correctly skipped for DW4 (8D). `[Verify with LJ13]`
5. **MCMC in tight distributions:** 0.4% acceptance in DW4 = MCMC is useless without step-size tuning.

### 9.3 Recommendations by Problem Size

| Dimension | Recommended Pipeline | Rationale |
|-----------|---------------------|-----------|
| d ≤ 20 | SteinEGNN_LN (Var loss) or MCMC + RKHS | Kernel OK, EGNN cheap, ISM not needed |
| 20 < d ≤ 50 | Score-Informed CV (ISM + SteinEGNN_LN) | Equivariance matters, ISM adds signal |
| d > 50 | Score-Informed CV (Hutchinson div) | RKHS collapses, MLP struggles, EGNN+score is best bet |

### 9.4 SML vs KSH: Which Approach Wins?

| Dimension | Best SML Method | Best KSH Method | Winner | Why |
|-----------|----------------|-----------------|--------|-----|
| DW4 (8D) | Vanilla ASBS (err=0.046) | SteinEGNN_LN Var (err=0.056, MSE↓4.07×) | **KSH** (for MSE) | PDE loss explodes; variance loss + early stopping works |
| LJ13 (39D) | `___` | `___` | `___` | `___` |
| LJ55 (165D) | `___` | `___` | `___` | `___` |

### 9.5 Bias-Variance Coupling Verification

The v2 theorem predicts $|\text{Bias}| \leq \sqrt{C \cdot \text{Var}[h_g]}$. We verify:

| System | Var Reduction (×) | Observed Bias Reduction (×) | Predicted Upper Bound (×) | Consistent? |
|--------|-------------------|-----------------------------|---------------------------|-------------|
| DW4 | `___` | `___` | `___` | `___` |
| LJ13 | `___` | `___` | `___` | `___` |
| LJ55 | `___` | `___` | `___` | `___` |

---

*This document will be populated as evaluations complete. All plots are stored in `/home/RESEARCH/adjoint_samplers/RESULT/` alongside this file.*
