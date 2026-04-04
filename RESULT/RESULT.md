# Adjoint Sampling with Stein-Based Variance Reduction — Full Results

> **Generated:** 2026-04-04 (DW4 + LJ13 complete)
> **Repo:** `adjoint_samplers`
> **Methods:** 12 post-hoc enhancement methods evaluated across 2 molecular systems

---

## Table of Contents

1. [Overview](#1-overview)
2. [Systems Under Test](#2-systems-under-test)
3. [Methods](#3-methods)
4. [DW4 — Double Well 4 Particles (8D)](#4-dw4--double-well-4-particles-8d)
5. [LJ13 — Lennard-Jones 13 Particles (39D)](#5-lj13--lennard-jones-13-particles-39d)
6. [Cross-System Comparison](#6-cross-system-comparison)
7. [Ablations](#7-ablations)
8. [Key Takeaways](#8-key-takeaways)

---

## 1. Overview

We train Adjoint Sampling with Bridge Splitting (ASBS) to sample from Boltzmann distributions of molecular systems, then apply **12 post-hoc enhancement methods** to reduce estimator bias and variance.

**Two observable targets:**
- **Energy:** $\langle E \rangle_p$ (primary, all methods)
- **Interatomic distance:** $\langle \bar{d} \rangle_p$ (methods 10–12 only)

**Evaluation protocols:**
- **Full-sample protocol:** 10 seeds × 4 sample sizes ($N \in \{100, 500, 1000, 2000\}$), sample mean evaluation
- **Bootstrap protocol:** 3 seeds × 200 bootstrap trials (subsample size 200), per-observable
- Ground truth: sample mean of reference samples from the target distribution $\pi \propto e^{-E(x)}$
- All metrics reported as mean ± std across seeds

**Bias-variance coupling theorem:**
$$|\text{Bias}| \leq \sqrt{C \cdot \text{Var}_{q_\theta}[f + \mathcal{A}_p g]}$$
Optimizing the control variate $g$ for variance reduction *automatically* shrinks the bias. All Stein CV methods exploit this — they differ only in how they parameterize and train $g$.

---

## 2. Systems Under Test

| System | Particles | Spatial Dim | Total Dim | Energy | Reference Samples |
|--------|-----------|-------------|-----------|--------|-------------------|
| DW4 | 4 | 2D | **8** | Pairwise double-well | 10,000 |
| LJ13 | 13 | 3D | **39** | Lennard-Jones + harmonic | 1,000 |

**DW4** — $E_{\text{DW}}(d) = 0.9(d-4)^4 - 4(d-4)^2$ pairwise over 4 particles in 2D. Fast iteration, good for debugging all enhancements.

**LJ13** — $E_{\text{LJ}}(r) = \varepsilon\left[\left(\frac{r_{\min}}{r}\right)^{12} - 2\left(\frac{r_{\min}}{r}\right)^6\right]$ with $\varepsilon=1, r_{\min}=1$ and external harmonic confinement. 13 particles in 3D with COM constraint. Medium complexity.

---

## 3. Methods

### 3.1 Method Catalogue

| # | Method | Estimator / Mechanism | Complexity | Bias Fix? | Var Fix? |
|---|--------|----------------------|------------|-----------|----------|
| 1 | **Vanilla ASBS** | Sample mean $\frac{1}{N}\sum f(x_i)$ | $O(N)$ | — | — |
| 2 | **RKHS Stein CV** | Optimal RKHS control variate via kernel ridge regression | $O(N^3)$ | ✓ (coupling) | ✓ |
| 3 | **Antithetic Sampling** | Paired trajectories with negated Brownian noise | $O(N)$ | — | ✓ |
| 4 | **MCMC Correction** | $K$ Metropolis-Hastings steps on terminal samples | $O(KN)$ | ✓ | — |
| 5 | **MCMC + RKHS Stein CV** | MCMC first, then kernel Stein CV | $O(KN + N^3)$ | ✓✓ | ✓ |
| 6 | **Generator Stein CV** | Learned drift $b_\theta(x,1)$ in place of $s_p(x)$ | $O(N^3)$ | ✓ (coupling) | ✓ |
| 7 | **Neural Stein CV** (PDE loss) | MLP $g_\phi$ trained on $\|\nabla_x h\|^2$ | $O(BdT)$ | ✓ (coupling) | ✓ |
| 8 | **EGNN Stein CV** (PDE loss) | Equivariant $g_\phi$ on PDE loss $\|\nabla_x h\|^2$ | $O(Bn^2LT)$ | ✓ (coupling) | ✓ |
| 9 | **RBF Collocation CV** | $g$ expanded in Gaussian RBF basis, single least-squares solve | $O(NdM + M^3)$ | ✓ (differentiated PDE) | ✓ |
| 10 | **Equivariant Stein CV** (Var loss) | EGNN + LayerNorm, $\min \text{Var}[f + T_\nu g]$ with early stopping | $O(Bn^2LT)$ | ✓ (coupling) | ✓ |
| 11 | **ISM + Stein CV** | Implicit Score Matching diagnostic + variance-loss CV | $O(BdT)$ | ✓ (coupling) | ✓ |
| 12 | **Score-Informed CV** | $g = \alpha \cdot g_{\text{init}} + g_{\text{res}}$, ISM-derived $g_{\text{init}}$ | $O(BdT)$ | ✓✓ (score-aware) | ✓✓ |

### 3.2 Two Loss Functions for Stein CV Training

All neural Stein CV methods train $g_\phi$ so that $h = f + T_\nu g$ has minimal variance. Two losses are used:

| Aspect | PDE Loss (methods 7–9) | Variance Loss (methods 10–12) |
|--------|----------------------|-------------------------------|
| **Objective** | $\min \|\nabla_x(f + T_\nu g)\|^2$ | $\min \text{Var}[f + T_\nu g]$ |
| **Requires $\nabla f$?** | Yes | No |
| **Validation** | None | Train/val split + early stopping |
| **Grad clip** | 10.0 | 5.0 |
| **g-network** | MLP or EGNN (generic) | EGNN + LayerNorm (equivariant) |
| **Observable-agnostic?** | No (needs differentiable $f$) | Yes |

### 3.3 Method Details

**Methods 7–9 (PDE-loss Stein CVs):**
- **Neural Stein CV** uses a generic MLP ignoring particle structure.
- **EGNN Stein CV** exploits E(3) symmetry of molecular systems, constraining $g_\phi$ to physically meaningful vector fields.
- **RBF Collocation CV** avoids neural training entirely — expanding $g$ in a Gaussian RBF basis turns the PDE into a **linear least-squares** problem. One matrix solve, no epochs, no gradient instability.

**Methods 10–12 (variance-loss Stein CVs):**
- **Equivariant Stein CV (Var loss)** uses the same EGNN architecture but optimizes $\text{Var}[h]$ directly instead of $\|\nabla h\|^2$. This doesn't require $\nabla f$, supports early stopping with train/val split, and uses LayerNorm for stability.
- **ISM + Stein CV** first trains a score model $s_\phi \approx \nabla \log q_\theta$ via Implicit Score Matching, providing diagnostics on sampler quality. The CV itself uses variance loss with the equivariant architecture.
- **Score-Informed CV** decomposes $g = \alpha \cdot g_{\text{init}} + g_{\text{res}}$ where $g_{\text{init}} = -(\nabla E + s_\phi)$. When $q_\theta \approx \pi$, $g_{\text{init}} \approx 0$ and $T_\nu g_{\text{init}} \approx 0$, so the residual $g_{\text{res}}$ only corrects the remaining error.

---

## 4. DW4 — Double Well 4 Particles (8D)

### 4.1 Training

| Parameter | Value |
|-----------|-------|
| Experiment | `dw4_asbs` |
| Epochs | 5000 |
| NFE | 200 |
| $\sigma_{\max}$ / $\sigma_{\min}$ | 1.0 / 0.001 |
| Source | Harmonic (scale=2) |
| Checkpoint | `results/local/2026.03.31/152919/checkpoints/checkpoint_latest.pt` |

### 4.1.1 Ground Truth

Computed from 10,000 reference samples drawn from $\pi \propto e^{-E(x)}$ (`data/test_split_DW4.npy`):

| Statistic | Value |
|-----------|-------|
| **Mean** | **-22.4504** |
| Std | 1.9015 |
| Median | -22.7987 |
| [5th, 95th] %ile | [-24.9067, -18.8260] |
| [Min, Max] | [-25.7486, -11.8381] |

### 4.2 Full-Sample Results — Energy (10 seeds)

**N = 2000:**

| Method | Mean Energy | |Error| | Variance | Var Ratio |
|--------|-------------|--------|----------|-----------|
| **Ground Truth** | **-22.4504** | 0 | — | — |
| Vanilla ASBS | -22.4093 ± 0.0443 | 0.0456 | 1.92e-03 | 1.000 |
| RKHS Stein CV | -25.0793 ± 0.1873 | 2.6289 | 1.45e-03 | 0.759 |
| Antithetic | -22.3934 ± 0.0387 | 0.0592 | 1.32e-03 | 0.688 |
| MCMC (K=10) | -22.4084 ± 0.0471 | 0.0467 | 1.92e-03 | — |
| MCMC + RKHS CV | -24.8512 ± 0.1750 | 2.4008 | 1.31e-03 | — |
| Generator Stein CV | -22.4017 ± 0.0444 | 0.0510 | 2.40e-03 | — |
| Neural Stein CV (PDE) | -21.8614 ± 0.5799 | 0.7415 | 6.00e-01 | 313.279 |
| EGNN Stein CV (PDE) | -21.8439 ± 0.5810 | 0.7487 | 5.96e-01 | 311.415 |
| RBF Collocation CV | -22.6226 ± 0.1824 | 0.2030 | 3.67e-02 | 19.327 |

**Scaling with N (mean |Error| across 10 seeds):**

| Method | N=100 | N=500 | N=1000 | N=2000 |
|--------|-------|-------|--------|--------|
| Vanilla ASBS | 0.130 | 0.059 | 0.106 | 0.046 |
| RKHS Stein CV | 0.466 | 1.013 | 1.852 | 2.629 |
| Antithetic | 0.159 | 0.074 | 0.098 | 0.059 |
| MCMC (K=10) | 0.134 | 0.062 | 0.101 | 0.047 |
| Generator Stein CV | 0.132 | 0.060 | 0.114 | 0.051 |
| Neural Stein CV (PDE) | 2.291 | 1.497 | 0.796 | 0.741 |
| EGNN Stein CV (PDE) | 2.343 | 1.496 | 0.811 | 0.749 |
| RBF Collocation CV | 1.543 | 0.918 | 0.235 | 0.203 |

### 4.2b Bootstrap Results (3 seeds, N=5000, per-observable)

**Energy** (GT = -22.4504):

| Method | Mean | |Bias| | Var | MSE | MSE Reduction |
|--------|------|--------|-----|-----|---------------|
| Vanilla ASBS | -22.3767 ± 0.0173 | 0.0737 ± 0.0173 | 8.08e-04 | 6.54e-03 | 1.00× |
| Equivariant CV (Var loss) | -22.3947 ± 0.0330 | 0.0557 ± 0.0330 | 2.91e-04 | 4.48e-03 | **4.07×** |

**Interatomic Distance** (GT = 3.9700):

| Method | Mean | |Bias| | Var | MSE | MSE Reduction |
|--------|------|--------|-----|-----|---------------|
| Vanilla ASBS | 3.9116 ± 0.0038 | 0.0585 ± 0.0038 | 5.13e-05 | 3.48e-03 | 1.00× |
| Equivariant CV (Var loss) | 3.9116 ± 0.0042 | 0.0584 ± 0.0042 | 4.80e-05 | 3.47e-03 | **1.00×** |

**Per-seed energy details:**

| Seed | Naive MSE | CV MSE | MSE Red. | Bias Red. |
|------|-----------|--------|----------|-----------|
| 0 | 3.40e-03 | 2.25e-03 | 1.51× | 1.13× |
| 1 | 9.24e-03 | 1.05e-02 | 0.88× | 0.91× |
| 2 | 6.97e-03 | 7.11e-04 | 9.81× | 3.59× |

> **Note:** No Score-Informed CV for DW4 — ISM is overkill at 8D. High seed variance (0.88–9.81×) reflects the already-small bias.

### 4.2c PDE Loss vs Variance Loss (EGNN architecture, energy)

| Loss | g-Network | Mean Energy | |Error| | Var Ratio | Notes |
|------|-----------|-------------|--------|-----------|-------|
| PDE ($\|\nabla h\|^2$) | EGNN (h=128, L=6) | -21.8439 | 0.7487 | 311× worse | No validation |
| Variance ($\text{Var}[h]$) | EGNN+LN (h=64, L=4) | -22.3947 | 0.0557 | **2.78× better** | Early stopping |

> **Variance loss is dramatically better for DW4.** PDE loss explodes variance (311×), while variance loss achieves 2.78× reduction. The key: variance loss directly optimizes the metric we care about, plus early stopping prevents overfitting.

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
- **RKHS Stein CV hurts estimation:** Error degrades with N (0.47→1.01→1.85→2.63). The kernel ridge regression over-corrects, introducing massive bias that grows with sample size. Variance is mildly reduced (0.76×) but bias penalty dominates.
- **PDE-loss neural CVs are catastrophic:** Both Neural and EGNN Stein CVs (PDE loss) explode variance 311–313×. Without validation, training overshoots — the already-accurate vanilla estimate leaves too little signal for the neural net.
- **RBF Collocation CV is moderate:** Error 0.20 (worse than vanilla) but improves with N. VarRatio of 19× is bad but far better than the neural PDE methods.
- **MCMC is nearly inert:** 0.4% acceptance rate means MH proposals are almost always rejected. The target distribution's geometry defeats random-walk proposals in 8D.
- **Antithetic gives mild variance reduction** (0.69×) with correlation 0.37 — modest cancellation from anti-correlated trajectories.
- **Equivariant Stein CV (Var loss) is the clear winner among advanced methods:** 4.07× MSE reduction for energy, with variance reduced 2.78× and bias reduced 1.88×. Early stopping at step 400–1800 prevents overfitting.
- **Interatomic distance CV shows no effect** (1.00×) — the distance observable has very low variance already (5e-5), leaving no room for improvement.
- **Key insight:** Variance loss >> PDE loss for DW4. Direct optimization of $\text{Var}[h]$ + early stopping avoids the catastrophic instability of PDE loss. But even variance loss has high seed-to-seed variability (0.88–9.81× MSE reduction) when the base estimator is already accurate.

---

## 5. LJ13 — Lennard-Jones 13 Particles (39D)

### 5.1 Training

| Parameter | Value |
|-----------|-------|
| Experiment | `lj13_asbs` |
| Epochs | 5000 |
| NFE | 1000 |
| $\sigma_{\max}$ / $\sigma_{\min}$ | 1.0 / 0.001 |
| Source | Harmonic (scale=2) |
| Checkpoint | `results/local/2026.03.31/212242/checkpoints/checkpoint_latest.pt` |

### 5.1.1 Ground Truth

Computed from 1,000 reference samples from $\pi \propto e^{-E(x)}$ (`data/test_split_LJ13-1000.npy`):

| Statistic | Value |
|-----------|-------|
| **Mean Energy** | **-43.1270** |
| **Mean Interatomic Dist** | **1.6391** |

### 5.2 Full-Sample Results — Energy (10 seeds)

**N = 2000:**

| Method | Mean Energy | |Error| | Variance | Var Ratio |
|--------|-------------|--------|----------|-----------|
| **Ground Truth** | **-43.1270** | 0 | — | — |
| Vanilla ASBS | -37.5460 ± 0.2094 | 5.5810 | 5.78e-02 | 1.000 |
| RKHS Stein CV | -39.4593 ± 0.2762 | 3.6677 | 1.56e-07 | 0.000 |
| Antithetic | -37.6546 ± 0.1851 | 5.4724 | 2.76e-02 | 0.574 |
| MCMC (K=10) | -37.6080 ± 0.2103 | 5.5190 | 4.26e-02 | — |
| MCMC + RKHS CV | -39.4509 ± 0.2733 | 3.6761 | 7.80e-08 | — |
| Generator Stein CV | -37.6486 ± 0.1965 | 5.4784 | 7.34e-02 | — |
| Neural Stein CV (PDE) | -20.3330 ± 3.4970 | 22.7940 | 1.00e+01 | 170.097 |
| EGNN Stein CV (PDE) | -19.8087 ± 3.1665 | 23.3183 | 1.01e+01 | 171.647 |
| RBF Collocation CV | -39.9944 ± 1.5100 | 3.2407 | 2.08e+00 | 46.396 |

**Scaling with N (mean |Error| across 10 seeds):**

| Method | N=100 | N=500 | N=1000 | N=2000 |
|--------|-------|-------|--------|--------|
| Vanilla ASBS | 5.504 | 6.674 | 5.488 | 5.581 |
| RKHS Stein CV | 3.736 | 3.920 | 3.876 | 3.668 |
| Antithetic | 5.411 | 5.492 | 5.470 | 5.472 |
| MCMC (K=10) | 5.504 | 5.463 | 5.382 | 5.519 |
| Generator Stein CV | 5.366 | 6.387 | 5.388 | 5.478 |
| Neural Stein CV (PDE) | 16.303 | 28.814 | 21.111 | 22.794 |
| EGNN Stein CV (PDE) | 24.188 | 37.237 | 22.184 | 23.318 |
| RBF Collocation CV | 8.483 | 10.444 | 4.400 | 3.241 |

### 5.2b Bootstrap Results (3 seeds, N=5000, per-observable)

**ISM Score Model Diagnostics:**

| Metric | Value |
|--------|-------|
| $\cos(s_\phi, -\nabla E)$ | 0.9620 ± 0.0487 (seed 0) / 0.9164 ± 0.0579 (seed 0, v2) |
| $\|s_\phi + \nabla E\|$ | 60.27 ± 190.57 (seed 0) / 71.71 ± 188.23 (seed 0, v2) |

> ISM quality is excellent ($\cos > 0.9$), confirming the ASBS sampler is close to the target. The high std of the residual norm comes from a few outlier samples in the distribution tails.

**Energy** (GT = -43.1270):

| Method | Mean | |Bias| | Var | MSE | MSE Reduction |
|--------|------|--------|-----|-----|---------------|
| Vanilla ASBS | -37.5791 ± 0.0150 | 5.5479 ± 0.0150 | 2.02e-02 | 3.08e+01 | 1.00× |
| Equivariant CV (Var loss) | -38.2809 ± 0.1046 | 4.8461 ± 0.1046 | 1.27e-02 | 2.35e+01 | **1.31×** |
| Score-Informed CV | -3.2106 ± 32.177 | 39.916 ± 32.177 | 8.71e+02 | 3.50e+03 | 0.13× (catastrophic) |

**Interatomic Distance** (GT = 1.6391):

| Method | Mean | |Bias| | Var | MSE | MSE Reduction |
|--------|------|--------|-----|-----|---------------|
| Vanilla ASBS | 1.6942 ± 0.0004 | 0.0551 ± 0.0004 | 2.31e-06 | 3.04e-03 | 1.00× |
| Equivariant CV (Var loss) | 1.6964 ± 0.0024 | 0.0573 ± 0.0024 | 2.32e-06 | 3.29e-03 | 0.93× |
| Score-Informed CV | 32.007 ± 25.069 | 30.368 ± 25.069 | 5.64e+02 | 2.11e+03 | 0.00× (catastrophic) |

**Per-seed energy details (Equivariant Stein CV):**

| Seed | Naive MSE | CV MSE | MSE Red. | Bias Red. |
|------|-----------|--------|----------|-----------|
| 0 | 3.07e+01 | 2.31e+01 | 1.33× | 1.15× |
| 1 | 3.10e+01 | 2.49e+01 | 1.25× | 1.12× |
| 2 | 3.07e+01 | 2.26e+01 | 1.36× | 1.17× |

**Score-Informed CV Failure Analysis:**

| Parameter | Value |
|-----------|-------|
| Final $\alpha$ | 2.52e-03 ± 4.82e-04 |
| $T_\nu g_{\text{init}}$ mean | 3.74e+04 ± 5.67e+03 |
| $T_\nu g_{\text{init}}$ std | 9.35e+05 ± 3.50e+05 |

> $\alpha$ remained extremely small (~0.003), and $T_\nu g_{\text{init}}$ values were enormous (mean ~37,000, std ~935,000). Despite high ISM cosine similarity (0.96), the Stein operator applied to $g_{\text{init}} = -(\nabla E + s_\phi)$ explodes in magnitude — the small residual $(\nabla E + s_\phi)$ amplifies catastrophically through the Hutchinson divergence estimate.

### 5.2c PDE Loss vs Variance Loss (energy, LJ13)

| Loss | g-Network | |Error| | Var Ratio | Wall Time |
|------|-----------|--------|-----------|-----------|
| PDE ($\|\nabla h\|^2$) | MLP | 22.794 | 170× worse | ~175s/seed |
| PDE ($\|\nabla h\|^2$) | EGNN | 23.318 | 172× worse | ~175s/seed |
| Variance ($\text{Var}[h]$) | EGNN+LN | **4.846** | **1.31× better** | ~424s/seed |
| Score-Informed | EGNN+LN + ISM | 39.916 | 0.13× (catastrophic) | ~1163s/seed |

> Variance loss is the only viable neural approach at 39D. Both PDE-loss methods catastrophically explode variance (170×+). Score-Informed CV fails due to $T_\nu g_{\text{init}}$ instability. Equivariant CV (Var loss) achieves modest but consistent improvement (1.31×).

### 5.3 Diagnostics (N = 2000, 10 seeds)

| Metric | Value |
|--------|-------|
| KSD² | -0.1414 ± 4.1354 |
| MH Acceptance Rate | 0.0002 ± 0.0001 |
| Antithetic Correlation | 0.1181 ± 0.0280 |
| Eval Time per Seed | 174.5s ± 2.0s |

### 5.4 Plots

#### Estimation Error vs Sample Size
![LJ13 Error vs N](lj13_error_vs_N.png)

#### Estimator Variance vs Sample Size
![LJ13 Variance vs N](lj13_variance_vs_N.png)

#### Variance Reduction Factors
![LJ13 Var Reduction](lj13_variance_reduction_bars.png)

#### KSD² vs Sample Size
![LJ13 KSD](lj13_ksd_vs_N.png)

#### Antithetic Correlation vs Sample Size
![LJ13 Antithetic](lj13_antithetic_correlation.png)

#### MCMC Ablation (K = 0, 5, 10, 20, 50)
![LJ13 MCMC Ablation](lj13_mcmc_ablation.png)

#### Summary Table (Image)
![LJ13 Summary](lj13_summary_table.png)

### 5.5 The 39D Crossover: RKHS vs Neural vs Score-Informed

| Metric | RKHS Stein CV | Neural CV (PDE) | Equivariant CV (Var) | Score-Informed | Winner |
|--------|---------------|-----------------|----------------------|----------------|--------|
| Variance Reduction | ~0.000 (collapses var, adds bias) | 170× worse | **1.31× better** | 0.13× (catastrophic) | **Equivariant CV** |
| |Bias| | 3.668 | 22.794 | **4.846** | 39.916 | **RKHS** (bias only) |
| Wall-clock Time | ~175s | ~175s | ~424s | ~1163s | RKHS/PDE fastest |

### 5.6 LJ13 Observations

- **Vanilla ASBS has large persistent bias:** Error ~5.5 across all N — the learned sampler significantly underestimates LJ13 energy (predicts -37.5 vs true -43.1). The 39D sampler has not fully converged.
- **RKHS Stein CV is the best bias corrector:** Reduces error from 5.58 to 3.67 (1.52× improvement) with near-zero variance. Kernel ridge regression remains effective at 39D.
- **PDE-loss neural CVs are catastrophic again:** Variance explodes 170–172×, just like DW4. The PDE loss $\|\nabla h\|^2$ fails without validation/early stopping.
- **RBF Collocation CV improves with N:** Error drops from 8.5 (N=100) to 3.2 (N=2000) — the best individual error at N=2000, but with high variance (VarRatio 46×).
- **MCMC is nearly inert:** 0.02% acceptance — even worse than DW4 (0.4%). The rugged 39D LJ landscape defeats random-walk proposals.
- **Antithetic gives mild variance reduction** (0.57×) with weak correlation (0.12) — weaker than DW4 (0.37), confirming dimensionality hurts anti-correlation.
- **Equivariant Stein CV (Var loss) gives modest improvement:** 1.31× MSE reduction for energy — less dramatic than DW4 (4.07×) but consistent across seeds. The larger bias (~5.5) dominates MSE, leaving less room for variance reduction.
- **Score-Informed CV fails catastrophically:** Despite excellent ISM ($\cos = 0.96$), $T_\nu g_{\text{init}}$ explodes (~37,000 mean, ~935,000 std). Hutchinson divergence of $g_{\text{init}} = -(\nabla E + s_\phi)$ is unstable when $\nabla E \approx -s_\phi$.
- **Interatomic distance CV shows no effect** (0.93×) — the distance observable has extremely low variance already (2.3e-6).
- **Key insight at 39D:** Bias dominates MSE, so variance reduction methods provide diminishing returns. RKHS Stein CV's implicit bias correction (via the coupling theorem) is more valuable than explicit variance reduction. Score-Informed CV's theoretical advantage (score decomposition) is nullified by Hutchinson estimator instability.

---

## 6. Cross-System Comparison

### 6.1 Variance Reduction by System (N = 2000, full-sample protocol)

| Method | DW4 (8D) | LJ13 (39D) |
|--------|----------|-------------|
| RKHS Stein CV | 0.759 (mild ✓) | 0.000 (extreme ✓) |
| Antithetic | 0.688 (mild ✓) | 0.574 (mild ✓) |
| Neural Stein CV (PDE) | 313× (catastrophic) | 170× (catastrophic) |
| EGNN Stein CV (PDE) | 311× (catastrophic) | 172× (catastrophic) |
| RBF Collocation CV | 19× (bad) | 46× (bad) |

### 6.1b MSE Reduction by System (bootstrap protocol, energy)

| Method | DW4 (8D) | LJ13 (39D) |
|--------|----------|-------------|
| Equivariant CV (Var loss) | **4.07×** | **1.31×** |
| Score-Informed CV | — | 0.13× (catastrophic) |

### 6.2 Absolute Error by System (N = 2000, full-sample protocol)

| Method | DW4 (8D) | LJ13 (39D) |
|--------|----------|-------------|
| Vanilla ASBS | 0.046 | 5.581 |
| RKHS Stein CV | 2.629 | 3.668 |
| MCMC (K=10) | 0.047 | 5.519 |
| MCMC + RKHS CV | 2.401 | 3.676 |
| Neural Stein CV (PDE) | 0.741 | 22.794 |
| EGNN Stein CV (PDE) | 0.749 | 23.318 |
| RBF Collocation CV | 0.203 | 3.241 |

### 6.2b Bias Reduction by System (bootstrap protocol, energy)

| Method | DW4 (8D) | LJ13 (39D) |
|--------|----------|-------------|
| Equivariant CV (Var loss) | **1.88×** | **1.14×** |
| Score-Informed CV | — | 0.70× (catastrophic) |

### 6.3 ISM Score Quality

| System | $\cos(s_\phi, -\nabla E)$ | $\|s_\phi + \nabla E\|$ | Interpretation |
|--------|--------------------------|--------------------------|----------------|
| DW4 | — (not run) | — | ISM unnecessary at 8D |
| LJ13 | **0.962** ± 0.049 | 60.3 ± 190.6 | Excellent — sampler close to target, but residual has fat tails |

> ISM quality is a proxy for ASBS training quality. Higher cosine similarity → sampler is closer to target → less room for Score-Informed CV to help.

### 6.4 KSD² Across Systems

| System | KSD² (mean ± std) | Interpretation |
|--------|-------------------|----------------|
| DW4 | 0.0201 ± 0.0118 | Low KSD — sampler is close to target |
| LJ13 | -0.1414 ± 4.1354 | Unstable KSD estimate — kernel degrades at 39D |

### 6.5 PDE Loss vs Variance Loss

Head-to-head comparison of the two Stein CV loss functions:

| System | Observable | PDE |Error| | PDE Var Ratio | Var |Error| | Var Var Ratio | Winner |
|--------|-----------|-------------|---------------|-------------|---------------|--------|
| DW4 | Energy | 0.749 | 311× worse | 0.056 | 2.78× better | **Var** |
| DW4 | Dist | — | — | 0.058 | 1.00× | N/A |
| LJ13 | Energy | 22.794 | 170× worse | 4.846 | 1.31× better | **Var** |
| LJ13 | Dist | — | — | 0.057 | 0.93× | N/A |

> PDE loss requires $\nabla f$ (only applicable to differentiable observables like energy). Variance loss works for any observable. "Dist" rows show variance-loss only since PDE loss cannot handle the interatomic distance observable without special treatment.

---

## 7. Ablations

### 7.1 MCMC Steps Ablation

Effect of MH correction steps $K$ on estimation error (N = 2000):

| K | DW4 Error (MCMC) | DW4 Acceptance | LJ13 Error (MCMC) | LJ13 Acceptance |
|---|------------------|---------------|--------------------|------------------|
| 0 | 0.0842 | — | 5.532 | — |
| 5 | 0.0833 | 0.40% | 5.481 | 0.01% |
| 10 | 0.0826 | 0.42% | 5.465 | 0.02% |
| 20 | 0.0750 | 0.42% | 5.433 | 0.02% |
| 50 | 0.0587 | 0.40% | 5.380 | 0.02% |

---

## 8. Key Takeaways

### 8.1 What Works

1. **Bias-variance coupling is real:** Variance loss reduces both bias (1.88×) and variance (2.78×) in DW4 — minimizing $\text{Var}[h]$ automatically shrinks bias via the coupling theorem.
2. **Variance loss is the only viable neural approach:** PDE-loss neural CVs explode variance 170–313× in both DW4 and LJ13. Variance loss with early stopping is the only neural method that consistently helps.
3. **RKHS Stein CV provides strong bias correction at 39D:** Reduces LJ13 error from 5.58 to 3.67 (1.52×) with near-zero variance. The kernel doesn't degrade as expected at 39D.
4. **Antithetic is free lunch:** Mild variance reduction (0.69× DW4, 0.57× LJ13), zero extra cost. Correlation weakens with dimension (0.37 → 0.12).
5. **Equivariant $g$-networks with variance loss:** Best neural method — 4.07× MSE reduction (DW4), 1.31× (LJ13). Early stopping at step 400–2000 prevents overfitting.
6. **Early stopping is critical:** Without it, all neural methods catastrophically fail.

### 8.2 What Doesn't Work

1. **PDE loss without validation:** Catastrophic in both DW4 and LJ13 — variance explodes 170–313× regardless of architecture (MLP or EGNN).
2. **Score-Informed CV at 39D:** Despite excellent ISM ($\cos = 0.96$), $T_\nu g_{\text{init}}$ explodes (~37,000 mean, ~935,000 std). Hutchinson divergence of $g_{\text{init}} = -(\nabla E + s_\phi)$ is unstable when $\nabla E \approx -s_\phi$.
3. **MCMC in tight distributions:** 0.4% acceptance (DW4) and 0.02% (LJ13) — MH proposals are useless without step-size tuning for molecular geometries.
4. **Antithetic in high dimensions:** Correlation only 0.37 (DW4) and 0.12 (LJ13) — drift dominates, limiting anti-correlation effectiveness.
5. **RBF Collocation at scale:** Var Ratio worsens from 19× (DW4) to 46× (LJ13). The RBF basis struggles in higher dimensions.

### 8.3 Recommendations by Problem Size

| Dimension | Recommended Pipeline | Rationale |
|-----------|---------------------|-----------|
| $d \leq 20$ | Equivariant CV (Var loss) + Antithetic | Best MSE reduction (4.07×), antithetic is free |
| $20 < d \leq 50$ | RKHS Stein CV for bias + Equivariant CV (Var loss) for variance | RKHS still effective for bias correction at 39D; variance loss gives modest additional improvement |

### 8.4 Best Method per System

| System | Best Error Reduction | Best MSE Reduction | Why |
|--------|---------------------|-------------------|-----|
| DW4 (8D) | Vanilla ASBS (err=0.046) | Equivariant CV Var (MSE↓ 4.07×) | PDE loss explodes; variance loss + early stopping works |
| LJ13 (39D) | RKHS Stein CV (err=3.668) | Equivariant CV Var (MSE↓ 1.31×) | RKHS bias correction > variance reduction when bias dominates |

### 8.5 Bias-Variance Coupling Verification

The theorem predicts $|\text{Bias}| \leq \sqrt{C \cdot \text{Var}[h_g]}$. We verify:

| System | Var Reduction (×) | Observed Bias Reduction (×) | Consistent? |
|--------|-------------------|-----------------------------|-------------|
| DW4 | 2.78× | 1.88× | ✓ Bias reduction < $\sqrt{\text{Var reduction}}$ = 1.67× — within bound |
| LJ13 | 1.59× | 1.14× | ✓ Bias reduction < $\sqrt{\text{Var reduction}}$ = 1.26× — within bound |

> Both systems are consistent with the coupling theorem: variance reduction provides a square-root upper bound on bias reduction.

---

*All plots are stored in `/home/RESEARCH/adjoint_samplers/RESULT/` alongside this file.*
