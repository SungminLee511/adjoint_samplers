# SML (adjoint_samplers) vs KSH (KSH_ASBS): Comprehensive Comparison

> **Repos:**
> - **SML** → `/home/RESEARCH/adjoint_samplers/` (enhancements in `enhancements/`)
> - **KSH** → `/home/RESEARCH/KSH_ASBS/` (enhancements in `stein_cv/`)
>
> Both repos build upon the same **ASBS** (Adjoint Schrödinger Bridge Sampler) base codebase from Facebook Research (Liu et al., NeurIPS 2025 Oral). The shared base lives in `adjoint_samplers/` within each repo and implements controlled SDE sampling from Boltzmann distributions. The core difference is the **post-hoc enhancement strategy** each repo implements on top of ASBS.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Shared Mathematical Baseline: ASBS](#2-shared-mathematical-baseline-asbs)
3. [Enhancement Philosophy](#3-enhancement-philosophy)
4. [Stein Operator: Shared Foundation](#4-stein-operator-shared-foundation)
5. [Stein Control Variates: The Core Divergence](#5-stein-control-variates-the-core-divergence)
6. [Score Matching (KSH-only)](#6-score-matching-ksh-only)
7. [RKHS Stein CV (SML-only)](#7-rkhs-stein-cv-sml-only)
8. [Neural Stein CV: Head-to-Head](#8-neural-stein-cv-head-to-head)
9. [Non-Stein Enhancements (SML-only)](#9-non-stein-enhancements-sml-only)
10. [Architecture Choices for g-Networks](#10-architecture-choices-for-g-networks)
11. [Divergence Computation](#11-divergence-computation)
12. [Training Hyperparameters](#12-training-hyperparameters)
13. [Experiment Setup & Evaluation Protocol](#13-experiment-setup--evaluation-protocol)
14. [Benchmark Problems & Configs](#14-benchmark-problems--configs)
15. [Summary Comparison Table](#15-summary-comparison-table)
16. [Theoretical Implications](#16-theoretical-implications)

---

## 1. Executive Summary

| Aspect | SML (adjoint_samplers) | KSH (KSH_ASBS) |
|--------|----------------------|----------------|
| **Enhancement count** | 7 methods | 2–3 methods |
| **Stein CV approach** | RKHS (kernel) + Neural (MLP) | Neural (EGNN) + Score-Informed Neural |
| **g-network architecture** | Plain MLP | Equivariant EGNN with LayerNorm |
| **Score model** | Not used | Pre-trained via Implicit Score Matching |
| **Non-Stein methods** | Antithetic sampling, MCMC, Generator kernel | None |
| **Evaluation style** | Unified pipeline, all methods at once | Per-experiment scripts, per-observable |
| **Scalability focus** | Neural CV for high-d, RKHS for low-d | EGNN everywhere, Hutchinson for high-d |
| **Mathematical novelty** | Bias-variance coupling theorem, generator kernel | Score-informed decomposition, running-mean variance |

**In short:** SML is a **breadth-first** approach (7 complementary methods), while KSH is a **depth-first** approach (fewer methods but deeper integration with equivariant architectures and learned score models).

---

## 2. Shared Mathematical Baseline: ASBS

Both repos share the identical ASBS base code. The core problem and training are the same.

### 2.1 Problem Statement

Sample from a Boltzmann target:

$$\pi(x) \propto \exp(-E(x))$$

where $E(x)$ is a known energy function (Double Well, Lennard-Jones, etc.).

### 2.2 Controlled SDE

Both repos solve this by training a controlled SDE:

$$dX_t = \big[f(X_t, t) + g(t)^2 u_\theta(X_t, t)\big] dt + g(t)\, dW_t, \quad X_0 \sim \mu$$

where:
- $f(t,x)$ = reference SDE drift (zero for VE-SDE)
- $g(t)$ = diffusion coefficient
- $u_\theta$ = learned neural controller (EGNN or FourierMLP)
- $\mu$ = source distribution (harmonic, Gaussian)

### 2.3 Adjoint Matching Loss

Both repos train $u_\theta$ via:

$$\mathcal{L}_{\text{AM}}(\theta) = \frac{1}{B} \sum_{i=1}^B \left\| u_\theta(t_i, x_i) - (-Y_{t_i}^{(i)}) \right\|^2$$

where $Y_t$ is the backward adjoint state:

$$dY_t = -[\nabla_x f(X_t, t)]^\top Y_t \, dt + \nabla E(X_t)\, dt$$

with terminal condition $Y_1 = \nabla E(X_1)$.

### 2.4 Two-Stage ASBS Training

Both repos implement the same two-stage Iterative Proportional Fitting:

1. **Stage 1 — Adjoint Matching (AM):** Train controller $u_\theta$ for `adj_num_epochs_per_stage` epochs
2. **Stage 2 — Corrector Matching (CM):** Train corrector $h_\phi$ for `ctr_num_epochs_per_stage` epochs
3. Alternate AM ↔ CM until total epochs exhausted

### 2.5 Reference SDEs (Identical)

| SDE | Drift $f(t,x)$ | Diffusion $g(t)$ | Used for |
|-----|----------------|------------------|----------|
| VE-SDE | $0$ | $\sigma_{\min}(\sigma_{\max}/\sigma_{\min})^t \sqrt{2\log(\sigma_{\max}/\sigma_{\min})}$ | All benchmarks |
| VP-SDE | $-\frac{\beta_t}{2}x$ | $\sqrt{\beta_t}$ | Alternative |
| GraphVE/VP | Same as above | Same, with COM projection on noise | Particle systems |

### 2.6 Energy Functions (Identical)

| Energy | Formula | Dimensions |
|--------|---------|-----------|
| **DW4** | $E(x) = \sum_{\text{pairs}} a(d_{ij}-d_0)^4 + b(d_{ij}-d_0)^2$ | 4 particles × 2D = 8D |
| **LJ13** | $E(x) = \sum_{\text{pairs}} \varepsilon\left[(\frac{r_m}{r})^{12} - 2(\frac{r_m}{r})^6\right] + \frac{\lambda}{2}\sum_i\|x_i-\bar{x}\|^2$ | 13 particles × 3D = 39D |
| **LJ55** | Same LJ formula | 55 particles × 3D = 165D |

Parameters identical: $a=0.9, b=-4, d_0=4$ (DW4); $\varepsilon=1, r_m=1, \lambda_{\text{osc}}=1$ (LJ).

### 2.7 Model Architectures (Identical for base)

| Model | Used for | Config |
|-------|----------|--------|
| **EGNN** | Particle systems (DW4, LJ13, LJ55) | hidden_nf=128, n_layers=5, attention=True, tanh=True |
| **FourierMLP** | Simple/demo systems | channels=64, num_layers=4, SiLU |

---

## 3. Enhancement Philosophy

### SML: Breadth-First, Post-Hoc Toolkit

SML treats the trained ASBS sampler as a **black box** and applies **7 independent post-hoc methods** to improve estimation quality:

```
ASBS samples → {KSD, Stein CV (RKHS), Stein CV (Neural), Antithetic,
                 MCMC, Generator Stein, Hybrid MCMC+Stein}
```

**Design principles:**
- Each method is self-contained (import and use independently)
- Unified evaluation pipeline (`enhanced_evaluator.py`) runs all 7 at once
- Methods can be combined (e.g., MCMC first, then Stein CV)
- Theoretical backing: bias-variance coupling theorem motivates variance reduction

### KSH: Depth-First, Architecture-Aware

KSH focuses on **fewer but more sophisticated** methods that leverage domain knowledge:

```
ASBS samples → Score Matching → {Basic Stein CV, Score-Informed Stein CV}
```

**Design principles:**
- Equivariant architecture (EGNN) used for both sampler and control variate
- Pre-trained score model provides strong initialization for CV
- Score-informed decomposition: $g = \alpha \cdot g_{\text{init}} + g_{\text{res}}$
- Careful variance-minimization training with running mean and validation

---

## 4. Stein Operator: Shared Foundation

Both repos implement the same fundamental **Stein operator** for a Boltzmann target $\nu \propto \exp(-E)$:

$$\mathcal{T}_\nu g(x) = \nabla \log \nu(x)^\top g(x) + \nabla \cdot g(x) = -\nabla E(x)^\top g(x) + \nabla \cdot g(x)$$

**Stein's Identity:** For any sufficiently smooth $g$:

$$\mathbb{E}_\nu[\mathcal{T}_\nu g(X)] = 0$$

This means $\mathcal{T}_\nu g$ is a **zero-mean function under the target**, making it a valid control variate.

### Control Variate Construction

Both repos construct the corrected estimator:

$$h(x) = f(x) + \mathcal{T}_\nu g_\psi(x)$$

so that $\mathbb{E}_\nu[h] = \mathbb{E}_\nu[f]$ (unbiased under target), while $\mathrm{Var}[h] < \mathrm{Var}[f]$ (reduced variance).

---

## 5. Stein Control Variates: The Core Divergence

This is where the two repos fundamentally differ.

### 5.1 SML Approach: Two Separate Methods

SML provides **two independent** Stein CV implementations:

#### A) RKHS Stein CV (`enhancements/stein_cv.py`)

A **non-parametric, kernel-based** method. No neural network training.

**Algorithm:**
1. Compute Stein kernel matrix $K_p \in \mathbb{R}^{N \times N}$ using RBF base kernel
2. Solve regularized system: $A = K_p + \lambda N I$
3. Compute: $a = A^{-1}f$, $b = A^{-1}\mathbf{1}$
4. **Normalized CF estimate:** $\hat{\mu}^{\text{SCV}} = \frac{\mathbf{1}^\top a}{\mathbf{1}^\top b}$

**Stein Kernel Formula:**

$$k_p(x, x') = s(x)^\top k(x,x') s(x') + s(x)^\top \nabla_{x'} k + (\nabla_x k)^\top s(x') + \mathrm{tr}(\nabla_x \nabla_{x'} k)$$

where $k(x,x') = \exp(-\|x-x'\|^2 / 2\ell^2)$ and $s = -\nabla E$.

**Complexity:** $O(N^3)$ solve + $O(N^2 d)$ kernel. Capped at `max_stein_samples=2000`.

#### B) Neural Stein CV (`enhancements/neural_stein_cv.py`)

A **parametric, MLP-based** method for high dimensions.

**Loss (PDE form):**

$$\mathcal{L}_{\text{PDE}}(\phi) = \mathbb{E}_{q_\theta}\left[\left\|\nabla_x\left[f(x) + \mathcal{T}_\nu g_\phi(x)\right]\right\|^2\right]$$

**Insight:** If $f + \mathcal{T}_\nu g^* = c$ (constant), then $\nabla_x[f + \mathcal{T}_\nu g^*] = 0$. The unknown constant $c$ vanishes under differentiation.

**Architecture:** Plain MLP with SiLU activation, last-layer near-zero init.

### 5.2 KSH Approach: Neural-Only, Score-Informed

KSH provides **one neural method with two variants**, both using EGNN:

#### A) Basic SteinBiasCorrector (`stein_cv/neural_cv.py`)

**Loss (Variance form):**

$$\mathcal{L}(\psi) = \mathrm{Var}\left[f(x) + \mathcal{T}_\nu g_\psi(x)\right]$$

with optional bias penalty: $+ \lambda_{\text{bias}} \cdot \left(\mathbb{E}[\mathcal{T}_\nu g_\psi]\right)^2$

#### B) ScoreInformedSteinCV (`stein_cv/neural_cv.py`)

A **two-part decomposition** leveraging a pre-trained score model $s_\phi \approx \nabla \log q_\theta$:

$$g = \alpha \cdot g_{\text{init}} + g_{\text{res}}$$

where:
- $g_{\text{init}}(x) = -(\nabla E(x) + s_\phi(x))$
- $\mathcal{T}_\nu g_{\text{init}}$ is **precomputed once** (frozen)
- Only $\alpha$ (scalar, initialized at 0.01 via log-parametrization) and $g_{\text{res}}$ are optimized

**Loss:**

$$\mathcal{L} = \mathrm{Var}\left[(h - \bar{h}_{\text{running}})^2\right], \quad h = f + \alpha \cdot \mathcal{T}_\nu g_{\text{init}} + \mathcal{T}_\nu g_{\text{res}}$$

Uses exponential moving average: $\bar{h} \leftarrow 0.99 \cdot \bar{h} + 0.01 \cdot \mathrm{mean}(h)$

### 5.3 Side-by-Side: Loss Function Comparison

| Aspect | SML (Neural) | KSH (Basic) | KSH (Score-Informed) |
|--------|-------------|-------------|---------------------|
| **Objective** | $\min \|\nabla_x h\|^2$ (PDE) | $\min \mathrm{Var}[h]$ | $\min \mathrm{Var}[(h-\bar{h})^2]$ |
| **What is $h$** | $f + \mathcal{T}_\nu g_\phi$ | $f + \mathcal{T}_\nu g_\psi$ | $f + \alpha \mathcal{T}_\nu g_{\text{init}} + \mathcal{T}_\nu g_{\text{res}}$ |
| **Requires $\nabla f$?** | Yes (via autograd) | No | No |
| **Requires score model?** | No | No | Yes (pre-trained ISM) |
| **Running mean?** | No | No | Yes (EMA, 0.99) |
| **Bias penalty?** | No (implicit via PDE) | Optional ($\lambda_{\text{bias}}$) | No |

**Key Mathematical Difference:**

- **SML's PDE loss** works because a perfect CV makes $h = c$ (constant), so $\nabla h = 0$. This is a **first-order optimality condition** that avoids estimating the constant.
- **KSH's variance loss** directly minimizes the sample variance of $h$. This is more straightforward but requires careful centering (hence the running mean in the score-informed variant).

---

## 6. Score Matching (KSH-only)

KSH includes a dedicated **Implicit Score Matching** module that SML does not have.

### 6.1 Purpose

Learn the **log-density gradient of the learned sampler** $q_\theta$:

$$s_\phi(x) \approx \nabla \log q_\theta(x)$$

This is **not** the target score $-\nabla E$, but the score of the **approximate distribution** produced by the trained ASBS sampler.

### 6.2 ISM Loss

$$\mathcal{L}_{\text{ISM}}(\phi) = \mathbb{E}_{q_\theta}\left[\frac{1}{2}\|s_\phi(x)\|^2 + \nabla \cdot s_\phi(x)\right]$$

This is the classic Hyvärinen score matching objective — requires only samples from $q_\theta$, not the density itself.

### 6.3 Architecture: ScoreEGNN

```python
ScoreEGNN(n_particles, spatial_dim, hidden_nf=128, n_layers=4, coord_init_gain=0.1)
```

- Same EGNN architecture as the ASBS controller
- Evaluates at $t=1.0$ (terminal time)
- Coordinate layer weights reinitialized with Xavier uniform (gain=0.1) for stability
- Outputs $s_\phi(x) \in \mathbb{R}^d$

### 6.4 Why SML Doesn't Have This

SML's approach doesn't need $\nabla \log q_\theta$ because:
- The **RKHS method** uses only the target score $-\nabla E$
- The **Neural CV (PDE loss)** uses $\nabla f$ and $\nabla E$ but not $\nabla \log q_\theta$
- The **Generator Stein kernel** uses the learned SDE drift $b_\theta$ as a proxy (see Section 9.2)

KSH's score-informed CV needs $s_\phi$ to construct $g_{\text{init}} = -(\nabla E + s_\phi)$, which ideally equals zero if $q_\theta = \pi$ (perfect sampling).

---

## 7. RKHS Stein CV (SML-only)

SML provides a **non-parametric, closed-form** Stein CV that KSH does not have.

### 7.1 Method

Uses the Stein kernel $K_p$ to perform **kernel ridge regression** in the RKHS:

$$\hat{\mu}^{\text{CF}} = \frac{\mathbf{1}^\top (K_p + \lambda N I)^{-1} f}{\mathbf{1}^\top (K_p + \lambda N I)^{-1} \mathbf{1}}$$

### 7.2 Properties

| Property | Value |
|----------|-------|
| Training | None (closed-form solve) |
| Compute | $O(N^3)$ for Cholesky + $O(N^2 d)$ for kernel |
| Memory | $O(N^2)$ for kernel matrix |
| Scalability | Poor above $N \approx 2000$ |
| Regularization | $\lambda = 10^{-4}$ |
| Bandwidth | Median heuristic with subsampling for $N > 2000$ |

### 7.3 Advantages over Neural CV

- **No hyperparameter tuning** (LR, epochs, architecture)
- **Deterministic** (no training randomness)
- **Optimal in RKHS** (minimizes variance among all functions in the kernel's RKHS)

### 7.4 Limitations

- $O(N^2)$ memory, $O(N^3)$ compute → impractical for large $N$
- RBF kernel degrades in high dimensions (curse of dimensionality)
- Not equivariant (ignores particle symmetry)

---

## 8. Neural Stein CV: Head-to-Head

Both repos implement neural Stein CVs, but with fundamentally different architectures and loss functions.

### 8.1 Architecture

| Aspect | SML | KSH |
|--------|-----|-----|
| **Network type** | Plain MLP | SteinEGNN_LN (equivariant GNN) |
| **Input** | Flat vector $(B, d)$ | Flat vector $(B, d)$ → reshaped to particles |
| **Symmetry** | None (generic) | SE(3)-equivariant (rotation, translation, permutation) |
| **LayerNorm** | No | Yes (in every MLP within GCL layers) |
| **Activation** | SiLU (default), GELU, Tanh, Softplus | SiLU |
| **Hidden dim** | $\min(256, \max(64, 2d))$ | 64 (DW4), 128 (LJ13) |
| **Layers** | 3 | 4–5 |
| **Output init** | Last layer $\sim \mathcal{N}(0, 10^{-3})$ | Default PyTorch init |
| **COM removal** | No (MLP doesn't know about particles) | Yes (remove_mean on output velocity) |

**SML's MLP:**
```python
NeuralSteinCV(dim=d, hidden_dim=min(256, max(64, 2*d)), n_layers=3, activation='silu')
# Architecture: Linear(d, h) → SiLU → [Linear(h, h) → SiLU]×(L-2) → Linear(h, d)
```

**KSH's SteinEGNN_LN:**
```python
SteinEGNN_LN(n_particles=N, spatial_dim=S, hidden_nf=H, n_layers=L, tanh=False)
# Architecture: Embedding → [E_GCL_LN (with LayerNorm)]×L → Output velocity
# Each E_GCL_LN: Edge MLP + Node MLP + Coord MLP, all with LayerNorm
```

### 8.2 Loss Function

| Aspect | SML (PDE Loss) | KSH (Variance Loss) |
|--------|---------------|---------------------|
| **Formula** | $\mathbb{E}[\|\nabla_x(f + \mathcal{T}_\nu g)\|^2]$ | $\mathrm{Var}[f + \mathcal{T}_\nu g]$ |
| **What's minimized** | Gradient of corrected function | Variance of corrected function |
| **Optimal solution** | $f + \mathcal{T}_\nu g^* = c$ | $f + \mathcal{T}_\nu g^* = c$ |
| **Needs $\nabla f$** | Yes | No |
| **Computational cost** | Higher (second derivatives of $g$) | Lower (first derivatives of $g$) |
| **Numerical stability** | Needs gradient clipping (10.0) | Needs gradient clipping (5.0) |

### 8.3 Training Protocol

| Parameter | SML | KSH (Basic) | KSH (Score-Informed) |
|-----------|-----|------------|---------------------|
| **Optimizer** | Adam | Adam | Adam |
| **LR** | 1e-3 | 1e-3 | 1e-3 |
| **LR Schedule** | CosineAnnealing | CosineAnnealing | CosineAnnealing |
| **Grad clip** | max_norm=10.0 | max_norm=5.0 | max_norm=5.0 |
| **Epochs/Iters** | 500 (DW4), 1000 (LJ55) | 5000–10000 iterations | 5000–10000 iterations |
| **Batch size** | 256 | 512–2500 | 512–2500 |
| **Validation** | None (no early stopping) | 0.2 split or fresh samples | 0.2 split or fresh samples |
| **Early stopping** | No | Yes (patience=6) | Yes (patience=6) |
| **Weight decay** | 0 | 0 (configurable) | 0 (configurable) |

### 8.4 Implications of Architecture Choice

**SML's MLP approach:**
- ✅ Simple, fast to train
- ✅ Works for any dimensionality (no particle structure needed)
- ❌ Doesn't respect symmetries → needs more data to learn them
- ❌ No COM constraint → may produce invalid corrections for particle systems
- ❌ Curse of dimensionality for high-d (no inductive bias)

**KSH's EGNN approach:**
- ✅ Respects rotation, translation, permutation symmetry → better sample efficiency
- ✅ COM-free by construction (remove_mean on output)
- ✅ LayerNorm improves training stability
- ❌ More complex, slower per iteration
- ❌ Only works for particle systems (not general distributions)
- ❌ Requires graph construction overhead

---

## 9. Non-Stein Enhancements (SML-only)

SML provides 4 additional methods that KSH does not implement.

### 9.1 KSD Diagnostic (`enhancements/stein_kernel.py`)

**Kernel Stein Discrepancy** — a diagnostic metric, not an estimator:

$$\widehat{\text{KSD}}^2 = \frac{1}{N(N-1)} \sum_{i \neq j} k_p(X_i, X_j)$$

U-statistic form (diagonal removed for unbiasedness). Used to track distributional quality over training.

### 9.2 Generator Stein Kernel (`enhancements/generator_stein.py`)

Replaces the generic score $s_p = -\nabla E$ with the **learned SDE drift** at terminal time:

$$b_\theta(x) = f(x, 1) + g(1)^2 u_\theta(x, 1)$$

**Modified Stein kernel:**

$$k_{\text{gen}}(x, x') = b(x)^\top k b(x') + b(x)^\top \nabla_{x'} k + (\nabla_x k)^\top b(x') + \frac{g(1)^2}{2} \mathrm{tr}(\nabla_x \nabla_{x'} k)$$

**Note the factor $g(1)^2/2$ in Term 4** — this comes from the diffusion coefficient of the generator, unlike the standard Stein kernel which has a simple $d/\ell^2$ term.

**Intuition:** If the learned drift captures the distribution structure better than the generic Langevin drift, the generator kernel may correlate better with observables.

**This is SML's alternative to KSH's score-informed approach.** Instead of learning a separate score model, SML reuses the already-trained SDE drift.

### 9.3 Antithetic SDE Sampling (`enhancements/antithetic.py`)

Generate paired trajectories with negated Brownian noise:

$$\text{Original:} \quad X_{t+dt} = X_t + b(t, X_t)\,dt + g(t)\sqrt{dt}\,\xi$$
$$\text{Antithetic:} \quad X'_{t+dt} = X'_t + b(t, X'_t)\,dt - g(t)\sqrt{dt}\,\xi$$

Same initial state, same noise magnitude, opposite sign.

**Estimator:**

$$\hat{\mu}^{\text{anti}} = \frac{1}{2N}\sum_{i=1}^N \left[f(X_1^{(i)}) + f(X_1'^{(i)})\right]$$

**Properties:**
- Zero additional energy evaluations
- Negative correlation when drift dominates noise → variance reduction
- Uses `sde.randn_like()` for COM-free noise projection
- Uses `sde.propagate()` for state updates

### 9.4 MCMC Post-Correction (`enhancements/mcmc_correction.py`)

Random-walk Metropolis-Hastings on terminal ASBS samples:

```
For k = 1, ..., K:
    x' = x + σ·ξ,  ξ ~ N(0, I)
    Accept x' with probability min(1, exp(-(E(x') - E(x))))
```

**Step size (auto):**

$$\sigma = \frac{2.38}{\sqrt{d}} \cdot \hat{\sigma}_{\text{marginal}}$$

Optimal scaling for high-dimensional Gaussian targets (target acceptance ≈ 0.234).

**Properties:**
- Asymptotically removes bias (converges to $\pi$ regardless of $q_\theta$ quality)
- ASBS provides excellent initialization → fast mixing
- Can be combined with Stein CV (hybrid pipeline)

### 9.5 Hybrid MCMC + Stein CV

SML's unified evaluator runs MCMC first, then applies RKHS Stein CV to the corrected samples:

```
ASBS samples → MCMC correction → Stein CV (RKHS) → final estimate
```

This combines bias removal (MCMC) with variance reduction (Stein CV).

---

## 10. Architecture Choices for g-Networks

### 10.1 Complete Architecture Comparison

```
SML Neural CV (MLP):
  Input: (B, d) ──→ Linear(d, h) ──→ SiLU ──→ [Linear(h, h) → SiLU] × (L-2)
                 ──→ Linear(h, d) ──→ Output: (B, d)

  Last layer: weights ~ N(0, 0.001), bias = 0

KSH Neural CV (SteinEGNN_LN):
  Input: (B, d) ──→ reshape (B, N, S)
                ──→ Fully-connected graph edges
                ──→ Embedding: Linear(1, h)
                ──→ [E_GCL_LN: EdgeMLP→NodeMLP→CoordMLP (all with LayerNorm)] × L
                ──→ Output velocity: (B, N, S) ──→ remove_mean ──→ flatten (B, d)
```

### 10.2 Equivariance Properties

| Property | SML (MLP) | KSH (EGNN) |
|----------|-----------|------------|
| Translation equivariance | ❌ | ✅ (remove_mean) |
| Rotation equivariance | ❌ | ✅ (E_GCL structure) |
| Permutation equivariance | ❌ | ✅ (graph message passing) |
| Reflection equivariance | ❌ | ✅ (inherent in EGNN) |
| Arbitrary distributions | ✅ | ❌ (particle systems only) |

---

## 11. Divergence Computation

Both repos need $\nabla \cdot g(x)$ for the Stein operator. The approaches are similar but differ in defaults and thresholds.

### 11.1 Exact Divergence

Both repos implement the same algorithm:

$$\nabla \cdot g(x) = \sum_{j=1}^d \frac{\partial g_j}{\partial x_j}$$

Requires $d$ backward passes through $g$. Only practical for low-dimensional systems.

### 11.2 Hutchinson Trace Estimator

$$\nabla \cdot g(x) \approx \frac{1}{K}\sum_{k=1}^K v_k^\top J_g(x) v_k, \quad v_k \sim \mathcal{N}(0, I)$$

Requires only $K$ backward passes (typically $K \ll d$).

### 11.3 Comparison

| Aspect | SML | KSH |
|--------|-----|-----|
| **Exact threshold** | $d \leq 20$ | Manual choice per experiment |
| **Hutchinson default probes** | 1 | 4 (CV), 8 (score matching) |
| **Probe distribution** | $\mathcal{N}(0, I)$ | $\mathcal{N}(0, I)$ |
| **DW4 (d=8)** | Exact | Exact (default=1 probe) |
| **LJ13 (d=39)** | Hutchinson (1 probe) | Hutchinson (4 probes CV, 8 probes ISM) |
| **LJ55 (d=165)** | Hutchinson (1 probe) | Not yet tested |

**Notable:** KSH uses **more probes** for divergence estimation, which reduces variance of the Hutchinson estimator but increases compute per step.

---

## 12. Training Hyperparameters

### 12.1 ASBS Base Training (Identical)

| Parameter | DW4 | LJ13 | LJ55 |
|-----------|-----|------|------|
| NFE (discretization steps) | 200 | 1000 | 1000 |
| Total epochs | 5000 | 5000 | 5000 |
| AM epochs per stage | 200 (KSH) / 300 (SML) | 300 | 300 |
| CM epochs per stage | 20 | 20 | 20 |
| Controller LR | 1e-4 | 1e-4 | 1e-4 |
| Buffer size | 10000 | 10000 | 10000 |
| Buffer duplicates | 10 | 25 | 25 |
| Resample batch size | 512 | 512 | 512 |
| Train batch size | 512 | 512 | 512 |
| Train iterations/epoch | 100 | 100 | 100 |
| σ_max | 1.0 | 1.0 | 1.0 |
| σ_min | 0.001 | 0.001 | 0.001 |

> **Note:** SML uses `adj_num_epochs_per_stage=300` for DW4 while KSH uses 200. This is a minor difference.

### 12.2 Enhancement Training

#### SML Neural Stein CV

| Parameter | DW4 | LJ13 | LJ55 |
|-----------|-----|------|------|
| Architecture | MLP | MLP | MLP |
| Hidden dim | 64 | 128 | 256 |
| Layers | 3 | 3 | 3 |
| Epochs | 500 | 500 | 1000 |
| Batch size | 256 | 256 | 256 |
| LR | 1e-3 | 1e-3 | 1e-3 |
| Scheduler | CosineAnnealing | CosineAnnealing | CosineAnnealing |
| Grad clip | 10.0 | 10.0 | 10.0 |
| Divergence | Exact | Hutchinson (1) | Hutchinson (1) |
| Validation | None | None | None |

#### KSH Neural Stein CV

| Parameter | DW4 | LJ13 |
|-----------|-----|------|
| Architecture | SteinEGNN_LN | SteinEGNN_LN |
| Hidden dim | 64 | 128 |
| Layers | 4 | 5 |
| Iterations | 10000 | 5000 |
| Batch size | 2500 | 256 |
| LR | 1e-3 | 1e-3 |
| Scheduler | CosineAnnealing | CosineAnnealing |
| Grad clip | 5.0 | 5.0 |
| Divergence | Exact (default) | Hutchinson (4 probes) |
| Validation | 0.2 split, patience=6 | 0.2 split, patience=6 |

#### KSH Score Matching (ISM)

| Parameter | LJ13 |
|-----------|------|
| Architecture | ScoreEGNN |
| Hidden dim | 128 |
| Layers | 4 |
| Iterations | 5000 |
| Batch size | 256 |
| LR | 3e-4 |
| Divergence | Hutchinson (8 probes) |
| Coord init gain | 0.1 |

---

## 13. Experiment Setup & Evaluation Protocol

### 13.1 SML: Unified Multi-Method Evaluation

**Script:** `run_evaluation.py` / `eval_enhanced.py`

**Protocol:**
1. Load ASBS checkpoint
2. Generate $N$ terminal samples
3. Run **all 7 methods** simultaneously via `evaluate_enhanced()`
4. Report: mean, variance, variance reduction, error (vs ground truth)

**Observables:** Primarily energy $E(x)$

**Multi-seed:** Configurable via Hydra (`seed=0,1,2 -m`)

**Output:** Single dict with all metrics for all methods

### 13.2 KSH: Per-Observable Experiment Scripts

**Scripts:** `experiments/dw4_steincv_final.py`, `experiments/lj13_steincv.py`

**Protocol (DW4):**
1. Load ASBS checkpoint
2. Generate fresh samples per seed (3 seeds)
3. For **each observable separately** (energy, interatomic distance):
   - Train a separate SteinEGNN_LN
   - Evaluate naive vs CV-corrected
4. Report per-seed: mean, bias, MSE, reduction ratios

**Protocol (LJ13):**
1. Load checkpoint, generate samples
2. **Phase 1:** Train ISM score model ($s_\phi$)
3. **Phase 2a:** Train basic Stein CV
4. **Phase 2b:** Train score-informed Stein CV
5. **Evaluation:** 200 trials, subsample_size=200, compare 3 methods on 2 observables

**Observables:** Energy AND interatomic distance (both tracked)

**Multi-trial:** 200 bootstrap trials with subsampling (KSH) vs multi-seed generation (SML)

### 13.3 Key Differences in Evaluation

| Aspect | SML | KSH |
|--------|-----|-----|
| **Methods per run** | 7 (all at once) | 2–3 (per script) |
| **Observables** | Energy only (in unified eval) | Energy + interatomic distance |
| **Statistical protocol** | Multi-seed, full samples | Multi-trial bootstrap subsampling |
| **Ground truth** | Reference samples (if available) | Reference samples |
| **Metrics** | Error, variance, VR factor | Bias, MSE, bias/MSE reduction |
| **Visualization** | Publication-quality plots module | Manual analysis |

---

## 14. Benchmark Problems & Configs

### 14.1 DW4 (Double Well, 4 particles)

| Setting | SML | KSH |
|---------|-----|-----|
| Dimensions | 8 (4×2) | 8 (4×2) |
| NFE | 200 | 200 |
| Source | Harmonic (scale=2.0) | Harmonic (scale=2.0) |
| SDE | GraphVESDE | GraphVESDE |
| Model | EGNN (h=128, L=5) | EGNN (h=128, L=5) |
| Reference data | `data/test_split_DW4.npy` | `data/test_split_DW4.npy` |
| CV architecture | MLP (h=64, L=3) | SteinEGNN_LN (h=64, L=4) |
| CV training | 500 epochs | 10000 iterations |
| Eval samples | 2000 | 2000 |

### 14.2 LJ13 (Lennard-Jones, 13 particles)

| Setting | SML | KSH |
|---------|-----|-----|
| Dimensions | 39 (13×3) | 39 (13×3) |
| NFE | 1000 | 1000 |
| Source | Harmonic (scale=2.0) | Harmonic (scale=2.0) |
| SDE | GraphVESDE | GraphVESDE |
| Model | EGNN (h=128, L=5) | EGNN (h=128, L=5) |
| Reference data | `data/test_split_LJ13-1000.npy` | `data/test_split_LJ13-1000.npy` |
| CV architecture | MLP (h=128, L=3) | SteinEGNN_LN (h=128, L=5) |
| CV training | 500 epochs | 5000 iterations |
| Score model | None | ScoreEGNN (h=128, L=4) |
| Eval samples | 2000 | 500 |

### 14.3 LJ55 (Lennard-Jones, 55 particles)

| Setting | SML | KSH |
|---------|-----|-----|
| Dimensions | 165 (55×3) | 165 (55×3) |
| Status | Planned (not yet run) | Config exists, not yet run |
| CV architecture | MLP (h=256, L=3) | TBD |
| Divergence | Hutchinson (1 probe) | TBD |

---

## 15. Summary Comparison Table

### 15.1 Methods Available

| Method | SML | KSH | Notes |
|--------|:---:|:---:|-------|
| KSD diagnostic | ✅ | ❌ | SML tracks distributional quality |
| RKHS Stein CV | ✅ | ❌ | Closed-form, $O(N^3)$, low-d only |
| Neural Stein CV (MLP) | ✅ | ❌ | Generic, scales to high-d |
| Neural Stein CV (EGNN) | ❌ | ✅ | Equivariant, particle-aware |
| Score-Informed CV | ❌ | ✅ | Pre-trained score + residual |
| Score Matching (ISM) | ❌ | ✅ | Learns $\nabla \log q_\theta$ |
| Generator Stein Kernel | ✅ | ❌ | Reuses SDE drift as score proxy |
| Antithetic Sampling | ✅ | ❌ | Paired trajectories, zero cost |
| MCMC Correction | ✅ | ❌ | Metropolis-Hastings post-hoc |
| Hybrid MCMC + Stein | ✅ | ❌ | Combined pipeline |

### 15.2 Design Trade-offs

| Dimension | SML | KSH |
|-----------|-----|-----|
| **Breadth** | 7 methods | 2–3 methods |
| **Depth** | Shallow per method | Deep per method |
| **Symmetry awareness** | None (MLP) | Full SE(3) equivariance |
| **Score knowledge** | Not used | Explicitly modeled |
| **Scalability strategy** | Switch from RKHS→Neural | EGNN + Hutchinson |
| **Validation** | No early stopping | Early stopping + val split |
| **Composability** | Methods combine freely | Methods are self-contained |
| **Applicability** | Any distribution | Particle systems |

### 15.3 Mathematical Foundations

| Theorem/Concept | SML | KSH |
|----------------|-----|-----|
| Stein's Identity | ✅ | ✅ |
| Bias-Variance Coupling | ✅ (explicit theorem) | Implicit |
| PDE characterization of optimal CV | ✅ (loss function) | ❌ |
| Variance minimization | ❌ (indirect via PDE) | ✅ (direct loss) |
| Score decomposition | ❌ | ✅ ($g = \alpha g_{\text{init}} + g_{\text{res}}$) |
| Generator-based Stein operator | ✅ (novel) | ❌ |

---

## 16. Theoretical Implications

### 16.1 SML's Bias-Variance Coupling Theorem

**Statement:** Under $q_\theta \neq \pi$, the Stein CV estimator satisfies:

$$|\text{Bias}| \leq \sqrt{C \cdot \mathrm{Var}_{q_\theta}[f + \mathcal{T}_\nu g]}$$

where $C = \sup_{\text{supp}(\pi)} \frac{\pi(x)}{q_\theta(x)}$.

**Implication:** Variance reduction automatically reduces bias. A $4\times$ variance reduction yields up to $2\times$ bias reduction (square root relationship).

### 16.2 KSH's Score-Informed Decomposition

**Insight:** If $q_\theta \approx \pi$, then:

$$\nabla \log q_\theta(x) + \nabla E(x) \approx 0$$

So $g_{\text{init}} = -(\nabla E + s_\phi) \approx 0$ when the sampler is good. The residual $g_{\text{res}}$ only needs to correct the remaining error, making optimization easier.

**Implication:** The score model provides a strong prior on the optimal CV direction, reducing the search space for $g_{\text{res}}$.

### 16.3 SML's Generator Kernel vs KSH's Score Model

Both repos try to go beyond the generic target score $-\nabla E$:

- **SML** uses the SDE drift $b_\theta(x,1)$ as a proxy for the "effective score" — no additional training needed
- **KSH** trains a separate score model $s_\phi \approx \nabla \log q_\theta$ — additional compute but more principled

The SDE drift $b_\theta$ and the score $\nabla \log q_\theta$ are related but not identical. The drift includes both the score-like component and the reference SDE dynamics.

---

*Generated: 2026-04-02 | Repos compared: adjoint_samplers (SML) vs KSH_ASBS (KSH)*
