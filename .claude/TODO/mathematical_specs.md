# Mathematical Specification: Enhancements to ASBS

## Variance Reduction and Diagnostics for Adjoint-Based Diffusion Samplers

-----

## 0. Baseline: What ASBS Produces

ASBS trains a controlled SDE:

$$
dX_t = \left[f(X_t, t) + g(t)^2,u_\theta(X_t, t)\right]dt + g(t),dW_t, \qquad X_0 \sim \mu
$$

where $f, g$ come from the reference SDE (e.g., VESDE: $f = 0$, $g(t) = \sigma(t)$) and $u_\theta$ is the learned controller.

After training, we generate $N$ terminal samples by simulating this SDE forward:

$$
X_1^{(1)}, X_1^{(2)}, \ldots, X_1^{(N)} \sim q_\theta
$$

where $q_\theta$ is the (intractable) terminal distribution of the SDE. The goal is $q_\theta \approx p$ where $p(x) \propto \exp(-E(x))$.

**What we have access to at each terminal sample $X_1^{(i)}$:**

- The sample itself: $X_1^{(i)} \in \mathbb{R}^d$
- The energy: $E(X_1^{(i)})$ (via `energy.eval()`)
- The energy gradient: $\nabla_x E(X_1^{(i)})$ (via `energy.grad_E()`)
- The score: $s_p(X_1^{(i)}) = -\nabla_x E(X_1^{(i)})$ (via `energy.score()`)
- The learned drift: $u_\theta(X_1^{(i)}, t=1)$ (via `controller(t, x)`)

**What we do NOT have access to:**

- The density $q_\theta(X_1^{(i)})$ — intractable for SDE-based models
- Importance weights $w_i = p(X_1^{(i)}) / q_\theta(X_1^{(i)})$

**Current evaluation:** Given terminal samples, the evaluator computes:

- Energy W2 distance against reference samples
- Interatomic distance W2
- Particle configuration W2

**The gap:** There is no mechanism for variance reduction of expectation estimates, no post-hoc correction, and no training-time distributional diagnostic beyond the Adjoint Matching loss.

-----

## 1. Enhancement 1: Kernel Stein Discrepancy (KSD) Diagnostic

### 1.1 Purpose

Provide a **training-time diagnostic** that directly measures how close the terminal distribution $q_\theta$ is to the target $p$, using only the score $s_p(x) = -\nabla_x E(x)$ evaluated at the terminal samples. No reference samples needed.

### 1.2 Mathematical Definition

The **Kernel Stein Discrepancy** between $q_\theta$ and $p$ is:

$$
\text{KSD}^2(q_\theta, p) = \mathbb{E}*{q*\theta \otimes q_\theta}[k_p(X, X’)]
$$

where $k_p$ is the **Stein kernel**, constructed from a base kernel $k: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ (e.g., RBF $k(x, x’) = \exp(-|x - x’|^2 / (2\ell^2))$) as:

$$
k_p(x, x’) = s_p(x)^T k(x, x’) s_p(x’) + s_p(x)^T \nabla_{x’} k(x, x’) + \nabla_x k(x, x’)^T s_p(x’) + \text{tr}(\nabla_x \nabla_{x’} k(x, x’))
$$

where $s_p(x) = -\nabla_x E(x)$.

### 1.3 Empirical Estimator

Given terminal samples ${X_1^{(i)}}*{i=1}^N \sim q*\theta$, the U-statistic estimator is:

$$
\widehat{\text{KSD}}^2 = \frac{1}{N(N-1)} \sum_{i \neq j} k_p(X_1^{(i)}, X_1^{(j)})
$$

### 1.4 Stein Kernel for RBF Base Kernel

For $k(x, x’) = \exp(-|x - x’|^2 / (2\ell^2))$, the derivatives are:

$$
\nabla_x k(x, x’) = -\frac{x - x’}{\ell^2} k(x, x’)
$$

$$
\nabla_{x’} k(x, x’) = \frac{x - x’}{\ell^2} k(x, x’)
$$

$$
\nabla_x \nabla_{x’}^T k(x, x’) = \frac{1}{\ell^2}\left(I_d - \frac{(x - x’)(x - x’)^T}{\ell^2}\right) k(x, x’)
$$

$$
\text{tr}(\nabla_x \nabla_{x’} k(x, x’)) = \frac{1}{\ell^2}\left(d - \frac{|x - x’|^2}{\ell^2}\right) k(x, x’)
$$

Substituting into the Stein kernel:

$$
k_p(x, x’) = k(x, x’) \left[ s_p(x)^T s_p(x’) + s_p(x)^T \frac{x - x’}{\ell^2} - \frac{x - x’}{\ell^2} \cdot s_p(x’)^T + \frac{1}{\ell^2}\left(d - \frac{|x - x’|^2}{\ell^2}\right) \right]
$$

Note: the second and third terms should use the correct signs from the gradient directions:

$$
k_p(x, x’) = k(x, x’) \left[ s_p(x)^T s_p(x’) + \frac{s_p(x)^T (x’ - x)}{\ell^2} + \frac{(x - x’)^T s_p(x’)}{\ell^2} + \frac{d}{\ell^2} - \frac{|x - x’|^2}{\ell^4} \right]
$$

### 1.5 Bandwidth Selection

Use the **median heuristic**:

$$
\ell = \text{median}\left({|X_1^{(i)} - X_1^{(j)}| : i < j}\right)
$$

### 1.6 Computational Cost

- Score evaluations: $N$ calls to `energy.score()` (one per sample, cached)
- Kernel matrix: $O(N^2 d)$ to compute all pairwise $k_p$
- Total: $O(N^2 d)$

-----

## 2. Enhancement 2: Stein Control Variates for Expectation Estimation

### 2.1 Purpose

Reduce the **variance** of expectation estimates $\hat{\mu} = \frac{1}{N}\sum_i f(X_1^{(i)})$ by subtracting an optimized zero-mean (under $p$) control variate.

### 2.2 The Stein Control Variate

For any $g: \mathbb{R}^d \to \mathbb{R}^d$, define:

$$
\mathcal{A}_p g(x) = s_p(x)^T g(x) + \nabla_x \cdot g(x)
$$

Stein’s identity: $\mathbb{E}_p[\mathcal{A}_p g(X)] = 0$. So the augmented estimator:

$$
\hat{\mu}^{\text{SCV}} = \frac{1}{N}\sum_{i=1}^N \left[f(X_1^{(i)}) - \sum_{j=1}^N a_j, k_p(X_1^{(j)}, X_1^{(i)})\right]
$$

has the same expectation as $\hat{\mu}$ under $p$ (up to the bias from $q_\theta \neq p$) but potentially much lower variance.

### 2.3 Optimal Coefficients

The coefficients $a = (a_1, \ldots, a_N)^T$ are found by solving:

$$
(K_p + \lambda N I_N), a = \mathbf{f}
$$

where:

- $K_p \in \mathbb{R}^{N \times N}$ with $(K_p)_{ij} = k_p(X_1^{(i)}, X_1^{(j)})$ is the Stein kernel matrix
- $\mathbf{f} = (f(X_1^{(1)}), \ldots, f(X_1^{(N)}))^T$ is the vector of function evaluations
- $\lambda > 0$ is a regularization parameter

### 2.4 The Corrected Estimator

$$
\hat{\mu}^{\text{SCV}} = \mathbf{1}^T a = \mathbf{1}^T (K_p + \lambda N I_N)^{-1} \mathbf{f}
$$

### 2.5 Choice of $f$ for ASBS Benchmarks

For the existing ASBS benchmarks, natural choices of $f$ are:

- $f(x) = E(x)$: mean energy (most fundamental thermodynamic observable)
- $f(x) = |x_i - x_j|$: interatomic distances
- $f(x) = x_k$: individual coordinate moments

### 2.6 Computational Cost

- Same as KSD ($O(N^2 d)$ for kernel matrix) plus $O(N^3)$ for the linear solve
- For large $N$: use Nyström approximation with $m \ll N$ inducing points, reducing to $O(N m^2)$

### 2.7 Bias Consideration

When $q_\theta \neq p$, the Stein CV introduces additional bias bounded by:

$$
|\text{additional bias}| \leq \sqrt{D_F(q_\theta | p)} \cdot |g|*{L^2(q*\theta)}
$$

where $D_F$ is the Fisher divergence. For well-trained ASBS this is small.

-----

## 3. Enhancement 3: Antithetic SDE Sampling

### 3.1 Purpose

Variance reduction via negative correlation between paired trajectories, at **zero additional energy evaluations**.

### 3.2 The Antithetic Pair

Given a forward trajectory driven by Brownian increments $\Delta W_i$:

$$
X_{t_{i+1}} = X_{t_i} + b_\theta(X_{t_i}, t_i),\Delta t + g(t_i),\sqrt{\Delta t},\xi_i, \qquad \xi_i \sim \mathcal{N}(0, I_d)
$$

The **antithetic trajectory** uses negated noise:

$$
X’*{t*{i+1}} = X’*{t_i} + b*\theta(X’_{t_i}, t_i),\Delta t - g(t_i),\sqrt{\Delta t},\xi_i
$$

Both start from the same $X_0 \sim \mu$ and use the same drift evaluations $b_\theta$, but opposite noise realizations $\xi_i$ and $-\xi_i$.

### 3.3 The Antithetic Estimator

For any function $f$:

$$
\hat{\mu}^{\text{anti}} = \frac{1}{2N}\sum_{i=1}^N \left[f(X_1^{(i)}) + f(X_1’^{(i)})\right]
$$

**Variance:**

$$
\text{Var}[\hat{\mu}^{\text{anti}}] = \frac{1}{4N}\left[\text{Var}[f(X_1)] + \text{Var}[f(X_1’)] + 2\text{Cov}[f(X_1), f(X_1’)]\right]
$$

If the covariance is negative (antithetic correlation), the variance is reduced compared to $\text{Var}[\hat{\mu}] = \frac{1}{N}\text{Var}[f(X_1)]$.

### 3.4 When Antithetic Correlation Holds

Antithetic correlation ($\text{Cov}[f(X_1), f(X_1’)] < 0$) occurs when:

- The learned drift $b_\theta$ dominates over the noise (well-trained model)
- $f$ is monotone in at least some directions
- The SDE dynamics are not too chaotic

For nearly deterministic transport (strong drift, weak noise), $X_1 \approx X_1’$ and there is positive correlation (no benefit). For moderate noise, $X_1$ and $X_1’$ tend to land on opposite sides of the distribution, giving negative correlation.

### 3.5 Cost

**Zero additional energy evaluations.** The antithetic trajectory reuses the same noise samples (negated) and requires one additional forward pass through the drift network. The cost is approximately $1 \times$ the cost of standard sampling (the drift evaluations are at different $x$ values, so they can’t be cached, but no energy calls are needed).

-----

## 4. Enhancement 4: MCMC Post-Correction

### 4.1 Purpose

Remove bias from ASBS terminal samples by running a short Metropolis-Hastings chain, producing asymptotically exact samples from $p$.

### 4.2 Algorithm

Given terminal ASBS samples ${X_1^{(i)}}_{i=1}^N$, for each sample independently:

```
X = X_1^{(i)}
for k = 1, ..., K:
    X' = X + σ * ξ,  ξ ~ N(0, I_d)
    ΔE = E(X') - E(X)
    α = min(1, exp(-ΔE))
    with probability α: X = X'
X_corrected^{(i)} = X
```

### 4.3 Properties

- The MH acceptance ratio uses only **energy differences** $\Delta E = E(X’) - E(X)$. No $q_\theta$ needed.
- ASBS provides excellent initializations (close to $p$), so the MH chain mixes quickly — few steps ($K = 5$–$20$) suffice.
- After correction, samples are asymptotically distributed as $p$, regardless of $q_\theta$’s quality.
- Samples become correlated if $K$ is too small, reducing effective sample size.

### 4.4 Step Size Selection

For random-walk MH in $d$ dimensions, the optimal step size for Gaussian targets is:

$$
\sigma = \frac{2.38}{\sqrt{d}} \cdot \hat{\sigma}_{\text{marginal}}
$$

where $\hat{\sigma}_{\text{marginal}}$ is the estimated marginal standard deviation of the terminal samples. Target acceptance rate: $\approx 0.234$.

### 4.5 Combining with Stein CVs

The full pipeline:

$$
\text{ASBS samples} \xrightarrow{K \text{ MH steps}} \text{corrected samples} \sim p \xrightarrow{\text{Stein CV}} \text{low-variance estimate}
$$

After MCMC correction, $\mathbb{E}_p[\mathcal{A}_p g(X)] = 0$ holds exactly, so the Stein CV introduces no additional bias.

### 4.6 Cost

- $K$ energy evaluations per sample per MH step (one for $E(X’)$; $E(X)$ is cached from previous step)
- Total: $K \times N$ energy evaluations
- For particle systems where energy evaluation is cheap (DW4, LJ13), this is negligible

-----

## 5. Enhancement 5: KSD Training Regularizer

### 5.1 Purpose

Add a KSD term to the ASBS training loss to directly penalize terminal distributional mismatch during training.

### 5.2 Modified Training Loss

The standard Adjoint Matching loss at each training step is:

$$
\mathcal{L}*{\text{AM}}(\theta) = \frac{1}{B}\sum*{i=1}^B |u_\theta(t_i, x_i) - (-Y_{t_i}^{(i)})|^2
$$

where $B$ is the batch size and $Y_t$ is the adjoint state.

The modified loss adds a KSD regularizer computed on the terminal samples from the most recent buffer:

$$
\mathcal{L}*{\text{total}}(\theta) = \mathcal{L}*{\text{AM}}(\theta) + \lambda_{\text{KSD}} \cdot \widehat{\text{KSD}}^2({X_1^{(j)}}_{j=1}^M, p)
$$

where ${X_1^{(j)}}_{j=1}^M$ are terminal samples from the current buffer (not differentiated through — the KSD is computed on detached samples).

### 5.3 Why Not Differentiate Through KSD

KSD depends on $s_p(X_1) = -\nabla_x E(X_1)$ and on $X_1$ itself. Differentiating through $X_1$ would require backpropagating through the entire SDE trajectory (backpropagation through time), which is exactly what ASBS avoids by design (the Adjoint Matching objective is specifically constructed to avoid this).

Instead, we use KSD as a **detached regularizer**: it provides a training signal about the quality of the terminal distribution, but the gradient flows only through the AM loss. The KSD value is logged as a diagnostic.

### 5.4 Alternative: KSD on Buffer Refresh

Instead of adding KSD to the loss, compute KSD every time the buffer is refreshed (during `populate_buffer`) and log it. If KSD increases, it indicates the model is diverging from $p$ — useful for early stopping or learning rate adjustment.

-----

## 6. Enhancement 6: SDE Generator Stein Operator

### 6.1 Purpose

Construct a **better control variate** using the learned SDE dynamics, rather than the generic Stein operator.

### 6.2 The SDE Generator

The infinitesimal generator of the controlled SDE at time $t = 1$ is:

$$
\mathcal{L}*{u*\theta} g(x) = b_\theta(x, 1)^T \nabla g(x) + \frac{g(1)^2}{2} \Delta g(x)
$$

where $b_\theta(x, 1) = f(x, 1) + g(1)^2 u_\theta(x, 1)$ is the total drift at $t = 1$.

If the SDE has converged (terminal distribution is $p$), then:

$$
\mathbb{E}*p[\mathcal{L}*{u_\theta} g(X)] = 0 \quad \text{for all suitable } g
$$

### 6.3 Why This Is Better Than the Standard Stein Operator

The standard Stein operator $\mathcal{A}_p g = s_p^T g + \nabla \cdot g$ corresponds to the generator of the **Langevin diffusion** $dX_t = s_p(X_t)dt + \sqrt{2},dW_t$, which has $p$ as stationary distribution.

The SDE generator $\mathcal{L}*{u*\theta}$ corresponds to the **learned dynamics** that actually produced the samples. If the learned drift $b_\theta$ captures the structure of $p$ better than the generic Langevin drift $s_p$, then $\mathcal{L}*{u*\theta} g$ will correlate better with $f$ and give better variance reduction.

### 6.4 The Generator Stein Kernel

Construct a Stein kernel from the generator:

$$
k_{u_\theta}(x, x’) = \mathcal{L}*{u*\theta}^x \mathcal{L}*{u*\theta}^{x’} k(x, x’)
$$

where $\mathcal{L}*{u*\theta}^x$ applies the generator in the $x$ variable. This involves:

- $b_\theta(x, 1) = f(x, 1) + g(1)^2 u_\theta(x, 1)$: one forward pass through the controller network
- First and second derivatives of $k$: closed-form for RBF
- First derivatives of $b_\theta$: one Jacobian-vector product (autodiff)

### 6.5 Practical Simplification

Computing the full generator kernel requires Jacobians of $b_\theta$, which is expensive. A practical simplification: use the **diffusion-free** part only:

$$
\tilde{k}*{u*\theta}(x, x’) = b_\theta(x, 1)^T k(x, x’) b_\theta(x’, 1) + b_\theta(x, 1)^T \nabla_{x’} k(x, x’) + \nabla_x k(x, x’)^T b_\theta(x’, 1) + \frac{g(1)^2}{2} \text{tr}(\nabla_x \nabla_{x’} k(x, x’))
$$

This replaces $s_p(x)$ with $b_\theta(x, 1)$ in the standard Stein kernel formula, plus the diffusion coefficient $g(1)^2/2$ in the trace term. This avoids computing Jacobians of $b_\theta$ while still incorporating the learned dynamics.

### 6.6 Cost

Same as standard KSD ($O(N^2 d)$), plus $N$ forward passes through the controller network to get $b_\theta(X_1^{(i)}, 1)$ (already available from the SDE simulation — it’s the drift at the last timestep).

-----

## 7. Experimental Design

### 7.1 Benchmarks

All experiments run on the three existing ASBS benchmarks:

- **DW4**: Double Well, 4 particles × 3D = 12 dimensions. Fast iteration.
- **LJ13**: Lennard-Jones, 13 particles × 3D = 39 dimensions. Medium.
- **LJ55**: Lennard-Jones, 55 particles × 3D = 165 dimensions. Tests high-$d$ scaling.

### 7.2 Metrics

For each enhancement, we measure:

**Existing metrics** (for consistency with ASBS paper):

- Energy W2 distance to reference samples
- Interatomic distance W2
- Particle configuration W2 (eq_w2)

**New metrics:**

- $\widehat{\text{KSD}}^2$: Stein discrepancy (lower = closer to $p$)
- Estimated $\mathbb{E}_p[E(X)]$ (mean energy): compare vanilla vs. Stein CV vs. MCMC+Stein CV against ground truth
- Variance of energy estimator across multiple runs
- Effective sample size (for MCMC-corrected samples)
- MH acceptance rate (for MCMC correction)

### 7.3 Comparisons

For each benchmark:

|Method                         |What it tests                          |
|-------------------------------|---------------------------------------|
|Vanilla ASBS                   |Baseline                               |
|+ KSD diagnostic               |Does KSD track training progress?      |
|+ Antithetic sampling          |Free variance reduction?               |
|+ MCMC correction ($K=5,10,20$)|Bias removal effectiveness?            |
|+ Stein CV (on vanilla samples)|Variance reduction with bias?          |
|+ MCMC + Stein CV              |Full pipeline: unbiased + low variance?|
|+ Generator Stein CV           |Does learned drift give better CVs?    |

### 7.4 Ablations

- **KSD bandwidth $\ell$**: median heuristic vs. fixed values
- **Stein CV regularization $\lambda$**: $10^{-6}, 10^{-4}, 10^{-2}$
- **MCMC steps $K$**: $1, 5, 10, 20, 50$
- **MCMC step size $\sigma$**: optimal vs. fixed
- **Number of samples $N$**: $100, 500, 1000, 5000$ (scaling behavior)

-----

## 8. Notation Reference

|Symbol                   |Meaning                                     |Codebase equivalent                             |
|-------------------------|--------------------------------------------|------------------------------------------------|
|$E(x)$                   |Energy function                             |`energy.eval(x)`                                |
|$\nabla_x E(x)$          |Energy gradient                             |`energy.grad_E(x)`                              |
|$s_p(x) = -\nabla_x E(x)$|Score of target                             |`energy.score(x)`                               |
|$u_\theta(t, x)$         |Learned controller                          |`controller(t, x)`                              |
|$b_\theta(t, x)$         |Total drift: $f(t,x) + g(t)^2 u_\theta(t,x)$|`sde.drift(t, x)`                               |
|$g(t)$                   |Diffusion coefficient                       |`sde.diff(t)`                                   |
|$X_0$                    |Source sample                               |`source.sample([B,])`                           |
|$X_1$                    |Terminal sample                             |Output of `sdeint(sde, x0, timesteps)`          |
|$k(x, x’)$               |Base kernel (RBF)                           |New: `rbf_kernel(x, x', ell)`                   |
|$k_p(x, x’)$             |Stein kernel                                |New: `stein_kernel(x, x', scores, ell)`         |
|$K_p$                    |Stein kernel matrix                         |New: `stein_kernel_matrix(samples, scores, ell)`|
|$\widehat{\text{KSD}}^2$ |Empirical KSD                               |New: `compute_ksd(samples, scores, ell)`        |