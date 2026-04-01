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

Reduce the **variance** of expectation estimates $\hat{\mu} = \frac{1}{N}\sum_i f(X_1^{(i)})$ by subtracting an optimized zero-mean (under $p$) control variate. As we show below, this variance reduction **simultaneously reduces bias** when samples come from $q_\theta \neq p$.

### 2.2 The Stein Control Variate

For any $g: \mathbb{R}^d \to \mathbb{R}^d$, define the augmented function:

$$
h_g(x) = f(x) + \mathcal{A}_p g(x) = f(x) + s_p(x)^T g(x) + \nabla_x \cdot g(x)
$$

**Under $p$**: $\mathbb{E}_p[h_g(X)] = \mathbb{E}_p[f(X)]$ for any $g$ (Stein’s identity).

**Under $q_\theta$**: $\mathbb{E}*{q*\theta}[h_g(X)] \neq \mathbb{E}_p[f(X)]$ in general. However, the bias is controlled by the variance, as we show in Section 2.7.

The estimator using samples $X_1^{(i)} \sim q_\theta$ is:

$$
\hat{\mu}^{\text{SCV}} = \frac{1}{N}\sum_{i=1}^N h_g(X_1^{(i)}) = \frac{1}{N}\sum_{i=1}^N \left[f(X_1^{(i)}) - \sum_{j=1}^N a_j, k_p(X_1^{(j)}, X_1^{(i)})\right]
$$

where the control variate is represented in the RKHS as $\mathcal{A}_p g(\cdot) = -\sum_j a_j k_p(X_1^{(j)}, \cdot)$.

### 2.3 Optimal Coefficients

The coefficients $a = (a_1, \ldots, a_N)^T$ are found by solving:

$$
(K_p + \lambda N I_N), a = \mathbf{f}
$$

where:

- $K_p \in \mathbb{R}^{N \times N}$ with $(K_p)_{ij} = k_p(X_1^{(i)}, X_1^{(j)})$ is the Stein kernel matrix
- $\mathbf{f} = (f(X_1^{(1)}), \ldots, f(X_1^{(N)}))^T$ is the vector of function evaluations
- $\lambda > 0$ is a regularization parameter

This minimizes the empirical variance of the corrected values $h_g(X_1^{(i)})$ within the RKHS function class.

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

### 2.7 The Bias-Variance Coupling (Key Theoretical Result)

This section shows that **optimizing $g$ for variance reduction under $q_\theta$ automatically reduces the bias**, even though $q_\theta \neq p$.

**Setup**: Define the augmented function $h(x) = f(x) + \mathcal{A}*p g(x)$. The estimator computes $\frac{1}{N}\sum_i h(X_i)$ with $X_i \sim q*\theta$, estimating $\mathbb{E}*{q*\theta}[h(X)]$. The bias is:

$$
\text{Bias} = \mathbb{E}*{q*\theta}[h(X)] - \mathbb{E}_p[f(X)]
$$

Since $\mathbb{E}_p[h(X)] = \mathbb{E}_p[f(X)]$ (Stein’s identity), this equals:

$$
\text{Bias} = \mathbb{E}*{q*\theta}[h(X)] - \mathbb{E}_p[h(X)]
$$

**Step 1 — Decompose $h$ into constant + residual**: Let $c = \mathbb{E}*{q*\theta}[h(X)]$ and $r(x) = h(x) - c$. Then $\mathbb{E}*{q*\theta}[r(X)] = 0$ by construction, so:

$$
|\text{Bias}| = |c - \mathbb{E}*p[h(X)]| = |\mathbb{E}*{q_\theta}[h(X)] - \mathbb{E}_p[h(X)]| = |\mathbb{E}_p[r(X)]|
$$

since $\mathbb{E}*{q*\theta}[r] = 0$ by definition.

**Step 2 — Bound using Cauchy-Schwarz**:

$$
|\mathbb{E}_p[r(X)]| \leq \sqrt{\mathbb{E}_p[r(X)^2]}
$$

**Step 3 — Relate $\mathbb{E}*p[r^2]$ to $\text{Var}*{q_\theta}[h]$**: If $\text{supp}(p) \subseteq \text{supp}(q_\theta)$ and the density ratio is bounded, $p(x)/q_\theta(x) \leq C$ on $\text{supp}(p)$, then:

$$
\mathbb{E}*p[r^2] = \int r(x)^2 p(x),dx = \int r(x)^2 \frac{p(x)}{q*\theta(x)} q_\theta(x),dx \leq C \cdot \mathbb{E}*{q*\theta}[r^2] = C \cdot \text{Var}*{q*\theta}[h]
$$

**The bound:**

$$
\boxed{|\text{Bias}| \leq \sqrt{C \cdot \text{Var}*{q*\theta}[f(X) + \mathcal{A}_p g(X)]}}
$$

where $C = \sup_{x \in \text{supp}(p)} p(x)/q_\theta(x)$ is the density ratio bound.

### 2.8 Consequences of the Bias-Variance Coupling

**Consequence 1**: Variance reduction is **not just variance reduction** — it is simultaneous bias reduction. The bias shrinks as the square root of the variance. Optimizing $g$ within any function class (RKHS, neural network, polynomial) to minimize $\text{Var}*{q*\theta}[h_g]$ automatically reduces both the variance and the bias.

**Consequence 2**: In the **zero-variance limit**, if there exists $g^*$ such that $\text{Var}*{q*\theta}[f + \mathcal{A}_p g^*] = 0$, then $h_{g^*}(x) = c$ is constant and:

- $c = \mathbb{E}_p[f(X)]$ (from Stein’s identity under $p$)
- The estimator returns $c$ at every sample, regardless of which distribution the samples come from
- Both bias and variance are exactly zero

The $g^*$ achieving this solves the **Poisson equation** $\mathcal{A}_p g^*(x) = \mathbb{E}_p[f(X)] - f(x)$, which is intractable (requires the unknown $\mathbb{E}_p[f(X)]$). But the RKHS optimization in Section 2.3 **approximates** this solution, and the closer the approximation, the closer both variance and bias are to zero.

**Consequence 3**: The implementation in `stein_cv.py` (Section 2.3–2.4) **already performs bias reduction implicitly**. The coefficients $a$ minimize the empirical variance of the corrected values $h(X_i)$, which by the bound above, simultaneously reduces the bias. No separate bias-correction mechanism is needed.

**Consequence 4**: This makes Stein CVs **more powerful than previously framed** for the ASBS setting. Even without MCMC correction (which removes bias by a separate mechanism), the Stein CV alone can substantially reduce bias — the extent determined by how well the RKHS can approximate the Poisson equation solution.

**Consequence 5**: The **hybrid pipeline** (MCMC correction + Stein CV) remains valuable but for a different reason than originally stated. MCMC correction makes the samples exactly from $p$ (in the limit), so Stein’s identity holds exactly and the CV provides pure variance reduction with no bias contamination. Without MCMC correction, the Stein CV provides variance reduction with proportional bias reduction — still good, but with the density ratio constant $C$ entering the bound.

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

### 4.5 Combining with Stein CVs — Two Regimes

Given the bias-variance coupling result from Section 2.7, there are now **two distinct regimes** for combining MCMC with Stein CVs:

**Regime 1: Stein CV alone (no MCMC)**

$$
\text{ASBS samples} \sim q_\theta \xrightarrow{\text{Stein CV}} \text{estimate with reduced variance AND bias}
$$

The bias bound from Section 2.7 applies: $|\text{Bias}| \leq \sqrt{C \cdot \text{Var}*{q*\theta}[h_g]}$. As $g$ is optimized, both variance and bias shrink. No MCMC needed. This is cheaper (no energy evaluations for MH steps) but the bias reduction depends on how well the RKHS can approximate the Poisson equation.

**Regime 2: MCMC + Stein CV (the full pipeline)**

$$
\text{ASBS samples} \xrightarrow{K \text{ MH steps}} \text{corrected samples} \sim p \xrightarrow{\text{Stein CV}} \text{low-variance unbiased estimate}
$$

After MCMC correction, $\mathbb{E}_p[\mathcal{A}_p g(X)] = 0$ holds exactly. The Stein CV provides **pure variance reduction** with zero bias contamination. The density ratio constant $C$ drops out entirely.

**Which regime to use depends on the application:**

- If energy evaluations are cheap and you need exact results: Regime 2
- If energy evaluations are expensive and you can tolerate approximate results: Regime 1
- The experiments should compare both and quantify the tradeoff

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

## 7. Enhancement 7: Neural Stein CV via Differentiated Poisson Equation

### 7.1 Purpose

Replace the RKHS-based Stein CV (Enhancement 2) with a **neural network** $g_\phi: \mathbb{R}^d \to \mathbb{R}^d$ that directly solves for the optimal control variate via the Poisson equation. This approach:

1. **Scales to high dimensions** — no $N \times N$ kernel matrix, cost is per-sample
1. **Breaks the circularity** — eliminates the unknown constant $\mathbb{E}_p[f]$ by differentiating
1. **Is more expressive** — neural networks can represent richer control variates than RKHS with RBF kernel
1. **Is reusable** — once trained, applies to any set of samples from the same target

### 7.2 The Circularity Problem

The zero-variance condition $f(x) + \mathcal{A}_p g(x) = c$ is the **Poisson equation**:

$$
\mathcal{A}_p g(x) = c - f(x) \qquad \text{where } c = \mathbb{E}_p[f(X)]
$$

We cannot solve this because $c$ is the unknown quantity we’re trying to estimate.

### 7.3 Breaking the Circularity by Differentiation

**Key insight**: If $f(x) + \mathcal{A}_p g(x) = c$ (constant), then its gradient is zero:

$$
\nabla_x!\left(f(x) + \mathcal{A}_p g(x)\right) = 0
$$

The unknown constant $c$ **vanishes** upon differentiation. Expanding $\mathcal{A}_p g(x) = s_p(x)^T g(x) + \nabla \cdot g(x) = (\partial_j \log p),g_j + \partial_j g_j$ (Einstein summation over $j$), and applying $\partial_i$:

$$
\partial_i(\partial_j g_j) + \partial_i!\left((\partial_j \log p),g_j\right) = -\partial_i f
$$

Expanding the product rule on the second term:

$$
\partial_i \partial_j g_j + (\partial_i g_j)(\partial_j \log p) + (\partial_i \partial_j \log p),g_j = -\partial_i f
$$

This is a **second-order PDE for $g$** that does not involve the unknown $c$.

### 7.4 The PDE in Vector Form

Writing $s = \nabla \log p = -\nabla E$ (the score) and $H = \nabla^2 \log p = -\nabla^2 E$ (the Hessian of log-density), the PDE becomes:

$$
\nabla(\nabla \cdot g) + (\nabla g),s + H,g = -\nabla f
$$

Component by component (for $i = 1, \ldots, d$):

$$
\sum_{j=1}^d \frac{\partial^2 g_j}{\partial x_i \partial x_j} + \sum_{j=1}^d \frac{\partial g_j}{\partial x_i},s_j(x) + \sum_{j=1}^d H_{ij}(x),g_j(x) = -\frac{\partial f}{\partial x_i}(x)
$$

### 7.5 Neural Network Training Objective

Parameterize $g_\phi: \mathbb{R}^d \to \mathbb{R}^d$ as a neural network. Train by minimizing the **PDE residual** at sample locations:

$$
\mathcal{L}*{\text{PDE}}(\phi) = \mathbb{E}*{q_\theta}!\left[\left|R_\phi(X)\right|^2\right]
$$

where the residual vector $R_\phi(x) \in \mathbb{R}^d$ has components:

$$
R_{\phi,i}(x) = \sum_j \frac{\partial^2 g_{\phi,j}}{\partial x_i \partial x_j}(x) + \sum_j \frac{\partial g_{\phi,j}}{\partial x_i}(x),s_j(x) + \sum_j H_{ij}(x),g_{\phi,j}(x) + \frac{\partial f}{\partial x_i}(x)
$$

This is estimated via mini-batch SGD over the terminal ASBS samples.

### 7.6 Computing Each Term via Autodiff

**Term 1**: $\sum_j \partial_i \partial_j g_{\phi,j}$ — the $i$-th component of $\nabla(\nabla \cdot g_\phi)$.

- First compute $\text{div}(g_\phi)(x) = \sum_j \partial_j g_{\phi,j}(x)$ via `torch.autograd.grad` with `create_graph=True`
- Then compute $\nabla_x [\text{div}(g_\phi)]$ — the gradient of the divergence
- Cost: $O(d)$ backward passes for the divergence (or Hutchinson trace estimator), then one more backward for the gradient
- **Hutchinson approximation**: $\text{div}(g_\phi)(x) \approx v^T J_{g_\phi}(x) v$ where $v \sim \mathcal{N}(0, I_d)$ and $J_{g_\phi}$ is the Jacobian. This requires only one Jacobian-vector product.

**Term 2**: $(\nabla g_\phi) s = J_{g_\phi}^T s$ — the Jacobian of $g_\phi$ times the score.

- Compute as $d$ Jacobian-vector products: for each $i$, $\sum_j (\partial_i g_{\phi,j}) s_j = \nabla_x(g_\phi(x) \cdot s(x))*i - g*\phi(x) \cdot (\partial_i s(x))$
- **Efficient alternative**: Compute $\nabla_x [s(x)^T g_\phi(x)]$ in one backward pass. This gives $J_{g_\phi}^T s + J_s^T g_\phi = J_{g_\phi}^T s + H,g_\phi$. This combines Terms 2 and 3.

**Term 3**: $H,g_\phi$ — the Hessian of $\log p$ times $g_\phi$.

- $H_{ij} = \partial_i \partial_j \log p = -\partial_i \partial_j E$
- Computing $Hg$ is one Hessian-vector product: `torch.autograd.functional.hvp` or by differentiating $s(x)^T g_\phi(x)$
- As noted above, Terms 2+3 together = $\nabla_x[s(x)^T g_\phi(x)]$, which is a single backward pass.

**Term 4**: $\nabla f$ — the gradient of the function of interest.

- For $f(x) = E(x)$: $\nabla f = \nabla E = -s(x)$ — already available.
- For other $f$: one backward pass through $f$.

### 7.7 Efficient Combined Computation

The key efficiency trick: **combine Terms 2 and 3** into one backward pass.

$$
\text{Term 2} + \text{Term 3} = J_{g_\phi}^T s + H g_\phi = \nabla_x!\left[s(x)^T g_\phi(x)\right]
$$

**Proof**: By the product rule, $\partial_i(s^T g_\phi) = \sum_j (\partial_i s_j) g_{\phi,j} + \sum_j s_j (\partial_i g_{\phi,j}) = (H g_\phi)*i + (J*{g_\phi}^T s)_i$.

So the full residual becomes:

$$
R_{\phi,i} = \partial_i(\text{div},g_\phi) + \partial_i(s^T g_\phi) + \partial_i f = \partial_i!\left[\text{div}(g_\phi) + s^T g_\phi + f\right]
$$

But $\text{div}(g_\phi) + s^T g_\phi = \mathcal{A}*p g*\phi$! So:

$$
R_\phi = \nabla_x!\left[\mathcal{A}*p g*\phi(x) + f(x)\right]
$$

This is just the gradient of $h(x) = f(x) + \mathcal{A}*p g*\phi(x)$. The PDE loss is:

$$
\boxed{\mathcal{L}*{\text{PDE}}(\phi) = \mathbb{E}*{q_\theta}!\left[\left|\nabla_x!\left[f(x) + \mathcal{A}*p g*\phi(x)\right]\right|^2\right]}
$$

**This is strikingly simple**: minimize the squared gradient norm of the augmented function $h = f + \mathcal{A}*p g*\phi$. When $|\nabla h| = 0$ everywhere, $h$ is constant, which is exactly the zero-variance condition.

### 7.8 Practical Training Algorithm

```
Input: terminal samples {X_i} from ASBS, energy E, function f
Output: trained neural CV g_phi

1. Initialize g_phi (small random weights, output dimension d)
2. For each training iteration:
   a. Sample mini-batch {X_1, ..., X_B} from the terminal samples
   b. Compute s(X_i) = -grad_E(X_i)   [score, cached if available]
   c. Forward: g = g_phi(X_i)          [neural net output, (B, d)]
   d. Compute A_p g(X_i) = s^T g + div(g)  [Stein operator applied to g]
      - div(g) via Hutchinson: v^T J_g v, v ~ N(0,I)
      - or exact: sum of diagonal Jacobian entries
   e. Compute h(X_i) = f(X_i) + A_p g(X_i)  [augmented function]
   f. Compute grad_h = grad_x(h)       [gradient of h w.r.t. x, one backward]
   g. Loss = mean(||grad_h||^2)        [PDE residual]
   h. Backprop through phi, optimizer step

3. After training:
   - Compute h(X_i) = f(X_i) + A_p g_phi(X_i) for all samples
   - Estimate = mean(h(X_i))
   - Variance = var(h(X_i)) / N
```

### 7.9 Architecture for $g_\phi$

A simple MLP works: $g_\phi: \mathbb{R}^d \to \mathbb{R}^d$ with a few hidden layers. Requirements:

- **Twice differentiable** activation functions (no ReLU — use SiLU, GELU, or Tanh) because the PDE involves second derivatives of $g_\phi$
- **Output dimension = input dimension** ($d$)
- Moderate size: 3–4 layers, 128–256 hidden units

For particle systems (DW4, LJ13, LJ55), an **equivariant** architecture matching the EGNN used by ASBS would be ideal but is not necessary for a first implementation.

### 7.10 Computational Cost

|Operation                                    |Cost per sample                   |
|---------------------------------------------|----------------------------------|
|$g_\phi(x)$ forward pass                     |$O(\text{net size})$              |
|$\text{div}(g_\phi)$ (Hutchinson, $k$ probes)|$k$ Jacobian-vector products      |
|$s(x)^T g_\phi(x)$                           |$O(d)$ dot product                |
|$\nabla_x h(x)$                              |1 backward pass                   |
|Total per iteration                          |$O(B \cdot (\text{net size} + d))$|

No $N \times N$ matrices. Training scales linearly in $N$ (via mini-batching). This is the critical advantage over the RKHS approach.

### 7.11 Comparison: RKHS vs. Neural Stein CV

|Property                 |RKHS (Enhancement 2)                  |Neural (Enhancement 7)              |
|-------------------------|--------------------------------------|------------------------------------|
|Function class           |Stein RKHS (kernel-based)             |Neural network                      |
|Compute cost             |$O(N^3 + N^2 d)$                      |$O(\text{epochs} \times B \times d)$|
|Memory                   |$O(N^2)$ (kernel matrix)              |$O(\text{net params})$              |
|Scalability to high $d$  |Poor (kernel degrades)                |Good                                |
|Scalability to large $N$ |Poor ($N^3$ solve)                    |Good (mini-batch)                   |
|Circularity              |Avoided via variance minimization     |Broken via differentiation          |
|Reusability              |Tied to sample set                    |Reusable across sample sets         |
|Theoretical guarantees   |Convergence rates known (Oates et al.)|Less well-characterized             |
|Implementation complexity|Simple (linear algebra)               |Moderate (autodiff through PDE)     |
|Best for                 |Low-$d$, small $N$, quick experiments |High-$d$, large $N$, production     |

### 7.12 When to Use Which

- **DW4 (12D)**: Either works. RKHS is simpler and fast enough.
- **LJ13 (39D)**: RKHS starts to struggle with kernel quality. Neural may outperform.
- **LJ55 (165D)**: Neural is strongly preferred. RKHS kernel becomes nearly constant at this dimensionality.

-----

## 8. Experimental Design

### 8.1 Benchmarks

All experiments run on the three existing ASBS benchmarks:

- **DW4**: Double Well, 4 particles × 3D = 12 dimensions. Fast iteration.
- **LJ13**: Lennard-Jones, 13 particles × 3D = 39 dimensions. Medium.
- **LJ55**: Lennard-Jones, 55 particles × 3D = 165 dimensions. Tests high-$d$ scaling.

### 8.2 Metrics

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

### 8.3 Comparisons

For each benchmark:

|Method                         |What it tests                          |
|-------------------------------|---------------------------------------|
|Vanilla ASBS                   |Baseline                               |
|+ KSD diagnostic               |Does KSD track training progress?      |
|+ Antithetic sampling          |Free variance reduction?               |
|+ MCMC correction ($K=5,10,20$)|Bias removal effectiveness?            |
|+ Stein CV (on vanilla samples)|Variance + bias reduction via coupling?|
|+ MCMC + Stein CV              |Full pipeline: unbiased + low variance?|
|+ Generator Stein CV           |Does learned drift give better CVs?    |
|+ Neural Stein CV              |Scales to high $d$? Better than RKHS?  |

### 8.4 Ablations

- **KSD bandwidth $\ell$**: median heuristic vs. fixed values
- **Stein CV regularization $\lambda$**: $10^{-6}, 10^{-4}, 10^{-2}$
- **MCMC steps $K$**: $1, 5, 10, 20, 50$
- **MCMC step size $\sigma$**: optimal vs. fixed
- **Number of samples $N$**: $100, 500, 1000, 5000$ (scaling behavior)

-----

## 9. Notation Reference

|Symbol                                      |Meaning                                                              |Codebase equivalent                             |
|--------------------------------------------|---------------------------------------------------------------------|------------------------------------------------|
|$E(x)$                                      |Energy function                                                      |`energy.eval(x)`                                |
|$\nabla_x E(x)$                             |Energy gradient                                                      |`energy.grad_E(x)`                              |
|$s_p(x) = -\nabla_x E(x)$                   |Score of target                                                      |`energy.score(x)`                               |
|$H(x) = \nabla^2 \log p(x) = -\nabla^2 E(x)$|Hessian of log-density                                               |autodiff through `energy.score`                 |
|$u_\theta(t, x)$                            |Learned controller                                                   |`controller(t, x)`                              |
|$b_\theta(t, x)$                            |Total drift: $f(t,x) + g(t)^2 u_\theta(t,x)$                         |`sde.drift(t, x)`                               |
|$g(t)$                                      |Diffusion coefficient                                                |`sde.diff(t)`                                   |
|$X_0$                                       |Source sample                                                        |`source.sample([B,])`                           |
|$X_1$                                       |Terminal sample                                                      |Output of `sdeint(sde, x0, timesteps)`          |
|$k(x, x’)$                                  |Base kernel (RBF)                                                    |New: `rbf_kernel(x, x', ell)`                   |
|$k_p(x, x’)$                                |Stein kernel                                                         |New: `stein_kernel(x, x', scores, ell)`         |
|$K_p$                                       |Stein kernel matrix                                                  |New: `stein_kernel_matrix(samples, scores, ell)`|
|$\widehat{\text{KSD}}^2$                    |Empirical KSD                                                        |New: `compute_ksd(samples, scores, ell)`        |
|$g_\phi(x)$                                 |Neural control variate network                                       |New: `NeuralSteinCV(d)`                         |
|$\mathcal{L}_{\text{PDE}}(\phi)$            |PDE residual loss: $\mathbb{E}[|\nabla(f + \mathcal{A}*p g*\phi)|^2]$|New: `neural_stein_cv_loss(...)`                |