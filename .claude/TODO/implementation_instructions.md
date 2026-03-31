# Implementation Guide: Stein Enhancements for ASBS

## Step-by-Step Instructions for Building on the `adjoint_samplers` Codebase

-----

## 0. Project Structure

**DO NOT modify any files in `adjoint_samplers/`.** The original codebase must remain intact for baseline comparison. All new code goes in `enhancements/` at the repo root.

```
adjoint_samplers/                  # ORIGINAL — do not modify
    adjoint_samplers/
        components/
            sde.py                 # ControlledSDE, sdeint
            matcher.py             # AdjointMatcher, AdjointVEMatcher
            evaluator.py           # SyntheticEnergyEvaluator
            model.py               # FourierMLP, EGNN_dynamics
            term_cost.py           # GradEnergy
            buffer.py              # BatchBuffer
            state_cost.py
        energies/
            base_energy.py         # BaseEnergy (eval, grad_E, score)
            double_well_energy.py
            lennard_jones_energy.py
        utils/
            train_utils.py
            eval_utils.py
            dist_utils.py
    train.py                       # Original training entry point
    configs/

enhancements/                      # NEW — all our code
    __init__.py
    stein_kernel.py                # Task 1: KSD + Stein kernel computation
    stein_cv.py                    # Task 2: Stein control variate estimator
    antithetic.py                  # Task 3: Antithetic SDE integration
    mcmc_correction.py             # Task 4: MH post-correction
    enhanced_evaluator.py          # Task 5: Enhanced evaluator wrapping all methods
    generator_stein.py             # Task 6: SDE generator Stein operator

eval_enhanced.py                   # NEW — evaluation script using enhancements
docs/
    MATH_SPEC.md
    IMPLEMENTATION_GUIDE.md        # This file
```

Create the `enhancements/` directory and `enhancements/__init__.py` (empty) first.

-----

## Task 1: Stein Kernel and KSD (`enhancements/stein_kernel.py`)

This is the foundational module. Everything else depends on it.

### 1.1 Functions to Implement

```python
"""
enhancements/stein_kernel.py

Core Stein kernel computations: RBF kernel, Stein kernel, KSD.
All functions operate on terminal samples + precomputed scores.
"""

import torch
from typing import Optional


def median_bandwidth(samples: torch.Tensor) -> torch.Tensor:
    """Compute median heuristic bandwidth for RBF kernel.

    Args:
        samples: (N, D) tensor of sample positions

    Returns:
        ell: scalar tensor, the median pairwise distance
    """
    # Compute pairwise distances (N, N)
    # Use upper triangle only (no duplicates, no diagonal)
    # Return median of these distances
    # Handle: if N is large, subsample to avoid O(N^2) memory
    pass


def rbf_kernel_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    ell: torch.Tensor,
) -> torch.Tensor:
    """Compute RBF kernel matrix k(x_i, y_j) = exp(-||x_i - y_j||^2 / (2*ell^2)).

    Args:
        x: (N, D) tensor
        y: (M, D) tensor
        ell: scalar bandwidth

    Returns:
        K: (N, M) kernel matrix
    """
    pass


def stein_kernel_matrix(
    samples: torch.Tensor,
    scores: torch.Tensor,
    ell: torch.Tensor,
) -> torch.Tensor:
    """Compute the Stein kernel matrix K_p where (K_p)_{ij} = k_p(x_i, x_j).

    Uses RBF base kernel. The Stein kernel is:
        k_p(x, x') = s(x)^T k(x,x') s(x')
                    + s(x)^T nabla_{x'} k(x,x')
                    + nabla_x k(x,x')^T s(x')
                    + tr(nabla_x nabla_{x'} k(x,x'))

    For RBF kernel k(x,x') = exp(-||x-x'||^2 / (2*ell^2)):
        nabla_x k = -(x-x')/ell^2 * k
        nabla_{x'} k = (x-x')/ell^2 * k
        tr(nabla_x nabla_{x'} k) = (d/ell^2 - ||x-x'||^2/ell^4) * k

    Args:
        samples: (N, D) tensor of sample positions
        scores: (N, D) tensor of scores s_p(x_i) = -grad_E(x_i)
        ell: scalar bandwidth

    Returns:
        K_p: (N, N) Stein kernel matrix
    """
    # IMPLEMENTATION NOTES:
    # 1. Compute pairwise difference matrix: diff[i,j] = x_i - x_j, shape (N, N, D)
    # 2. Compute squared distances: sq_dist[i,j] = ||x_i - x_j||^2, shape (N, N)
    # 3. Compute RBF kernel: K[i,j] = exp(-sq_dist / (2*ell^2)), shape (N, N)
    # 4. Compute four terms of the Stein kernel:
    #    term1[i,j] = s_i^T s_j * K[i,j]
    #         -> (scores @ scores.T) * K
    #    term2[i,j] = s_i^T * (x_j - x_i) / ell^2 * K[i,j]
    #         -> need einsum or manual: sum over d of scores[i,d] * (-diff[i,j,d]) / ell^2 * K[i,j]
    #    term3[i,j] = (x_i - x_j)^T / ell^2 * s_j * K[i,j]
    #         -> symmetric to term2 with transposed roles
    #    term4[i,j] = (d/ell^2 - sq_dist[i,j]/ell^4) * K[i,j]
    #
    # Be very careful with signs. Refer to MATH_SPEC.md Section 1.4.
    pass


def compute_ksd(
    samples: torch.Tensor,
    scores: torch.Tensor,
    ell: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the U-statistic estimator of KSD^2.

    Args:
        samples: (N, D) tensor
        scores: (N, D) tensor of s_p(x_i)
        ell: bandwidth (if None, use median heuristic)

    Returns:
        ksd_squared: scalar tensor
    """
    # 1. If ell is None, compute median bandwidth
    # 2. Compute Stein kernel matrix K_p
    # 3. KSD^2 = (1/(N*(N-1))) * (sum(K_p) - trace(K_p))
    #    The subtraction of the trace removes diagonal terms (U-statistic)
    pass
```

### 1.2 Testing

Write a simple test: for samples drawn from a known Gaussian $p = \mathcal{N}(0, I)$, the KSD should be near zero. For samples from a different distribution (e.g., $\mathcal{N}(1, I)$), KSD should be positive.

```python
def test_ksd():
    d = 10
    N = 1000
    # True samples from N(0, I)
    x = torch.randn(N, d)
    scores = -x  # score of N(0,I) is -x
    ksd = compute_ksd(x, scores)
    print(f"KSD (true samples): {ksd.item():.6f}")  # Should be ~0

    # Biased samples from N(1, I)
    x_biased = torch.randn(N, d) + 1.0
    scores_biased = -x_biased  # Wrong! Using score of N(0,I) but samples from N(1,I)
    # Actually for testing, scores should be the TARGET scores at the sample locations:
    # s_p(x) = -x for p = N(0,I), regardless of where x came from
    scores_at_biased = -x_biased  # score of target N(0,I) evaluated at biased samples
    ksd_biased = compute_ksd(x_biased, scores_at_biased)
    print(f"KSD (biased samples): {ksd_biased.item():.6f}")  # Should be > 0
```

-----

## Task 2: Stein Control Variate Estimator (`enhancements/stein_cv.py`)

### 2.1 Functions to Implement

```python
"""
enhancements/stein_cv.py

Stein control variate estimator for variance reduction.
"""

import torch
from typing import Optional, Callable
from enhancements.stein_kernel import stein_kernel_matrix, median_bandwidth


def stein_cv_estimate(
    samples: torch.Tensor,
    scores: torch.Tensor,
    f_values: torch.Tensor,
    ell: Optional[torch.Tensor] = None,
    reg_lambda: float = 1e-4,
) -> dict:
    """Compute the Stein control variate estimator for E_p[f(X)].

    Args:
        samples: (N, D) terminal samples
        scores: (N, D) scores s_p(x_i) = -grad_E(x_i)
        f_values: (N,) function evaluations f(x_i)
        ell: RBF bandwidth (None = median heuristic)
        reg_lambda: regularization for kernel ridge regression

    Returns:
        dict with keys:
            'estimate': scalar, the Stein CV estimate of E_p[f]
            'naive_estimate': scalar, the vanilla sample mean
            'variance_naive': scalar, variance of naive estimator
            'variance_stein': scalar, estimated variance of Stein estimator
            'coefficients': (N,) optimal coefficients a
    """
    N = samples.shape[0]

    # 1. Compute bandwidth if not provided
    if ell is None:
        ell = median_bandwidth(samples)

    # 2. Compute Stein kernel matrix K_p (N, N)
    K_p = stein_kernel_matrix(samples, scores, ell)

    # 3. Solve (K_p + lambda*N*I) a = f_values
    A = K_p + reg_lambda * N * torch.eye(N, device=samples.device)
    # Use torch.linalg.solve for numerical stability
    a = torch.linalg.solve(A, f_values)

    # 4. Compute Stein CV estimate: 1^T a
    estimate = a.sum()

    # 5. Compute naive estimate for comparison
    naive_estimate = f_values.mean()

    # 6. Compute variance estimates
    # Naive variance: var(f) / N
    variance_naive = f_values.var() / N
    # Stein corrected values: f_i - sum_j a_j k_p(x_j, x_i)
    correction = K_p @ a
    corrected_values = f_values - correction
    variance_stein = corrected_values.var() / N

    return {
        'estimate': estimate.item(),
        'naive_estimate': naive_estimate.item(),
        'variance_naive': variance_naive.item(),
        'variance_stein': variance_stein.item(),
        'coefficients': a.detach(),
    }


def multi_function_stein_cv(
    samples: torch.Tensor,
    scores: torch.Tensor,
    f_dict: dict,
    ell: Optional[torch.Tensor] = None,
    reg_lambda: float = 1e-4,
) -> dict:
    """Apply Stein CV to multiple functions simultaneously.

    The Stein kernel matrix is computed once and reused.

    Args:
        samples: (N, D)
        scores: (N, D)
        f_dict: dict mapping function names to (N,) tensors of evaluations
        ell: bandwidth
        reg_lambda: regularization

    Returns:
        dict mapping function names to result dicts from stein_cv_estimate
    """
    N = samples.shape[0]

    if ell is None:
        ell = median_bandwidth(samples)

    K_p = stein_kernel_matrix(samples, scores, ell)
    A = K_p + reg_lambda * N * torch.eye(N, device=samples.device)

    # Cholesky factorize once, solve for multiple RHS
    L = torch.linalg.cholesky(A)

    results = {}
    for name, f_values in f_dict.items():
        a = torch.cholesky_solve(f_values.unsqueeze(1), L).squeeze(1)
        estimate = a.sum()
        naive_estimate = f_values.mean()
        correction = K_p @ a
        corrected_values = f_values - correction
        results[name] = {
            'estimate': estimate.item(),
            'naive_estimate': naive_estimate.item(),
            'variance_naive': f_values.var().item() / N,
            'variance_stein': corrected_values.var().item() / N,
        }

    return results
```

-----

## Task 3: Antithetic SDE Integration (`enhancements/antithetic.py`)

### 3.1 Functions to Implement

```python
"""
enhancements/antithetic.py

Antithetic SDE integration: generate paired trajectories with negated noise.
"""

import torch
from typing import List
from adjoint_samplers.components.sde import BaseSDE


@torch.no_grad()
def sdeint_antithetic(
    sde: BaseSDE,
    state0: torch.Tensor,
    timesteps: torch.Tensor,
    only_boundary: bool = False,
) -> tuple:
    """Simulate two SDE trajectories with antithetic noise.

    Returns both the original and antithetic terminal samples.
    Uses the SAME drift evaluations but NEGATED noise.

    Args:
        sde: the controlled SDE
        state0: (B, D) initial states
        timesteps: (T,) time discretization
        only_boundary: if True, return only (x0, x1, x1_anti)

    Returns:
        If only_boundary:
            (state0, state1, state1_anti) — three tensors
        Else:
            (states, states_anti) — two lists of length T
    """
    T = len(timesteps)
    assert T > 1

    sde.train(False)

    state = state0.clone()
    state_anti = state0.clone()

    states = [state0]
    states_anti = [state0]

    for i in range(T - 1):
        t = timesteps[i]
        dt = timesteps[i + 1] - t

        # Shared noise
        noise = sde.randn_like(state)

        # Original trajectory
        drift = sde.drift(t, state) * dt
        diffusion = sde.diff(t) * dt.sqrt() * noise
        d_state = drift + diffusion
        state = sde.propagate(state, d_state)
        states.append(state)

        # Antithetic trajectory (negated noise, recomputed drift at anti state)
        drift_anti = sde.drift(t, state_anti) * dt
        diffusion_anti = sde.diff(t) * dt.sqrt() * (-noise)  # NEGATED
        d_state_anti = drift_anti + diffusion_anti
        state_anti = sde.propagate(state_anti, d_state_anti)
        states_anti.append(state_anti)

    if only_boundary:
        return states[0], states[-1], states_anti[-1]
    return states, states_anti


def antithetic_estimate(
    f_values: torch.Tensor,
    f_values_anti: torch.Tensor,
) -> dict:
    """Compute antithetic estimator and compare to naive.

    Args:
        f_values: (N,) function values from original trajectories
        f_values_anti: (N,) function values from antithetic trajectories

    Returns:
        dict with 'estimate', 'naive_estimate', 'variance_naive',
        'variance_anti', 'correlation'
    """
    N = f_values.shape[0]

    # Naive: just use original samples
    naive_est = f_values.mean()
    var_naive = f_values.var() / N

    # Antithetic: average the two
    paired = 0.5 * (f_values + f_values_anti)
    anti_est = paired.mean()
    var_anti = paired.var() / N

    # Correlation between the two
    cov = ((f_values - f_values.mean()) * (f_values_anti - f_values_anti.mean())).mean()
    corr = cov / (f_values.std() * f_values_anti.std() + 1e-8)

    return {
        'estimate': anti_est.item(),
        'naive_estimate': naive_est.item(),
        'variance_naive': var_naive.item(),
        'variance_anti': var_anti.item(),
        'correlation': corr.item(),
        'variance_reduction_factor': (var_anti / (var_naive + 1e-10)).item(),
    }
```

-----

## Task 4: MCMC Post-Correction (`enhancements/mcmc_correction.py`)

### 4.1 Functions to Implement

```python
"""
enhancements/mcmc_correction.py

Metropolis-Hastings post-correction of ASBS terminal samples.
"""

import torch
from adjoint_samplers.energies.base_energy import BaseEnergy


@torch.no_grad()
def mh_correct(
    samples: torch.Tensor,
    energy: BaseEnergy,
    n_steps: int = 10,
    step_size: float | None = None,
) -> dict:
    """Apply random-walk Metropolis-Hastings to ASBS terminal samples.

    Each sample is corrected independently with n_steps MH iterations.
    Uses energy DIFFERENCES only — no q_theta needed.

    Args:
        samples: (N, D) terminal ASBS samples
        energy: energy function with eval() method
        n_steps: number of MH steps per sample
        step_size: proposal std. If None, use 2.38/sqrt(D) * marginal_std

    Returns:
        dict with keys:
            'corrected_samples': (N, D) corrected samples
            'acceptance_rate': float, overall acceptance rate
            'energies_before': (N,) energies of original samples
            'energies_after': (N,) energies of corrected samples
    """
    N, D = samples.shape
    device = samples.device

    # Auto step size
    if step_size is None:
        marginal_std = samples.std(dim=0).mean()
        step_size = 2.38 / (D ** 0.5) * marginal_std.item()

    x = samples.clone()
    E_x = energy.eval(x)  # (N,)

    total_accepted = 0
    total_proposed = 0

    for _ in range(n_steps):
        # Propose
        x_prop = x + step_size * torch.randn_like(x)
        E_prop = energy.eval(x_prop)

        # Acceptance ratio (symmetric proposal, so just exp(-ΔE))
        delta_E = E_prop - E_x  # (N,)
        log_alpha = -delta_E
        log_alpha = torch.clamp(log_alpha, max=0.0)  # min(1, exp(-ΔE))
        accept_prob = torch.exp(log_alpha)

        # Accept/reject per sample
        u = torch.rand(N, device=device)
        accept = u < accept_prob  # (N,) boolean

        x = torch.where(accept.unsqueeze(1), x_prop, x)
        E_x = torch.where(accept, E_prop, E_x)

        total_accepted += accept.sum().item()
        total_proposed += N

    return {
        'corrected_samples': x,
        'acceptance_rate': total_accepted / total_proposed,
        'energies_before': energy.eval(samples),
        'energies_after': E_x,
    }
```

-----

## Task 5: Enhanced Evaluator (`enhancements/enhanced_evaluator.py`)

This wraps everything together and runs all enhancements on a set of terminal samples.

### 5.1 Functions to Implement

```python
"""
enhancements/enhanced_evaluator.py

Enhanced evaluator that wraps vanilla ASBS evaluation with all enhancements.
"""

import torch
from typing import Optional
from adjoint_samplers.energies.base_energy import BaseEnergy

from enhancements.stein_kernel import compute_ksd, median_bandwidth
from enhancements.stein_cv import multi_function_stein_cv
from enhancements.mcmc_correction import mh_correct


def evaluate_enhanced(
    samples: torch.Tensor,
    energy: BaseEnergy,
    ref_energies: Optional[torch.Tensor] = None,
    mh_steps: int = 10,
    stein_reg_lambda: float = 1e-4,
    max_stein_samples: int = 2000,
) -> dict:
    """Run all enhancements on terminal ASBS samples.

    Args:
        samples: (N, D) terminal samples from ASBS
        energy: energy function
        ref_energies: (M,) reference energy values for ground truth comparison
        mh_steps: number of MH correction steps
        stein_reg_lambda: regularization for Stein CV
        max_stein_samples: max samples for Stein kernel (O(N^2) memory)

    Returns:
        dict with all metrics
    """
    N, D = samples.shape
    device = samples.device
    results = {}

    # --- 1. Compute energies and scores ---
    with torch.no_grad():
        energies = energy.eval(samples)        # (N,)
    scores = energy.score(samples)             # (N, D) = -grad_E

    # --- 2. KSD diagnostic ---
    # Subsample if N is too large for O(N^2)
    N_ksd = min(N, max_stein_samples)
    idx = torch.randperm(N)[:N_ksd]
    ksd_sq = compute_ksd(samples[idx], scores[idx])
    results['ksd_squared'] = ksd_sq.item()

    # --- 3. Vanilla estimates ---
    results['mean_energy_naive'] = energies.mean().item()
    results['std_energy_naive'] = energies.std().item()
    results['var_energy_estimator_naive'] = (energies.var() / N).item()

    # --- 4. Stein CV estimates ---
    N_stein = min(N, max_stein_samples)
    idx_stein = torch.randperm(N)[:N_stein]
    f_dict = {
        'energy': energies[idx_stein],
    }
    stein_results = multi_function_stein_cv(
        samples[idx_stein],
        scores[idx_stein],
        f_dict,
        reg_lambda=stein_reg_lambda,
    )
    for fname, fres in stein_results.items():
        results[f'stein_cv_{fname}_estimate'] = fres['estimate']
        results[f'stein_cv_{fname}_var_naive'] = fres['variance_naive']
        results[f'stein_cv_{fname}_var_stein'] = fres['variance_stein']
        if fres['variance_naive'] > 0:
            results[f'stein_cv_{fname}_var_reduction'] = (
                fres['variance_stein'] / fres['variance_naive']
            )

    # --- 5. MCMC correction ---
    mh_result = mh_correct(samples, energy, n_steps=mh_steps)
    corrected = mh_result['corrected_samples']
    results['mh_acceptance_rate'] = mh_result['acceptance_rate']

    corrected_energies = energy.eval(corrected)
    results['mean_energy_mcmc'] = corrected_energies.mean().item()
    results['var_energy_estimator_mcmc'] = (corrected_energies.var() / N).item()

    # --- 6. MCMC + Stein CV (the full pipeline) ---
    corrected_scores = energy.score(corrected)
    N_hybrid = min(N, max_stein_samples)
    idx_hybrid = torch.randperm(N)[:N_hybrid]
    hybrid_f_dict = {
        'energy': corrected_energies[idx_hybrid],
    }
    hybrid_results = multi_function_stein_cv(
        corrected[idx_hybrid],
        corrected_scores[idx_hybrid],
        hybrid_f_dict,
        reg_lambda=stein_reg_lambda,
    )
    for fname, fres in hybrid_results.items():
        results[f'hybrid_{fname}_estimate'] = fres['estimate']
        results[f'hybrid_{fname}_var_stein'] = fres['variance_stein']
        if fres['variance_naive'] > 0:
            results[f'hybrid_{fname}_var_reduction'] = (
                fres['variance_stein'] / fres['variance_naive']
            )

    # --- 7. Ground truth comparison ---
    if ref_energies is not None:
        gt_mean = ref_energies.mean().item()
        results['gt_mean_energy'] = gt_mean
        results['error_naive'] = abs(results['mean_energy_naive'] - gt_mean)
        results['error_stein'] = abs(stein_results['energy']['estimate'] - gt_mean)
        results['error_mcmc'] = abs(results['mean_energy_mcmc'] - gt_mean)
        results['error_hybrid'] = abs(hybrid_results['energy']['estimate'] - gt_mean)

    return results
```

-----

## Task 6: Generator Stein Operator (`enhancements/generator_stein.py`)

### 6.1 Functions to Implement

```python
"""
enhancements/generator_stein.py

SDE generator-based Stein kernel: uses the learned drift b_theta
instead of the generic score s_p for potentially better control variates.
"""

import torch
from typing import Optional
from adjoint_samplers.components.sde import ControlledSDE
from enhancements.stein_kernel import median_bandwidth


def generator_stein_kernel_matrix(
    samples: torch.Tensor,
    sde: ControlledSDE,
    ell: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Stein kernel using the SDE generator instead of generic score.

    Replaces s_p(x) with b_theta(x, 1) = f(x,1) + g(1)^2 * u_theta(x, 1)
    in the Stein kernel formula.

    The diffusion coefficient g(1)^2/2 replaces the factor 1 in the
    trace term of the standard Stein kernel.

    Args:
        samples: (N, D) terminal samples
        sde: the trained ControlledSDE
        ell: bandwidth (None = median heuristic)

    Returns:
        K_gen: (N, N) generator Stein kernel matrix
    """
    N, D = samples.shape

    if ell is None:
        ell = median_bandwidth(samples)

    # Get the total drift at t=1 for each sample
    t1 = torch.ones(N, 1, device=samples.device)
    with torch.no_grad():
        drift = sde.drift(t1, samples)  # (N, D) = f(x,1) + g(1)^2 * u(x,1)
    diff_sq = sde.diff(t1[0]).item() ** 2  # g(1)^2, scalar

    # Compute pairwise differences
    diff = samples.unsqueeze(1) - samples.unsqueeze(0)  # (N, N, D)
    sq_dist = (diff ** 2).sum(-1)  # (N, N)

    # RBF kernel
    K = torch.exp(-sq_dist / (2 * ell ** 2))

    # Term 1: drift_i^T drift_j * K
    term1 = (drift @ drift.T) * K

    # Term 2: drift_i^T (x_j - x_i) / ell^2 * K
    # diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    term2 = -torch.einsum('id,ijd->ij', drift, diff) / (ell ** 2) * K

    # Term 3: (x_i - x_j)^T drift_j / ell^2 * K
    term3 = torch.einsum('ijd,jd->ij', diff, drift) / (ell ** 2) * K

    # Term 4: (g(1)^2 / 2) * (d/ell^2 - sq_dist/ell^4) * K
    # Note: factor of g(1)^2/2 from the diffusion part of the generator
    term4 = (diff_sq / 2) * (D / (ell ** 2) - sq_dist / (ell ** 4)) * K

    K_gen = term1 + term2 + term3 + term4
    return K_gen


def generator_stein_cv_estimate(
    samples: torch.Tensor,
    sde: ControlledSDE,
    f_values: torch.Tensor,
    ell: Optional[torch.Tensor] = None,
    reg_lambda: float = 1e-4,
) -> dict:
    """Stein CV using the SDE generator kernel.

    Same as stein_cv_estimate but with the generator kernel.

    Args:
        samples: (N, D)
        sde: trained ControlledSDE
        f_values: (N,)
        ell: bandwidth
        reg_lambda: regularization

    Returns:
        dict with 'estimate', 'naive_estimate', 'variance_naive', 'variance_gen_stein'
    """
    N = samples.shape[0]

    if ell is None:
        ell = median_bandwidth(samples)

    K_gen = generator_stein_kernel_matrix(samples, sde, ell)
    A = K_gen + reg_lambda * N * torch.eye(N, device=samples.device)
    a = torch.linalg.solve(A, f_values)

    estimate = a.sum()
    naive_estimate = f_values.mean()

    correction = K_gen @ a
    corrected_values = f_values - correction

    return {
        'estimate': estimate.item(),
        'naive_estimate': naive_estimate.item(),
        'variance_naive': f_values.var().item() / N,
        'variance_gen_stein': corrected_values.var().item() / N,
    }
```

-----

## Task 7: Evaluation Script (`eval_enhanced.py`)

This is the main entry point for running all enhancements on a trained ASBS checkpoint.

### 7.1 Implementation

```python
"""
eval_enhanced.py

Load a trained ASBS checkpoint and run all enhancements.
Usage:
    python eval_enhanced.py experiment=dw4_asbs checkpoint=checkpoints/checkpoint_latest.pt
"""

import hydra
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.utils import train_utils

# Our enhancements
from enhancements.enhanced_evaluator import evaluate_enhanced
from enhancements.antithetic import sdeint_antithetic, antithetic_estimate
from enhancements.stein_kernel import compute_ksd
from enhancements.generator_stein import generator_stein_cv_estimate


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Setup (same as train.py) ---
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    # Load checkpoint
    checkpoint_path = Path(cfg.checkpoint or "checkpoints/checkpoint_latest.pt")
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    controller.load_state_dict(checkpoint["controller"])
    print(f"Loaded checkpoint from {checkpoint_path}")

    # --- Load reference samples for ground truth ---
    evaluator = hydra.utils.instantiate(cfg.evaluator, energy=energy)
    if hasattr(evaluator, 'ref_samples'):
        ref_samples = evaluator.ref_samples.to(device)
        ref_energies = energy.eval(ref_samples)
    else:
        ref_energies = None

    # --- Generate samples ---
    N = cfg.get('num_eval_samples', 2000)
    B = cfg.get('eval_batch_size', 2000)
    timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)

    print(f"Generating {N} samples...")
    x1_list = []
    n_gen = 0
    while n_gen < N:
        b = min(B, N - n_gen)
        x0 = source.sample([b]).to(device)
        _, x1 = sdeint(sde, x0, timesteps, only_boundary=True)
        x1_list.append(x1)
        n_gen += b
    samples = torch.cat(x1_list, dim=0)
    print(f"Generated {samples.shape[0]} samples, shape {samples.shape}")

    # --- Run enhanced evaluation ---
    print("\n=== Enhanced Evaluation ===")
    results = evaluate_enhanced(
        samples, energy,
        ref_energies=ref_energies,
        mh_steps=10,
        stein_reg_lambda=1e-4,
        max_stein_samples=min(N, 2000),
    )

    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.6f}")

    # --- Antithetic sampling ---
    print("\n=== Antithetic Sampling ===")
    x0 = source.sample([min(N, 1000)]).to(device)
    _, x1_orig, x1_anti = sdeint_antithetic(sde, x0, timesteps, only_boundary=True)
    E_orig = energy.eval(x1_orig)
    E_anti = energy.eval(x1_anti)
    anti_results = antithetic_estimate(E_orig, E_anti)
    for k, v in sorted(anti_results.items()):
        print(f"  {k}: {v:.6f}")

    # --- Generator Stein CV ---
    print("\n=== Generator Stein CV ===")
    N_gen = min(N, 1000)
    idx = torch.randperm(N)[:N_gen]
    gen_results = generator_stein_cv_estimate(
        samples[idx], sde, energy.eval(samples[idx]),
        reg_lambda=1e-4,
    )
    for k, v in sorted(gen_results.items()):
        print(f"  {k}: {v:.6f}")

    # --- Summary ---
    print("\n=== SUMMARY ===")
    if ref_energies is not None:
        gt = ref_energies.mean().item()
        print(f"  Ground truth mean energy: {gt:.6f}")
        print(f"  Naive estimate:           {results['mean_energy_naive']:.6f}  (error: {results.get('error_naive', 'N/A')})")
        print(f"  Stein CV estimate:        {results.get('stein_cv_energy_estimate', 'N/A')}  (error: {results.get('error_stein', 'N/A')})")
        print(f"  MCMC corrected:           {results['mean_energy_mcmc']:.6f}  (error: {results.get('error_mcmc', 'N/A')})")
        print(f"  MCMC + Stein CV:          {results.get('hybrid_energy_estimate', 'N/A')}  (error: {results.get('error_hybrid', 'N/A')})")
        print(f"  Antithetic:               {anti_results['estimate']:.6f}")
        print(f"  Generator Stein CV:       {gen_results['estimate']:.6f}")
    print(f"\n  KSD^2:                    {results['ksd_squared']:.6f}")
    print(f"  MH acceptance rate:       {results['mh_acceptance_rate']:.4f}")
    print(f"  Antithetic correlation:   {anti_results['correlation']:.4f}")


if __name__ == "__main__":
    main()
```

-----

## Task 8: Order of Implementation

Build and test in this order:

1. **`stein_kernel.py`** — Foundation. Test with synthetic Gaussians before anything else.
1. **`stein_cv.py`** — Depends on `stein_kernel.py`. Test on Gaussians.
1. **`antithetic.py`** — Independent of Stein. Test on a simple SDE.
1. **`mcmc_correction.py`** — Independent of Stein. Test with known energy.
1. **`generator_stein.py`** — Depends on `stein_kernel.py`. Needs a trained SDE.
1. **`enhanced_evaluator.py`** — Integrates everything.
1. **`eval_enhanced.py`** — Full pipeline.

For each module, after implementation:

1. Run the unit test (synthetic Gaussian or simple energy)
1. Run on DW4 (fastest benchmark, 12D)
1. If working, run on LJ13 (39D) and LJ55 (165D) to test scaling

-----

## Task 9: Running Experiments

### 9.1 First: Train Baseline ASBS

```bash
# Download reference samples
bash scripts/download.sh

# Train DW4 ASBS baseline (fastest)
python train.py experiment=dw4_asbs seed=0 use_wandb=false num_epochs=1000
```

This creates a checkpoint at `checkpoints/checkpoint_latest.pt`.

### 9.2 Then: Run Enhanced Evaluation

```bash
# Run all enhancements on the trained checkpoint
python eval_enhanced.py experiment=dw4_asbs checkpoint=checkpoints/checkpoint_latest.pt
```

### 9.3 Scaling Test

```bash
# LJ13 (39D)
python train.py experiment=lj13_asbs seed=0 use_wandb=false
python eval_enhanced.py experiment=lj13_asbs checkpoint=checkpoints/checkpoint_latest.pt

# LJ55 (165D) — test if Stein methods scale
python train.py experiment=lj55_asbs seed=0 use_wandb=false
python eval_enhanced.py experiment=lj55_asbs checkpoint=checkpoints/checkpoint_latest.pt
```

### 9.4 Key Questions Each Experiment Should Answer

|Enhancement       |Key question                               |How to measure                                  |
|------------------|-------------------------------------------|------------------------------------------------|
|KSD diagnostic    |Does KSD correlate with W2 metrics?        |Plot KSD vs energy_w2 across training           |
|Stein CV          |How much variance reduction on mean energy?|`variance_stein / variance_naive`               |
|Antithetic        |Is the correlation negative?               |`correlation` field                             |
|MCMC correction   |Does mean energy shift toward ground truth?|`error_mcmc` vs `error_naive`                   |
|MCMC + Stein CV   |Best of both worlds?                       |`error_hybrid` and `variance_hybrid`            |
|Generator Stein CV|Better than standard Stein CV?             |Compare `variance_gen_stein` vs `variance_stein`|

-----

## Important Notes for Claude Code

1. **Tensor devices:** Always keep tensors on the same device. The energy functions and SDE run on CUDA. All new code should respect `samples.device`.
1. **Memory:** The Stein kernel matrix is $O(N^2)$. For $N > 5000$, subsample. The `max_stein_samples` parameter handles this.
1. **Numerical stability:** The Stein kernel matrix can be ill-conditioned. The regularization $\lambda N I$ in the linear solve prevents this. If `torch.linalg.solve` fails, fall back to `torch.linalg.lstsq`.
1. **Graph particle systems:** DW4, LJ13, LJ55 use center-of-mass-free coordinates. The `Graph` mixin in `sde.py` handles this via `propagate` and `randn_like`. Make sure antithetic sampling respects this — use `sde.randn_like` (which projects to COM-free subspace) rather than `torch.randn_like`.
1. **Energy interface:** `energy.eval(x)` returns scalar energies (N,). `energy.grad_E(x)` returns (N, D). `energy.score(x)` returns (N, D) = `-grad_E(x)`. The `__call__` method returns `{"forces": grad_E(x)}`.
1. **No modification of originals:** All enhancements import from `adjoint_samplers` but never modify it. The baseline must always be reproducible.