"""
enhancements/antithetic.py

Antithetic SDE integration: generate paired trajectories with negated noise.
Zero additional energy evaluations — only one extra forward pass through the drift.
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
    Uses the SAME initial state but NEGATED noise increments.
    Each trajectory gets its own drift evaluations (at different x values).

    Uses sde.randn_like() for noise (respects COM-free projection for
    graph particle systems) and sde.propagate() for state updates.

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

        # Shared noise (uses sde.randn_like for COM-free projection)
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
        'variance_anti', 'correlation', 'variance_reduction_factor'
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
