"""
enhancements/mcmc_correction.py

Metropolis-Hastings post-correction of ASBS terminal samples.
Uses energy DIFFERENCES only — no q_theta density needed.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from adjoint_samplers.energies.base_energy import BaseEnergy


@torch.no_grad()
def mh_correct(
    samples: torch.Tensor,
    energy: "BaseEnergy",
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

    # Auto step size: optimal scaling for Gaussian targets
    if step_size is None:
        marginal_std = samples.std(dim=0).mean()
        step_size = 2.38 / (D ** 0.5) * marginal_std.item()

    x = samples.clone()
    E_x = energy.eval(x)  # (N,)
    energies_before = E_x.clone()

    total_accepted = 0
    total_proposed = 0

    for _ in range(n_steps):
        # Propose: symmetric random walk
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
        'energies_before': energies_before,
        'energies_after': E_x,
    }
