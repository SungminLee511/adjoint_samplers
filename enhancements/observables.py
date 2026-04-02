"""
enhancements/observables.py

Observable functions for particle systems.

All functions accept flat input (B, dim) and reshape internally when needed.
This is consistent with the SML convention (flat vectors everywhere).

Ported from KSH_ASBS/stein_cv/observables.py, adapted to flat-input API.
"""

import torch
from torch import Tensor


def mean_energy_observable(x: Tensor, energy_fn) -> Tensor:
    """Compute energy for each sample.

    Args:
        x: (B, dim) flat coordinates
        energy_fn: callable with .eval(x) → (B,)

    Returns:
        (B,) energy values
    """
    return energy_fn.eval(x)


def mean_interatomic_distance(x: Tensor, n_particles: int, spatial_dim: int) -> Tensor:
    """Mean interatomic distance per sample.

    Args:
        x: (B, dim) flat coordinates where dim = n_particles * spatial_dim
        n_particles: number of particles
        spatial_dim: spatial dimensions per particle

    Returns:
        (B,) mean interatomic distance for each sample
    """
    B = x.shape[0]
    x_parts = x.view(B, n_particles, spatial_dim)  # (B, n, d)
    triu = torch.triu_indices(n_particles, n_particles, offset=1, device=x.device)
    xi = x_parts[:, triu[0]]  # (B, n_pairs, d)
    xj = x_parts[:, triu[1]]  # (B, n_pairs, d)
    dists = torch.norm(xi - xj, dim=-1)  # (B, n_pairs)
    return dists.mean(dim=-1)  # (B,)


def interatomic_dist_histogram(x: Tensor, n_particles: int, spatial_dim: int,
                                bins: int = 100, range_min: float = 0.0,
                                range_max: float = 5.0) -> Tensor:
    """Compute normalized histogram of all interatomic distances.

    Args:
        x: (B, dim) flat coordinates
        n_particles: number of particles
        spatial_dim: spatial dimensions per particle
        bins: number of histogram bins
        range_min: minimum distance for histogram
        range_max: maximum distance for histogram

    Returns:
        (bins,) normalized histogram
    """
    B = x.shape[0]
    x_parts = x.view(B, n_particles, spatial_dim)
    triu = torch.triu_indices(n_particles, n_particles, offset=1, device=x.device)
    xi = x_parts[:, triu[0]]
    xj = x_parts[:, triu[1]]
    dists = torch.norm(xi - xj, dim=-1).flatten()

    hist = torch.histc(dists, bins=bins, min=range_min, max=range_max)
    hist = hist / (hist.sum() + 1e-8)
    return hist


def observable_gradient(x: Tensor, observable_fn) -> Tensor:
    """Compute ∇f(x) w.r.t. flat x via autograd.

    Args:
        x: (B, dim) flat coordinates
        observable_fn: callable mapping (B, dim) → (B,)

    Returns:
        (B, dim) gradient of observable
    """
    x_req = x.detach().requires_grad_(True)
    f = observable_fn(x_req)
    grad_f = torch.autograd.grad(f.sum(), x_req, create_graph=True)[0]
    return grad_f
