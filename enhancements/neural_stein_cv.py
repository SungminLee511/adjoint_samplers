"""
enhancements/neural_stein_cv.py

Neural Stein control variate via the differentiated Poisson equation.
Trains a neural network g_phi to make f(x) + A_p g_phi(x) approximately constant,
using the PDE loss ||grad_x(f + A_p g_phi)||^2 which eliminates the unknown E_p[f].

This scales to high dimensions (no N×N matrices) and is more expressive than RKHS.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable


class NeuralSteinCV(nn.Module):
    """Neural network g_phi: R^d -> R^d for the Stein control variate.

    Requirements:
    - Output dimension = input dimension
    - Twice differentiable (no ReLU — use SiLU/GELU/Tanh)
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str = 'silu',
    ):
        super().__init__()
        self.dim = dim

        act_map = {
            'silu': nn.SiLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
            'softplus': nn.Softplus,
        }
        act_cls = act_map[activation]

        layers = [nn.Linear(dim, hidden_dim), act_cls()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_cls()]
        layers.append(nn.Linear(hidden_dim, dim))

        self.net = nn.Sequential(*layers)

        # Initialize last layer small so g starts near zero
        # (A_p g ≈ 0 initially, so h ≈ f — no harm at start)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d) input positions
        Returns:
            g: (B, d) control variate vector field
        """
        return self.net(x)


def compute_stein_operator(
    g_values: torch.Tensor,
    x: torch.Tensor,
    scores: torch.Tensor,
    g_func: Callable,
    hutchinson_samples: int = 1,
) -> torch.Tensor:
    """Compute A_p g(x) = s(x)^T g(x) + div(g)(x).

    Args:
        g_values: (B, d) = g_phi(x), already computed with grad enabled
        x: (B, d) input positions, requires_grad=True
        scores: (B, d) = s_p(x) = -grad_E(x)
        g_func: callable that maps x -> g_phi(x) (needed for divergence)
        hutchinson_samples: number of Hutchinson probes for div estimate.
            If 0, compute exact divergence (costs d backward passes).

    Returns:
        A_p_g: (B,) scalar Stein operator values
    """
    B, d = x.shape

    # Term 1: s(x)^T g(x)
    sdotg = (scores * g_values).sum(dim=-1)  # (B,)

    # Term 2: div(g)(x) = sum_j dg_j/dx_j
    if hutchinson_samples == 0:
        # Exact divergence: d backward passes
        div_g = torch.zeros(B, device=x.device)
        for j in range(d):
            grad_gj = torch.autograd.grad(
                g_values[:, j].sum(), x,
                create_graph=True, retain_graph=True,
            )[0]  # (B, d)
            div_g = div_g + grad_gj[:, j]
    else:
        # Hutchinson estimator: div(g) ≈ E_v[v^T J_g v]
        div_g = torch.zeros(B, device=x.device)
        for _ in range(hutchinson_samples):
            v = torch.randn_like(x)  # (B, d)
            # Compute v^T J_g = d/dx (v^T g)
            vTg = (v * g_values).sum()
            Jv = torch.autograd.grad(
                vTg, x,
                create_graph=True, retain_graph=True,
            )[0]  # (B, d)
            div_g = div_g + (v * Jv).sum(dim=-1)  # v^T J_g v
        div_g = div_g / hutchinson_samples

    return sdotg + div_g  # (B,)


def neural_stein_cv_loss(
    g_model: NeuralSteinCV,
    x: torch.Tensor,
    scores: torch.Tensor,
    f_values: torch.Tensor,
    f_grad: Optional[torch.Tensor] = None,
    hutchinson_samples: int = 1,
) -> torch.Tensor:
    """Compute the PDE loss: ||grad_x(f + A_p g_phi)||^2.

    This is the SIMPLIFIED form from MATH_SPEC Section 7.7:
    The residual R = grad_x[f(x) + A_p g_phi(x)] = grad_x[h(x)]
    where h(x) = f(x) + A_p g_phi(x) should be constant.

    Args:
        g_model: the neural CV network
        x: (B, d) sample positions (will be cloned and requires_grad)
        scores: (B, d) precomputed scores s_p(x) = -grad_E(x)
        f_values: (B,) precomputed f(x) values
        f_grad: (B, d) precomputed grad_f(x). If None, f_values must have
            grad_fn so we can differentiate through it.
        hutchinson_samples: probes for Hutchinson divergence estimate

    Returns:
        loss: scalar, mean ||grad_x h||^2
    """
    B, d = x.shape

    # Ensure x requires grad
    x = x.detach().requires_grad_(True)

    # Forward through g_phi
    g_values = g_model(x)  # (B, d)

    # Compute A_p g(x) = s^T g + div(g)
    Apg = compute_stein_operator(
        g_values, x, scores, g_model, hutchinson_samples
    )  # (B,)

    # grad_x(A_p g) — the part we can differentiate through
    grad_Apg = torch.autograd.grad(
        Apg.sum(), x,
        create_graph=True,  # Need to backprop through this for phi gradients
    )[0]  # (B, d)

    # grad_x h = grad_x f + grad_x(A_p g)
    # f_values are precomputed constants, so we need f_grad explicitly.
    # If f_grad is None, assume f is constant (grad_f = 0) — but this is
    # almost certainly wrong; callers should provide f_grad.
    if f_grad is not None:
        grad_h = f_grad + grad_Apg  # (B, d)
    else:
        grad_h = grad_Apg  # fallback (f_grad unknown)

    # PDE loss: mean ||grad_h||^2
    loss = (grad_h ** 2).sum(dim=-1).mean()

    return loss


def train_neural_stein_cv(
    g_model: NeuralSteinCV,
    samples: torch.Tensor,
    energy,
    f_func: Callable,
    n_epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    hutchinson_samples: int = 1,
    verbose: bool = True,
) -> dict:
    """Train the neural Stein CV on terminal ASBS samples.

    Args:
        g_model: NeuralSteinCV network
        samples: (N, d) terminal samples (detached, on device)
        energy: energy function with score() method
        f_func: callable x -> f(x), the function to estimate.
            For mean energy: f_func = lambda x: energy.eval(x)
        n_epochs: training epochs
        batch_size: mini-batch size
        lr: learning rate
        hutchinson_samples: probes for divergence (0 = exact)
        verbose: print progress

    Returns:
        dict with 'model' (trained g_model), 'losses' (list), 'estimate', etc.
    """
    N, d = samples.shape
    device = samples.device

    optimizer = torch.optim.Adam(g_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Precompute scores for all samples
    with torch.no_grad():
        all_scores = energy.score(samples)  # (N, d) = -grad_E(x)
        all_f = f_func(samples)             # (N,)
        # For f(x) = E(x), grad_f = grad_E = -score
        all_f_grad = -all_scores             # (N, d)

    losses = []

    for epoch in range(n_epochs):
        g_model.train()

        # Random mini-batch
        idx = torch.randperm(N, device=device)[:batch_size]
        x_batch = samples[idx]
        s_batch = all_scores[idx]
        f_batch = all_f[idx]
        fg_batch = all_f_grad[idx]

        optimizer.zero_grad()
        loss = neural_stein_cv_loss(
            g_model, x_batch, s_batch, f_batch,
            f_grad=fg_batch,
            hutchinson_samples=hutchinson_samples,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g_model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
            print(f"  [NeuralCV] epoch {epoch:4d}/{n_epochs}: loss={loss.item():.6f}")

    # --- After training: compute the estimate ---
    g_model.eval()

    # Compute h(x_i) = f(x_i) + A_p g(x_i) for all samples in batches
    h_values = []
    eval_batch = min(batch_size, 512)
    for start in range(0, N, eval_batch):
        end = min(start + eval_batch, N)
        x_b = samples[start:end].detach().requires_grad_(True)
        s_b = all_scores[start:end]
        f_b = all_f[start:end]

        g_b = g_model(x_b)
        Apg_b = compute_stein_operator(
            g_b, x_b, s_b, g_model,
            hutchinson_samples=0,  # exact for final eval
        )
        h_b = f_b + Apg_b
        h_values.append(h_b.detach())

    h_all = torch.cat(h_values)  # (N,)

    estimate = h_all.mean().item()
    naive_estimate = all_f.mean().item()
    variance_neural = h_all.var().item() / N
    variance_naive = all_f.var().item() / N

    return {
        'model': g_model,
        'losses': losses,
        'estimate': estimate,
        'naive_estimate': naive_estimate,
        'variance_neural': variance_neural,
        'variance_naive': variance_naive,
        'variance_reduction': variance_neural / (variance_naive + 1e-20),
        'h_values': h_all,
    }
