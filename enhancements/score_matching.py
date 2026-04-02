"""
enhancements/score_matching.py

Implicit Score Matching (ISM) with EGNN-based score model.

Learns s_ϕ(x) ≈ ∇log q_θ(x) — the score of the *learned sampler distribution*,
NOT the target score -∇E. This is used by ScoreInformedSteinCV to construct a
better initial direction for the Stein control variate.

ISM objective:  L = E_q[½||s_ϕ(x)||² + div(s_ϕ)(x)]
Supports exact divergence (low-dim) or Hutchinson trace estimator (high-dim).

Ported from KSH_ASBS/stein_cv/score_matching.py
"""

import torch
import torch.nn as nn
from torch import Tensor


def _build_edges(n_particles, n_systems, device):
    """Build fully-connected edge indices for n_systems particle graphs.

    Returns:
        edge_index: (2, n_edges) tensor of [row, col] indices
        batch: (n_atoms,) system membership for each atom
    """
    n = n_particles
    src = torch.arange(n, device=device)
    row = src.repeat(n)
    col = src.repeat_interleave(n)
    mask = row != col
    row, col = row[mask], col[mask]
    offsets = torch.arange(n_systems, device=device) * n
    row_all = (row.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
    col_all = (col.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
    batch = torch.arange(n_systems, device=device).repeat_interleave(n)
    return torch.stack([row_all, col_all]), batch


class ScoreEGNN(nn.Module):
    """EGNN-based score model s_ϕ(x) ≈ ∇log q_θ(x).

    Wraps EGNN_dynamics internally. Evaluates at t=1.0 (terminal time).
    Coordinate layer weights reinitialized with small gain for stability.

    Forward signature matches particle format: (x, batch, edge_index).
    For flat-space usage, call via ImplicitScoreModel which handles reshaping.
    """

    def __init__(self, n_particles=4, spatial_dim=2, hidden_nf=128, n_layers=4,
                 coord_init_gain=0.1):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim

        from adjoint_samplers.components.model import EGNN_dynamics

        self.model = EGNN_dynamics(
            n_particles=n_particles, spatial_dim=spatial_dim,
            hidden_nf=hidden_nf, n_layers=n_layers,
            act_fn=nn.SiLU(), recurrent=True, tanh=False,
            attention=True, condition_time=True, agg="sum",
        )
        self._reinit_coord_weights(coord_init_gain)

    def _reinit_coord_weights(self, gain):
        """Reinitialize coord MLP last layers with small gain for stable ISM training."""
        for name, module in self.model.egnn.named_modules():
            if hasattr(module, 'coord_mlp'):
                last_linear = [m for m in module.coord_mlp if isinstance(m, nn.Linear)][-1]
                nn.init.xavier_uniform_(last_linear.weight, gain=gain)

    def forward(self, x: Tensor, batch: Tensor, edge_index: Tensor) -> Tensor:
        """Particle-format forward pass.

        Args:
            x: (n_atoms, spatial_dim) particle positions
            batch: (n_atoms,) system membership indices
            edge_index: (2, n_edges) edge indices

        Returns:
            (n_atoms, spatial_dim) score vectors
        """
        n_systems = batch.max().item() + 1
        t = torch.tensor(1.0, device=x.device)
        x_flat = x.view(n_systems, self.n_particles * self.spatial_dim)
        vel = self.model(t, x_flat)
        return vel.view(-1, self.spatial_dim)


class ImplicitScoreModel:
    """Train and hold an ISM score model.

    ISM loss: L = E[½||s_ϕ(x)||² + div(s_ϕ)(x)]
    This only requires samples from q_θ, not the density itself.

    After training, s_ϕ ≈ ∇log q_θ. If q_θ ≈ π, then s_ϕ ≈ -∇E.
    The diagnostic method checks this by computing cosine similarity.

    Usage:
        s_net = ScoreEGNN(n_particles=13, spatial_dim=3, hidden_nf=128, n_layers=4)
        ism = ImplicitScoreModel(s_net, dim=39, use_hutchinson=True, n_probes=8)
        ism.fit(samples, lr=3e-4, n_iters=5000)
        ism.diagnose(samples, grad_E)
    """

    def __init__(self, s_net, dim=None, use_hutchinson=False,
                 n_probes=8, grad_clip=5.0, device="cuda"):
        self.s_net = s_net.to(device)
        self.score_net = self.s_net   # Alias for ScoreInformedSteinCV compatibility
        self.dim = dim
        self.use_hutchinson = use_hutchinson
        self.n_probes = n_probes
        self.grad_clip = grad_clip
        self.device = torch.device(device)
        self.n_particles = s_net.n_particles
        self.spatial_dim = s_net.spatial_dim

    def _ism_loss_flat(self, x_flat_batch):
        """ISM loss computed in flat space (batch, dim).

        L = E[½||s_ϕ(x)||² + div(s_ϕ)(x)]

        For exact divergence: d backward passes (one per dimension).
        For Hutchinson: n_probes backward passes with random vectors.
        """
        x_req = x_flat_batch.detach().requires_grad_(True)
        dim = x_req.shape[1]

        t = torch.tensor(1.0, device=x_req.device)
        s_flat = self.s_net.model(t, x_req)  # (batch, dim) — calls EGNN_dynamics directly

        sq_norm = 0.5 * s_flat.pow(2).sum(dim=-1)  # (batch,)

        if self.use_hutchinson:
            div_est = torch.zeros(x_req.shape[0], device=x_req.device)
            for _ in range(self.n_probes):
                eps = torch.randn_like(x_req)
                s_eps = (s_flat * eps).sum()
                grad_s_eps = torch.autograd.grad(s_eps, x_req, create_graph=True)[0]
                div_est = div_est + (grad_s_eps * eps).sum(dim=-1)
            div_est = div_est / self.n_probes
        else:
            div_est = torch.zeros(x_req.shape[0], device=x_req.device)
            for d in range(dim):
                grad_d = torch.autograd.grad(
                    s_flat[:, d].sum(), x_req, create_graph=True, retain_graph=True
                )[0]
                div_est = div_est + grad_d[:, d]

        return (sq_norm + div_est).mean()

    def fit(self, samples, lr=3e-4, n_iters=5000, batch_size=256, verbose=True):
        """Train ISM on flat (N, dim) samples.

        Args:
            samples: (N, dim) terminal ASBS samples
            lr: learning rate (default 3e-4, lower than CV training)
            n_iters: number of training iterations
            batch_size: mini-batch size
            verbose: print loss every 500 steps
        """
        N = samples.shape[0]
        optimizer = torch.optim.Adam(self.s_net.parameters(), lr=lr)

        for step in range(n_iters):
            idx = torch.randint(0, N, (batch_size,))
            x_flat = samples[idx].to(self.device)

            loss = self._ism_loss_flat(x_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.s_net.parameters(), self.grad_clip
            )
            optimizer.step()

            if verbose and (step + 1) % 500 == 0:
                print(f"  [ScoreMatch] step {step+1}/{n_iters}  "
                      f"loss={loss.item():.6f}")

        self.s_net.eval()

    @torch.no_grad()
    def diagnose(self, samples, grad_E, batch_size=256):
        """Diagnose score model quality after training.

        Prints:
          - cosine_similarity(s_ϕ, -∇E): should be close to 1.0 if q_θ ≈ π
          - ||s_ϕ + ∇E||: should be close to 0.0 if q_θ ≈ π

        Args:
            samples: (N, dim) terminal samples
            grad_E: (N, dim) energy gradients at those samples
        """
        self.s_net.eval()
        N = samples.shape[0]
        cos_sims, diff_norms = [], []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_flat = samples[start:end].to(self.device)
            t = torch.tensor(1.0, device=self.device)
            s_phi = self.s_net.model(t, x_flat)

            neg_gE = -grad_E[start:end]
            cos = torch.nn.functional.cosine_similarity(
                s_phi, neg_gE, dim=-1,
            )
            diff = (s_phi + grad_E[start:end]).norm(dim=-1)
            cos_sims.append(cos)
            diff_norms.append(diff)

        cos_sims = torch.cat(cos_sims)
        diff_norms = torch.cat(diff_norms)
        print(f"  [ScoreDiag] cos_sim(s_phi, -∇E): "
              f"mean={cos_sims.mean():.4f}  std={cos_sims.std():.4f}")
        print(f"  [ScoreDiag] |s_phi + ∇E|:        "
              f"mean={diff_norms.mean():.4f}  std={diff_norms.std():.4f}")
