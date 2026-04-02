"""
enhancements/egnn_stein_cv.py

EGNN-based Stein control variate architectures for molecular systems.

Contains:
  - EGNNSteinCV:   Wraps base EGNN_dynamics (SML-original)
  - SteinEGNN:     Lightweight wrapper (KSH-style, no LayerNorm)
  - E_GCL_LN:      E(n)-equivariant GCL with LayerNorm in all MLPs
  - SteinEGNN_LN:  Standalone EGNN with LayerNorm (KSH-style, preferred)

All share the same flat API: forward(x_flat: (B, dim)) -> (B, dim)
"""

import torch
import torch.nn as nn
from torch import Tensor

from adjoint_samplers.utils.graph_utils import remove_mean


# ────────────────────────────────────────────────────────────
# SML-original: wrapper around base EGNN_dynamics
# ────────────────────────────────────────────────────────────

class EGNNSteinCV(nn.Module):
    """EGNN-based g_phi: R^d -> R^d for the Stein control variate.

    Exploits particle structure and E(3) equivariance.
    Output = coordinate displacement (equivariant vector field) with
    center-of-mass removed (done inside EGNN_dynamics.forward).

    Requirements (same as NeuralSteinCV):
      - Output dimension = input dimension
      - Twice differentiable (EGNN uses SiLU by default — OK)
    """

    def __init__(
        self,
        n_particles: int,
        spatial_dim: int,
        hidden_nf: int = 64,
        n_layers: int = 4,
        recurrent: bool = True,
        attention: bool = True,
        tanh: bool = True,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim

        # Import from original codebase
        from adjoint_samplers.components.model import EGNN_dynamics

        self.egnn = EGNN_dynamics(
            n_particles=n_particles,
            spatial_dim=spatial_dim,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            condition_time=False,   # No time conditioning — Stein CV at terminal time only
            act_fn=torch.nn.SiLU(),
        )

        # Scale output small initially so g ≈ 0 at start
        # => A_p g ≈ 0 => h ≈ f, no harm initially
        self.output_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_particles * spatial_dim) flat coordinates
        Returns:
            g: (B, n_particles * spatial_dim) equivariant vector field
        """
        # EGNN_dynamics.forward expects (t, xs) — pass dummy t=0
        # condition_time=False means t is ignored in the node features,
        # but the method signature still requires it.
        dummy_t = torch.zeros(1, device=x.device)
        g = self.egnn(dummy_t, x)
        return g * self.output_scale


# ────────────────────────────────────────────────────────────
# KSH-style: lightweight EGNN wrapper (no LayerNorm)
# ────────────────────────────────────────────────────────────

class SteinEGNN(nn.Module):
    """EGNN-based vector field g_ψ: R^dim → R^dim in flat space.

    Thin wrapper around EGNN_dynamics with condition_time=False.
    Unlike EGNNSteinCV, this has no learnable output_scale.
    """

    def __init__(self, n_particles=4, spatial_dim=2, hidden_nf=128, n_layers=5):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim

        from adjoint_samplers.components.model import EGNN_dynamics

        self.model = EGNN_dynamics(
            n_particles=n_particles, spatial_dim=spatial_dim,
            hidden_nf=hidden_nf, n_layers=n_layers,
            act_fn=nn.SiLU(), recurrent=True, tanh=False,
            attention=True, condition_time=False, agg="sum",
        )

    def forward(self, x_flat: Tensor) -> Tensor:
        """x_flat: (B, dim) → (B, dim)"""
        t = torch.tensor(0.0, device=x_flat.device)
        return self.model(t, x_flat)


# ────────────────────────────────────────────────────────────
# KSH-style: E_GCL with LayerNorm (standalone implementation)
# ────────────────────────────────────────────────────────────

def _unsorted_segment_sum(data, segment_ids, num_segments):
    """Scatter-add: aggregate data by segment_ids into num_segments buckets."""
    result = data.new_full((num_segments, data.size(1)), 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


class E_GCL_LN(nn.Module):
    """E(n)-equivariant Graph Convolutional Layer with LayerNorm.

    Identical structure to E_GCL from the base ASBS code, but adds
    nn.LayerNorm after each linear layer in edge/node/coord MLPs.
    This improves training stability for Stein CV optimization.
    """

    def __init__(self, hidden_nf, edges_in_d=1, act_fn=nn.SiLU(),
                 recurrent=True, attention=False, tanh=False,
                 coords_range=1.0, agg="sum"):
        super().__init__()
        self.recurrent = recurrent
        self.attention = attention
        self.tanh = tanh
        self.agg_type = agg

        input_edge = hidden_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + 1 + edges_in_d, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )

        coord_last = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(coord_last.weight, gain=0.001)
        coord_layers = [
            nn.Linear(hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            coord_last,
        ]
        if self.tanh:
            coord_layers.append(nn.Tanh())
            self.coords_range = coords_range
        self.coord_mlp = nn.Sequential(*coord_layers)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def forward(self, h, edge_index, coord, edge_attr=None):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = (coord_diff ** 2).sum(1, keepdim=True)
        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff / (norm + 1)  # Normalized direction + damping

        inp = torch.cat([h[row], h[col], radial, edge_attr], dim=1) if edge_attr is not None \
            else torch.cat([h[row], h[col], radial], dim=1)
        edge_feat = self.edge_mlp(inp)
        if self.attention:
            edge_feat = edge_feat * self.att_mlp(edge_feat)

        if self.tanh:
            trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(edge_feat)

        agg_coord = _unsorted_segment_sum(trans, row, coord.size(0))
        coord = coord + agg_coord

        agg_node = _unsorted_segment_sum(edge_feat, row, h.size(0))
        out = self.node_mlp(torch.cat([h, agg_node], dim=1))
        if self.recurrent:
            out = h + out
        return out, coord


class SteinEGNN_LN(nn.Module):
    """EGNN with LayerNorm — standalone implementation for Stein CV.

    Key differences from EGNNSteinCV / SteinEGNN:
      - Uses E_GCL_LN layers (LayerNorm in all MLPs) for training stability
      - Standalone implementation (does not wrap EGNN_dynamics)
      - Normalizes coord_diff: coord_diff / (norm + 1)
      - Edge attributes computed once before all layers (not recomputed)
      - Caches edge indices per (n_batch, device) for efficiency
      - No output_scale parameter (starts with small coord init instead)

    Same flat API: forward(x_flat: (B, dim)) -> (B, dim)
    """

    def __init__(self, n_particles=4, spatial_dim=2, hidden_nf=128,
                 n_layers=5, tanh=False, condition_time=False):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self._condition_time = condition_time

        self.embedding = nn.Linear(1, hidden_nf)
        act = nn.SiLU()
        self.layers = nn.ModuleList([
            E_GCL_LN(hidden_nf, edges_in_d=1, act_fn=act,
                      recurrent=True, attention=True, tanh=tanh, agg="sum")
            for _ in range(n_layers)
        ])

        self._edges_cache = {}

    def _get_edges(self, n_batch, device):
        """Build fully-connected edge indices for n_batch particle systems.
        Cached per (n_batch, device) to avoid recomputation."""
        key = (n_batch, device)
        if key not in self._edges_cache:
            n = self.n_particles
            rows, cols = [], []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        rows.append(i)
                        cols.append(j)
            r = torch.tensor(rows, dtype=torch.long, device=device)
            c = torch.tensor(cols, dtype=torch.long, device=device)
            r_all = torch.cat([r + i * n for i in range(n_batch)])
            c_all = torch.cat([c + i * n for i in range(n_batch)])
            self._edges_cache[key] = [r_all, c_all]
        return self._edges_cache[key]

    def forward(self, x_flat: Tensor) -> Tensor:
        """
        Args:
            x_flat: (B, n_particles * spatial_dim) flat coordinates
        Returns:
            (B, n_particles * spatial_dim) equivariant vector field, COM-free
        """
        B = x_flat.shape[0]
        n, d = self.n_particles, self.spatial_dim
        device = x_flat.device

        coord = x_flat.view(B * n, d).clone()
        h = torch.ones(B * n, 1, device=device)
        if self._condition_time:
            h = h * 0.0
        h = self.embedding(h)

        edges = self._get_edges(B, device)
        edge_attr = ((coord[edges[0]] - coord[edges[1]]) ** 2).sum(1, keepdim=True)

        x_init = coord.clone()
        for layer in self.layers:
            h, coord = layer(h, edges, coord, edge_attr)

        vel = coord - x_init
        vel = vel.view(B, n, d)
        vel = remove_mean(vel, n, d)
        return vel.view(B, n * d)
