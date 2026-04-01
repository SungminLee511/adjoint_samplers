"""
enhancements/egnn_stein_cv.py

EGNN-based Stein control variate that exploits E(3) equivariance of molecular
systems.  Replaces the flat-vector MLP in neural_stein_cv.py with the same
EGNN_dynamics backbone used by the ASBS controller.

Training uses the *identical* train_neural_stein_cv() loop — only the model
object is swapped.
"""

import torch
import torch.nn as nn


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
