"""
Tests for enhancements/egnn_stein_cv.py

Verifies:
  1. EGNNSteinCV instantiation and forward pass shape
  2. Output is equivariant (translation, rotation)
  3. COM removal (output sums to zero per sample)
  4. Small initial output (output_scale = 0.01)
  5. Compatible with train_neural_stein_cv (can be passed as g_model)
  6. Stein operator computes through EGNN graph (autograd works)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math

from enhancements.egnn_stein_cv import EGNNSteinCV
from enhancements.neural_stein_cv import compute_stein_operator


def test_forward_shape():
    """Output shape matches input shape."""
    n_particles, spatial_dim = 4, 2
    model = EGNNSteinCV(n_particles=n_particles, spatial_dim=spatial_dim,
                        hidden_nf=32, n_layers=2)
    B, d = 16, n_particles * spatial_dim
    x = torch.randn(B, d)
    g = model(x)
    assert g.shape == (B, d), f"Expected ({B}, {d}), got {g.shape}"
    print("PASS: test_forward_shape")


def test_small_initial_output():
    """Output starts small due to output_scale=0.01."""
    model = EGNNSteinCV(n_particles=4, spatial_dim=2,
                        hidden_nf=32, n_layers=2)
    x = torch.randn(32, 8)
    g = model(x)
    rms = g.pow(2).mean().sqrt().item()
    # EGNN coord_mlp has xavier init with gain=0.001, times output_scale=0.01
    # Should be very small
    assert rms < 0.5, f"Initial output RMS = {rms:.4f}, expected < 0.5"
    print(f"PASS: test_small_initial_output (RMS = {rms:.6f})")


def test_com_removal():
    """EGNN_dynamics calls remove_mean, so output should be COM-free."""
    model = EGNNSteinCV(n_particles=4, spatial_dim=2,
                        hidden_nf=32, n_layers=2)
    x = torch.randn(16, 8)
    g = model(x)

    # Reshape to (B, n_particles, spatial_dim) and check COM = 0
    g_particles = g.view(-1, 4, 2)
    com = g_particles.mean(dim=1)  # (B, 2) — should be ~0
    max_com = com.abs().max().item()
    assert max_com < 1e-5, f"COM not removed: max |COM| = {max_com:.2e}"
    print(f"PASS: test_com_removal (max |COM| = {max_com:.2e})")


def test_translation_equivariance():
    """g(x + t) = g(x) for translation t applied uniformly to all particles."""
    torch.manual_seed(42)
    model = EGNNSteinCV(n_particles=4, spatial_dim=2,
                        hidden_nf=32, n_layers=2)
    model.eval()

    B, n_p, s_d = 8, 4, 2
    x = torch.randn(B, n_p * s_d)

    # Uniform translation: shift all particles by the same vector
    shift = torch.randn(1, s_d)  # (1, 2)
    shift_flat = shift.repeat(1, n_p)  # (1, 8) — same shift for each particle

    with torch.no_grad():
        g_orig = model(x)
        g_shifted = model(x + shift_flat)

    # Equivariant vector field: g(x+t) should equal g(x)
    # (translation equivariance for the displacement = translation invariance)
    diff = (g_orig - g_shifted).abs().max().item()
    assert diff < 1e-4, f"Translation equivariance violated: max diff = {diff:.2e}"
    print(f"PASS: test_translation_equivariance (max diff = {diff:.2e})")


def test_rotation_equivariance():
    """g(Rx) = R g(x) for rotation R applied to all particles."""
    torch.manual_seed(42)
    model = EGNNSteinCV(n_particles=4, spatial_dim=2,
                        hidden_nf=32, n_layers=2)
    model.eval()

    B, n_p, s_d = 8, 4, 2

    # 2D rotation matrix (45 degrees)
    theta = math.pi / 4
    R = torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)],
    ])

    # COM-free input (important for EGNN)
    x = torch.randn(B, n_p, s_d)
    x = x - x.mean(dim=1, keepdim=True)  # remove COM
    x_flat = x.view(B, n_p * s_d)

    # Rotate all particles
    x_rot = (x @ R.T).view(B, n_p * s_d)

    with torch.no_grad():
        g_orig = model(x_flat).view(B, n_p, s_d)
        g_rot_input = model(x_rot).view(B, n_p, s_d)

    # Expected: g(Rx) = R g(x)
    g_expected = (g_orig @ R.T)
    diff = (g_rot_input - g_expected).abs().max().item()
    assert diff < 1e-3, f"Rotation equivariance violated: max diff = {diff:.2e}"
    print(f"PASS: test_rotation_equivariance (max diff = {diff:.2e})")


def test_autograd_stein_operator():
    """Stein operator can be computed through EGNN (autograd works)."""
    model = EGNNSteinCV(n_particles=4, spatial_dim=2,
                        hidden_nf=32, n_layers=2)
    B, d = 8, 8
    x = torch.randn(B, d, requires_grad=True)
    scores = torch.randn(B, d)

    g = model(x)
    Apg = compute_stein_operator(g, x, scores, model, hutchinson_samples=1)

    assert Apg.shape == (B,), f"Expected ({B},), got {Apg.shape}"
    # Check gradients flow back through g_model params
    loss = Apg.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "No gradients flowed to model parameters"
    print("PASS: test_autograd_stein_operator")


def test_compatible_with_train_api():
    """EGNNSteinCV has the right interface for train_neural_stein_cv."""
    model = EGNNSteinCV(n_particles=4, spatial_dim=2,
                        hidden_nf=32, n_layers=2)

    # Check it's an nn.Module with forward(x) -> (B, d) signature
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, 'forward')
    assert hasattr(model, 'parameters')
    assert model.dim == 8

    # Quick check: parameters are optimizable
    params = list(model.parameters())
    assert len(params) > 0
    n_params = sum(p.numel() for p in params)
    print(f"PASS: test_compatible_with_train_api ({n_params} parameters)")


if __name__ == "__main__":
    test_forward_shape()
    test_small_initial_output()
    test_com_removal()
    test_translation_equivariance()
    test_rotation_equivariance()
    test_autograd_stein_operator()
    test_compatible_with_train_api()
    print("\n=== ALL EGNN STEIN CV TESTS PASSED ===")
