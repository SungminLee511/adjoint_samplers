"""Phase 1: Compute ground truth mean energy from LJ13 reference samples."""
import numpy as np
import torch

# Load reference samples
ref_samples = np.load("data/test_split_LJ13-1000.npy")
print(f"Reference samples shape: {ref_samples.shape}")
print(f"Number of configs: {ref_samples.shape[0]}")

# Convert to torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
samples = torch.tensor(ref_samples, dtype=torch.float32, device=device)

# Reshape if needed: should be (N, 39) for 13 particles x 3D
if samples.ndim == 3:
    N = samples.shape[0]
    samples = samples.reshape(N, -1)
    print(f"Reshaped to: {samples.shape}")

# Instantiate LJ energy
from adjoint_samplers.energies.lennard_jones_energy import LennardJonesEnergy
energy = LennardJonesEnergy(dim=39, n_particles=13)

# Compute energies
with torch.no_grad():
    energies = energy.eval(samples)

print(f"\n{'='*50}")
print(f"LJ13 Ground Truth Energy Statistics")
print(f"{'='*50}")
print(f"Mean energy:   {energies.mean().item():.6f}")
print(f"Std energy:    {energies.std().item():.6f}")
print(f"Median energy: {energies.median().item():.6f}")
print(f"Min energy:    {energies.min().item():.6f}")
print(f"Max energy:    {energies.max().item():.6f}")
print(f"25th pctile:   {torch.quantile(energies, 0.25).item():.6f}")
print(f"75th pctile:   {torch.quantile(energies, 0.75).item():.6f}")
print(f"N samples:     {len(energies)}")
print(f"{'='*50}")
