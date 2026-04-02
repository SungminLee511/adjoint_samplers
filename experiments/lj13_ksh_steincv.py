#!/usr/bin/env python
"""
LJ-13: KSH-style ASBS + Stein Bias Correction Experiment.

Protocol:
  1. Load ASBS checkpoint, generate samples
  2. Phase 1: Train ISM score model (ScoreEGNN, s_ϕ ≈ ∇log q_θ)
  3. For each observable (energy, interatomic_dist):
     a. Phase 2a: Train basic SteinBiasCorrector (SteinEGNN_LN, variance loss)
     b. Phase 2b: Train ScoreInformedSteinCV (decomposition: g = α·g_init + g_res)
     c. Evaluate: 200 trials with subsampling, compare 3 methods
  4. Save results JSON

Usage:
    /root/miniconda3/envs/Sampling_env/bin/python -u experiments/lj13_ksh_steincv.py \
        --checkpoint <path_to_checkpoint.pt> \
        --n_samples 2000 --score_iters 5000 --cv_iters 5000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from adjoint_samplers.energies.lennard_jones_energy import LennardJonesEnergy
from adjoint_samplers.components.sde import GraphVESDE, ControlledSDE, sdeint
from adjoint_samplers.components.model import EGNN_dynamics
from adjoint_samplers.utils.dist_utils import CenteredParticlesHarmonic
from adjoint_samplers.utils.graph_utils import remove_mean
from adjoint_samplers.utils.eval_utils import interatomic_dist

from enhancements.egnn_stein_cv import SteinEGNN_LN
from enhancements.variance_stein_cv import (
    SteinBiasCorrector,
    ScoreInformedSteinCV,
    stein_operator_on_net,
)
from enhancements.score_matching import ImplicitScoreModel, ScoreEGNN

cudnn.benchmark = True

LJ13_CFG = dict(
    dim=39, n_particles=13, spatial_dim=3,
    source_scale=2.0, sigma_max=1.0, sigma_min=0.001, nfe=1000,
    hidden_nf=128, n_layers=5,
    eval_batch_size=500,
)


def build_asbs(cfg, device):
    """Build ASBS components (energy, source, SDE, controller)."""
    energy = LennardJonesEnergy(
        dim=cfg["dim"], n_particles=cfg["n_particles"], device=device
    )
    source = CenteredParticlesHarmonic(
        n_particles=cfg["n_particles"], spatial_dim=cfg["spatial_dim"],
        scale=cfg["source_scale"], device=device,
    )
    ref_sde = GraphVESDE(
        n_particles=cfg["n_particles"], spatial_dim=cfg["spatial_dim"],
        sigma_max=cfg["sigma_max"], sigma_min=cfg["sigma_min"],
    ).to(device)

    controller = EGNN_dynamics(
        n_particles=cfg["n_particles"], spatial_dim=cfg["spatial_dim"],
        hidden_nf=cfg["hidden_nf"], n_layers=cfg["n_layers"],
        act_fn=torch.nn.SiLU(), recurrent=True, tanh=True,
        attention=True, condition_time=True, agg="sum",
    ).to(device)

    sde = ControlledSDE(ref_sde, controller).to(device)
    return dict(energy=energy, source=source, sde=sde, controller=controller)


def load_checkpoint(comp, ckpt_path, device):
    """Load controller weights from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    comp["controller"].load_state_dict(ckpt["controller"])
    return ckpt.get("epoch", "?")


@torch.no_grad()
def generate_samples(comp, cfg, n_samples, device):
    """Generate fresh ASBS terminal samples."""
    sde, source = comp["sde"], comp["source"]
    sde.eval()
    chunks = []
    n = 0
    while n < n_samples:
        B = min(cfg["eval_batch_size"], n_samples - n)
        x0 = source.sample([B]).to(device)
        ts = torch.linspace(0, 1, cfg["nfe"], device=device)
        _, x1 = sdeint(sde, x0, ts, only_boundary=True)
        chunks.append(x1)
        n += B
        print(f"  Generated {B} (total {n}/{n_samples})")
    return torch.cat(chunks)[:n_samples]


def load_reference(cfg, device):
    """Load reference samples for ground truth."""
    root = Path(__file__).parent.parent
    ref = np.load(root / "data" / "test_split_LJ13-1000.npy", allow_pickle=True)
    ref = remove_mean(
        torch.tensor(ref, dtype=torch.float32),
        cfg["n_particles"], cfg["spatial_dim"],
    )
    return ref.to(device)


def compute_observables(samples, energy, cfg):
    """Compute energy and interatomic distance observables."""
    e = energy.eval(samples)
    d = interatomic_dist(
        samples, cfg["n_particles"], cfg["spatial_dim"]
    ).mean(-1)
    return {"energy": e, "interatomic_dist": d}


def metrics(ests, gt):
    """Compute bias, variance, MSE from a list of estimates."""
    t = torch.tensor(ests)
    mean = t.mean().item()
    bias = abs(mean - gt)
    var = t.var().item()
    mse = bias ** 2 + var
    return {"mean": mean, "bias": bias, "var": var, "mse": mse}


def run_experiment(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = LJ13_CFG
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building ASBS ...")
    comp = build_asbs(cfg, device)
    epoch = load_checkpoint(comp, args.checkpoint, device)
    print(f"Loaded checkpoint (epoch {epoch})")

    energy = comp["energy"]

    print(f"Generating {args.n_samples} samples (nfe={cfg['nfe']}) ...")
    samples = generate_samples(comp, cfg, args.n_samples, device)

    print("Loading reference ...")
    ref = load_reference(cfg, device)

    ref_obs = compute_observables(ref, energy, cfg)
    I_true = {k: v.mean().item() for k, v in ref_obs.items()}
    print(f"  Ground truth:  energy={I_true['energy']:.4f}  "
          f"dist={I_true['interatomic_dist']:.4f}")

    print("Computing ∇E and observables ...")
    grad_E = energy.grad_E(samples)
    obs = compute_observables(samples, energy, cfg)

    # ── Phase 1: Implicit Score Matching ──
    print("\n" + "=" * 60)
    print("  Phase 1: Implicit Score Matching  "
          "(EGNN s_ϕ ≈ ∇log p^{u_θ})")
    print("=" * 60)
    s_net = ScoreEGNN(
        n_particles=cfg["n_particles"], spatial_dim=cfg["spatial_dim"],
        hidden_nf=128, n_layers=4,
    ).to(device)
    score_model = ImplicitScoreModel(
        s_net=s_net, dim=cfg["dim"], use_hutchinson=True, n_probes=8,
    )
    score_model.fit(
        samples, lr=3e-4, n_iters=args.score_iters,
        batch_size=256, verbose=True,
    )
    # Diagnose score quality
    score_model.diagnose(samples, grad_E)

    all_results = {}

    # ── Phase 2: Per-observable Stein CV ──
    for obs_name, f_vals in obs.items():
        gt = I_true[obs_name]

        print(f"\n{'=' * 60}")
        print(f"  Observable: {obs_name}    I_true = {gt:.6f}")
        print(f"{'=' * 60}")

        # Phase 2a: Basic Stein CV (EGNN+LN, variance loss)
        print("\n  [Phase 2a] Stein CV  (EGNN+LN g_ψ, variance loss)")
        g_egnn = SteinEGNN_LN(
            n_particles=cfg["n_particles"],
            spatial_dim=cfg["spatial_dim"],
            hidden_nf=128, n_layers=5,
        ).to(device)
        stein_basic = SteinBiasCorrector(
            g_net=g_egnn, use_hutchinson=True, n_probes=4,
        )
        stein_basic.fit(
            samples, grad_E, f_vals,
            lr=1e-3, n_iters=args.cv_iters,
            batch_size=256, verbose=True,
        )

        # Phase 2b: Score-informed Stein CV
        print("\n  [Phase 2b] Score-informed Stein CV  (EGNN+LN g_res)")
        g_egnn_res = SteinEGNN_LN(
            n_particles=cfg["n_particles"],
            spatial_dim=cfg["spatial_dim"],
            hidden_nf=128, n_layers=5,
        ).to(device)
        stein_score = ScoreInformedSteinCV(
            g_residual_net=g_egnn_res, score_model=score_model,
            use_hutchinson=True, n_probes=4,
        )
        stein_score.fit(
            samples, energy, f_vals,
            lr=1e-3, n_iters=args.cv_iters,
            batch_size=256, verbose=True,
        )

        print("  Precomputing corrected h for all samples ...")
        h_score_all = stein_score.eval_all(
            samples, grad_E, f_vals, stein_score._Tg_init,
        )
        alpha_final = stein_score.alpha.item()
        print(f"    Final α = {alpha_final:.2e}")

        # ── Evaluation: subsample trials ──
        print(f"  Evaluating ({args.n_trials} trials, "
              f"subsample={args.subsample_size}) ...")

        naive_ests, basic_ests, score_ests = [], [], []
        N = samples.shape[0]

        for _ in range(args.n_trials):
            idx = torch.randperm(N, device=device)[:args.subsample_size]

            sub_f = f_vals[idx]
            naive_ests.append(sub_f.mean().item())

            est_b, _ = stein_basic.estimate(
                samples[idx], grad_E[idx], sub_f,
            )
            basic_ests.append(est_b)

            score_ests.append(h_score_all[idx].mean().item())

        m_naive = metrics(naive_ests, gt)
        m_basic = metrics(basic_ests, gt)
        m_score = metrics(score_ests, gt)

        print(f"\n  {'Method':<25} {'Mean':>10} {'|Bias|':>10} "
              f"{'Var':>12} {'MSE':>12}")
        print(f"  {'-' * 69}")
        for name, m in [("Naive ASBS", m_naive),
                        ("Stein CV", m_basic),
                        ("Score+Stein CV", m_score)]:
            print(f"  {name:<25} {m['mean']:>10.5f} "
                  f"{m['bias']:>10.5f} "
                  f"{m['var']:>12.2e} {m['mse']:>12.2e}")

        if m_naive["bias"] > 1e-10:
            print(f"\n  Bias reduction vs Naive:")
            print(f"    SteinCV:        "
                  f"{m_naive['bias']/max(m_basic['bias'],1e-10):.2f}x")
            print(f"    Score+SteinCV:  "
                  f"{m_naive['bias']/max(m_score['bias'],1e-10):.2f}x")
        if m_naive["mse"] > 1e-15:
            print(f"  MSE reduction vs Naive:")
            print(f"    SteinCV:        "
                  f"{m_naive['mse']/max(m_basic['mse'],1e-15):.2f}x")
            print(f"    Score+SteinCV:  "
                  f"{m_naive['mse']/max(m_score['mse'],1e-15):.2f}x")

        all_results[obs_name] = {
            "I_true": gt,
            "naive": m_naive,
            "stein_cv": m_basic,
            "score_stein_cv": m_score,
        }

        # Cleanup
        del g_egnn, stein_basic, g_egnn_res, stein_score, h_score_all
        torch.cuda.empty_cache()

    result_path = out_dir / "results.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--subsample_size", type=int, default=200)
    parser.add_argument("--score_iters", type=int, default=5000)
    parser.add_argument("--cv_iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="results/lj13_ksh_steincv")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
