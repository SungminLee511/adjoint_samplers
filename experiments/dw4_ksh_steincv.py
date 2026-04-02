#!/usr/bin/env python
"""DW4 KSH-style SteinCV evaluation.

Protocol:
  - 3 seeds, each generating fresh ASBS samples
  - 2 observables: energy, interatomic distance
  - Per-observable: train SteinEGNN_LN + SteinBiasCorrector (variance loss)
  - Evaluate: naive vs CV via bootstrap (2000 resamples)
  - Report: mean, bias, var, MSE, reduction ratios

Usage:
    /root/miniconda3/envs/Sampling_env/bin/python -u experiments/dw4_ksh_steincv.py \
        --checkpoint <path_to_checkpoint.pt> \
        --n_samples 5000 --cv_iters 10000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path

from adjoint_samplers.energies.double_well_energy import DoubleWellEnergy
from adjoint_samplers.components.sde import GraphVESDE, ControlledSDE, sdeint
from adjoint_samplers.components.model import EGNN_dynamics
from adjoint_samplers.utils.dist_utils import CenteredParticlesHarmonic
from adjoint_samplers.utils.graph_utils import remove_mean
from adjoint_samplers.utils.eval_utils import interatomic_dist

from enhancements.egnn_stein_cv import SteinEGNN_LN
from enhancements.variance_stein_cv import SteinBiasCorrector, stein_operator_on_net

cudnn.benchmark = True

DW4_CFG = dict(
    dim=8, n_particles=4, spatial_dim=2,
    source_scale=2.0, sigma_max=1.0, sigma_min=0.001, nfe=200,
    hidden_nf=128, n_layers=5, eval_batch_size=2000,
    cv_hidden_nf=64, cv_n_layers=4,
)


def build_sampler(cfg, ckpt_path, device):
    """Build energy, SDE, and load reference. Returns reusable objects."""
    energy = DoubleWellEnergy(dim=cfg["dim"], n_particles=cfg["n_particles"],
                              device=device)
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
        act_fn=nn.SiLU(), recurrent=True, tanh=True,
        attention=True, condition_time=True, agg="sum",
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt["controller"])
    epoch = ckpt.get("epoch", "?")

    sde = ControlledSDE(ref_sde, controller).to(device)
    sde.eval()

    root = Path(__file__).parent.parent
    ref = np.load(root / "data" / "test_split_DW4.npy", allow_pickle=True)
    ref = remove_mean(torch.tensor(ref, dtype=torch.float32),
                      cfg["n_particles"], cfg["spatial_dim"]).to(device)
    return energy, source, sde, ref, epoch


def generate_samples(source, sde, cfg, n_samples, device):
    """Generate fresh ASBS samples (stochastic, depends on current RNG)."""
    chunks, n = [], 0
    with torch.no_grad():
        while n < n_samples:
            B = min(cfg["eval_batch_size"], n_samples - n)
            x0 = source.sample([B]).to(device)
            ts = torch.linspace(0, 1, cfg["nfe"], device=device)
            _, x1 = sdeint(sde, x0, ts, only_boundary=True)
            chunks.append(x1)
            n += B
    return torch.cat(chunks)[:n_samples]


def evaluate(g_net, samples, grad_E, f_vals, n_boot=2000):
    """Return dict with naive/cv stats via bootstrap."""
    g_net.eval()
    score = -grad_E
    N = samples.shape[0]
    h_all = []
    for s in range(0, N, 500):
        e = min(s + 500, N)
        with torch.enable_grad():
            Tg = stein_operator_on_net(g_net, samples[s:e], score[s:e])
        h_all.append((f_vals[s:e] + Tg).detach())
    h_all = torch.cat(h_all)

    cv_mean = h_all.mean().item()
    naive_mean = f_vals.mean().item()

    boots_naive, boots_cv = [], []
    for _ in range(n_boot):
        idx = torch.randint(0, N, (N,), device=samples.device)
        boots_naive.append(f_vals[idx].mean().item())
        boots_cv.append(h_all[idx].mean().item())

    return dict(
        naive_mean=naive_mean,
        cv_mean=cv_mean,
        naive_var=np.var(boots_naive),
        cv_var=np.var(boots_cv),
    )


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DW4_CFG
    seeds = [0, 1, 2]

    print("=" * 70)
    print("  DW4 KSH-Style SteinCV Evaluation")
    print(f"  Model: SteinEGNN_LN  h={cfg['cv_hidden_nf']}  L={cfg['cv_n_layers']}")
    print(f"  Val split: 0.2,  Patience: 6,  Cosine LR,  Iters: {args.cv_iters}")
    print(f"  Seeds: {seeds}  (each seed re-samples from ASBS)")
    print("=" * 70)

    print("\nBuilding ASBS sampler ...")
    energy, source, sde, ref, epoch = build_sampler(cfg, args.checkpoint, device)
    print(f"Loaded epoch {epoch}")

    ref_e = energy.eval(ref).mean().item()
    ref_d = interatomic_dist(ref, cfg["n_particles"],
                             cfg["spatial_dim"]).mean(-1).mean().item()
    print(f"Ground truth:  energy={ref_e:.4f}   dist={ref_d:.4f}")

    all_results = {"energy": [], "dist": []}

    for seed in seeds:
        print(f"\n{'─' * 70}")
        print(f"  Seed {seed}")
        print(f"{'─' * 70}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

        samples = generate_samples(source, sde, cfg, args.n_samples, device)
        grad_E = energy.grad_E(samples)
        e_vals = energy.eval(samples)
        d_vals = interatomic_dist(samples, cfg["n_particles"],
                                  cfg["spatial_dim"]).mean(-1)

        naive_e_bias = abs(e_vals.mean().item() - ref_e)
        naive_d_bias = abs(d_vals.mean().item() - ref_d)
        print(f"  Naive: energy={e_vals.mean():.4f} (bias={naive_e_bias:.5f})  "
              f"dist={d_vals.mean():.4f} (bias={naive_d_bias:.5f})")

        for obs_name, f_vals, gt in [("energy", e_vals, ref_e),
                                      ("dist", d_vals, ref_d)]:
            cv_seed = seed * 100 + (0 if obs_name == "energy" else 1)
            torch.manual_seed(cv_seed)
            np.random.seed(cv_seed)
            torch.cuda.manual_seed(cv_seed)

            g_net = SteinEGNN_LN(
                n_particles=cfg["n_particles"], spatial_dim=cfg["spatial_dim"],
                hidden_nf=cfg["cv_hidden_nf"], n_layers=cfg["cv_n_layers"],
                tanh=False,
            ).to(device)
            corrector = SteinBiasCorrector(g_net=g_net)

            print(f"\n  [{obs_name}] Training SteinCV ...")
            corrector.fit(
                samples, grad_E, f_vals,
                lr=1e-3, n_iters=args.cv_iters, batch_size=2500,
                cosine_lr=True,
            )
            res = evaluate(g_net, samples, grad_E, f_vals)
            res["naive_bias"] = abs(res["naive_mean"] - gt)
            res["cv_bias"] = abs(res["cv_mean"] - gt)
            res["naive_mse"] = res["naive_bias"]**2 + res["naive_var"]
            res["cv_mse"] = res["cv_bias"]**2 + res["cv_var"]
            res["mse_reduction"] = res["naive_mse"] / max(res["cv_mse"], 1e-15)
            res["bias_reduction"] = res["naive_bias"] / max(res["cv_bias"], 1e-10)

            print(f"    Naive: mean={res['naive_mean']:.5f}  "
                  f"bias={res['naive_bias']:.5f}  var={res['naive_var']:.2e}  "
                  f"MSE={res['naive_mse']:.2e}")
            print(f"    CV:    mean={res['cv_mean']:.5f}  "
                  f"bias={res['cv_bias']:.5f}  var={res['cv_var']:.2e}  "
                  f"MSE={res['cv_mse']:.2e}")
            print(f"    MSE reduction: {res['mse_reduction']:.2f}x  "
                  f"Bias reduction: {res['bias_reduction']:.2f}x")
            all_results[obs_name].append(res)

            del g_net, corrector
            torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY (mean ± std over {len(seeds)} seeds)")
    print(f"{'=' * 70}")

    for obs_name in ["energy", "dist"]:
        results = all_results[obs_name]
        print(f"\n  ── {obs_name} ──")
        for key in ["naive_mean", "naive_bias", "naive_mse",
                     "cv_mean", "cv_bias", "cv_var", "cv_mse",
                     "mse_reduction", "bias_reduction"]:
            vals = [r[key] for r in results]
            m, s = np.mean(vals), np.std(vals)
            if "mse" in key.lower() and "reduction" not in key:
                print(f"    {key:18s}: {m:.2e} ± {s:.2e}")
            else:
                print(f"    {key:18s}: {m:.5f} ± {s:.5f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--cv_iters", type=int, default=10000)
    run(p.parse_args())
