#!/usr/bin/env python
"""LJ13 Phase 6: KSH-style full pipeline — ISM → Basic CV → Score-Informed CV.

Optimized training loop (Hutchinson everywhere, subsample val).

Usage:
    /root/miniconda3/envs/Sampling_env/bin/python -u experiments/run_lj13_phase6.py
"""
import sys, os, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path

from adjoint_samplers.energies.lennard_jones_energy import LennardJonesEnergy
from adjoint_samplers.components.sde import GraphVESDE, ControlledSDE, sdeint
from adjoint_samplers.components.model import EGNN_dynamics
from adjoint_samplers.utils.dist_utils import CenteredParticlesHarmonic
from adjoint_samplers.utils.graph_utils import remove_mean
from adjoint_samplers.utils.eval_utils import interatomic_dist

from enhancements.egnn_stein_cv import SteinEGNN_LN
from enhancements.variance_stein_cv import stein_operator_on_net, ScoreInformedSteinCV
from enhancements.score_matching import ScoreEGNN, ImplicitScoreModel

cudnn.benchmark = True

CKPT = "/home/RESEARCH/adjoint_samplers/results/local/2026.03.31/212242/checkpoints/checkpoint_latest.pt"
N_PARTICLES, SPATIAL_DIM = 13, 3
DIM = N_PARTICLES * SPATIAL_DIM  # 39
N_SAMPLES = 5000
SEEDS = [0, 1, 2]


def build_sampler(device):
    energy = LennardJonesEnergy(dim=DIM, n_particles=N_PARTICLES, device=device)
    source = CenteredParticlesHarmonic(
        n_particles=N_PARTICLES, spatial_dim=SPATIAL_DIM,
        scale=2.0, device=device)
    ref_sde = GraphVESDE(
        n_particles=N_PARTICLES, spatial_dim=SPATIAL_DIM,
        sigma_max=1.0, sigma_min=0.001).to(device)
    controller = EGNN_dynamics(
        n_particles=N_PARTICLES, spatial_dim=SPATIAL_DIM,
        hidden_nf=128, n_layers=5,
        act_fn=nn.SiLU(), recurrent=True, tanh=True,
        attention=True, condition_time=True, agg="sum").to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt["controller"])
    sde = ControlledSDE(ref_sde, controller).to(device)
    sde.eval()
    root = Path(__file__).parent.parent
    ref = np.load(root / "data" / "test_split_LJ13-1000.npy", allow_pickle=True)
    ref = remove_mean(torch.tensor(ref, dtype=torch.float32),
                      N_PARTICLES, SPATIAL_DIM).to(device)
    return energy, source, sde, ref, ckpt.get("epoch", "?")


def gen_samples(source, sde, n, device):
    chunks, done = [], 0
    with torch.no_grad():
        while done < n:
            B = min(500, n - done)  # smaller batch for LJ13 (NFE=1000)
            x0 = source.sample([B]).to(device)
            ts = torch.linspace(0, 1, 1000, device=device)
            _, x1 = sdeint(sde, x0, ts, only_boundary=True)
            chunks.append(x1)
            done += B
    return torch.cat(chunks)[:n]


def train_steincv_fast(g_net, train_x, train_s, train_f,
                       val_x, val_s, val_f,
                       n_iters=2000, batch_size=256, lr=1e-3,
                       patience=5, eval_every=200):
    """Fast training with Hutchinson div."""
    optimizer = torch.optim.Adam(g_net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=lr*0.01)

    N_train = train_x.shape[0]
    best_var = float("inf")
    best_state = None
    best_step = 0
    no_improve = 0

    for step in range(n_iters):
        g_net.train()
        idx = torch.randint(0, N_train, (batch_size,), device=train_x.device)
        Tg = stein_operator_on_net(g_net, train_x[idx], train_s[idx],
                                   use_hutchinson=True, n_probes=4)
        h = train_f[idx] + Tg
        loss = h.var()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g_net.parameters(), 5.0)
        optimizer.step()
        sched.step()

        if (step + 1) % eval_every == 0:
            g_net.eval()
            n_val = min(500, val_x.shape[0])
            vidx = torch.randperm(val_x.shape[0], device=val_x.device)[:n_val]
            with torch.enable_grad():
                Tg_val = stein_operator_on_net(g_net, val_x[vidx], val_s[vidx],
                                                use_hutchinson=True, n_probes=4)
            h_val = (val_f[vidx] + Tg_val).detach()
            val_var = h_val.var().item()

            cur_lr = optimizer.param_groups[0]['lr']
            marker = ""
            if val_var < best_var:
                best_var = val_var
                best_state = copy.deepcopy(g_net.state_dict())
                best_step = step + 1
                no_improve = 0
                marker = " *best*"
            else:
                no_improve += 1

            print(f"    step {step+1}/{n_iters}  loss={loss.item():.4f}  "
                  f"val_var={val_var:.4f}  lr={cur_lr:.1e}{marker}", flush=True)

            if no_improve >= patience:
                print(f"    Early stop at step {step+1} (no improve for {patience} evals)")
                break

    if best_state is not None:
        g_net.load_state_dict(best_state)
    print(f"    Best val_var={best_var:.4f} at step {best_step}")
    return g_net


def evaluate_cv(g_net, samples, score, f_vals, gt, n_boot=2000):
    """Final evaluation with Hutchinson div."""
    g_net.eval()
    N = samples.shape[0]
    h_all = []
    for s in range(0, N, 256):
        e = min(s + 256, N)
        with torch.enable_grad():
            Tg = stein_operator_on_net(g_net, samples[s:e], score[s:e],
                                       use_hutchinson=True, n_probes=8)
        h_all.append((f_vals[s:e] + Tg).detach())
    h_all = torch.cat(h_all)

    cv_mean = h_all.mean().item()
    naive_mean = f_vals.mean().item()

    boots_naive, boots_cv = [], []
    for _ in range(n_boot):
        idx = torch.randint(0, N, (N,), device=samples.device)
        boots_naive.append(f_vals[idx].mean().item())
        boots_cv.append(h_all[idx].mean().item())

    res = dict(
        naive_mean=naive_mean, cv_mean=cv_mean,
        naive_var=np.var(boots_naive), cv_var=np.var(boots_cv),
    )
    res["naive_bias"] = abs(naive_mean - gt)
    res["cv_bias"] = abs(cv_mean - gt)
    res["naive_mse"] = res["naive_bias"]**2 + res["naive_var"]
    res["cv_mse"] = res["cv_bias"]**2 + res["cv_var"]
    res["mse_reduction"] = res["naive_mse"] / max(res["cv_mse"], 1e-15)
    res["bias_reduction"] = res["naive_bias"] / max(res["cv_bias"], 1e-10)
    return res


def evaluate_score_cv(sicv, samples, grad_E, f_vals, Tg_init, gt, n_boot=2000):
    """Evaluate ScoreInformedSteinCV with bootstrap."""
    h_all = sicv.eval_all(samples, grad_E, f_vals, Tg_init).detach()
    N = samples.shape[0]

    cv_mean = h_all.mean().item()
    naive_mean = f_vals.mean().item()

    boots_naive, boots_cv = [], []
    for _ in range(n_boot):
        idx = torch.randint(0, N, (N,), device=samples.device)
        boots_naive.append(f_vals[idx].mean().item())
        boots_cv.append(h_all[idx].mean().item())

    res = dict(
        naive_mean=naive_mean, cv_mean=cv_mean,
        naive_var=np.var(boots_naive), cv_var=np.var(boots_cv),
    )
    res["naive_bias"] = abs(naive_mean - gt)
    res["cv_bias"] = abs(cv_mean - gt)
    res["naive_mse"] = res["naive_bias"]**2 + res["naive_var"]
    res["cv_mse"] = res["cv_bias"]**2 + res["cv_var"]
    res["mse_reduction"] = res["naive_mse"] / max(res["cv_mse"], 1e-15)
    res["bias_reduction"] = res["naive_bias"] / max(res["cv_bias"], 1e-10)
    res["alpha"] = sicv.alpha.item()
    res["Tg_init_mean"] = Tg_init.mean().item()
    res["Tg_init_std"] = Tg_init.std().item()
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  LJ13 Phase 6: KSH-Style Full Pipeline")
    print(f"  Part A: ISM score matching")
    print(f"  Part B: Basic SteinEGNN_LN (Var loss)")
    print(f"  Part C: Score-Informed CV (ISM + residual)")
    print(f"  SteinEGNN_LN h=128 L=5, Hutchinson(4 probes)")
    print(f"  2000 iters, batch=256, eval every 200 steps")
    print("=" * 70)

    energy, source, sde, ref, epoch = build_sampler(device)
    print(f"Loaded epoch {epoch}")

    ref_e = energy.eval(ref).mean().item()
    ref_d = interatomic_dist(ref, N_PARTICLES, SPATIAL_DIM).mean(-1).mean().item()
    print(f"Ground truth:  energy={ref_e:.4f}   dist={ref_d:.4f}")

    all_results = {"energy": [], "dist": []}
    all_score_results = {"energy": [], "dist": []}
    ism = None  # Will be trained on seed 0

    for seed in SEEDS:
        print(f"\n{'─'*70}")
        print(f"  Seed {seed}")
        print(f"{'─'*70}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

        t0 = time.time()
        samples = gen_samples(source, sde, N_SAMPLES, device)
        grad_E = energy.grad_E(samples)
        score = -grad_E
        e_vals = energy.eval(samples)
        d_vals = interatomic_dist(samples, N_PARTICLES, SPATIAL_DIM).mean(-1)
        print(f"  Generated {N_SAMPLES} samples in {time.time()-t0:.1f}s")

        print(f"  Naive: energy={e_vals.mean():.4f} (bias={abs(e_vals.mean().item()-ref_e):.5f})  "
              f"dist={d_vals.mean():.4f} (bias={abs(d_vals.mean().item()-ref_d):.5f})")

        # ============ Part A: ISM Score Matching ============
        if seed == 0:  # Train ISM once on seed 0 samples
            print(f"\n  [ISM] Training score model ...")
            t_ism = time.time()
            score_net = ScoreEGNN(
                n_particles=N_PARTICLES, spatial_dim=SPATIAL_DIM,
                hidden_nf=128, n_layers=4, coord_init_gain=0.1,
            ).to(device)
            ism = ImplicitScoreModel(score_net, dim=DIM,
                                     use_hutchinson=True, n_probes=8,
                                     grad_clip=5.0, device=device)
            ism.fit(samples, n_iters=3000, batch_size=256, lr=3e-4)
            print(f"  [ISM] Done in {time.time()-t_ism:.1f}s")
            ism.diagnose(samples, grad_E)

        # ============ Part B: Basic SteinCV ============
        perm = torch.randperm(N_SAMPLES, device=device)
        n_train = int(N_SAMPLES * 0.8)
        tr_idx, va_idx = perm[:n_train], perm[n_train:]

        for obs_name, f_vals, gt in [("energy", e_vals, ref_e),
                                      ("dist", d_vals, ref_d)]:
            cv_seed = seed * 100 + (0 if obs_name == "energy" else 1)
            torch.manual_seed(cv_seed)
            torch.cuda.manual_seed(cv_seed)

            g_net = SteinEGNN_LN(
                n_particles=N_PARTICLES, spatial_dim=SPATIAL_DIM,
                hidden_nf=128, n_layers=5, tanh=False,
            ).to(device)

            print(f"\n  [{obs_name}] Basic CV Training ...")
            t1 = time.time()
            g_net = train_steincv_fast(
                g_net,
                samples[tr_idx], score[tr_idx], f_vals[tr_idx],
                samples[va_idx], score[va_idx], f_vals[va_idx],
                n_iters=2000, batch_size=256, lr=1e-3,
                patience=5, eval_every=200,
            )
            print(f"  [{obs_name}] Training done in {time.time()-t1:.1f}s")

            print(f"  [{obs_name}] Evaluating ...")
            t2 = time.time()
            res = evaluate_cv(g_net, samples, score, f_vals, gt)
            print(f"  [{obs_name}] Eval done in {time.time()-t2:.1f}s")

            print(f"    Naive: mean={res['naive_mean']:.5f}  "
                  f"bias={res['naive_bias']:.5f}  var={res['naive_var']:.2e}  "
                  f"MSE={res['naive_mse']:.2e}")
            print(f"    CV:    mean={res['cv_mean']:.5f}  "
                  f"bias={res['cv_bias']:.5f}  var={res['cv_var']:.2e}  "
                  f"MSE={res['cv_mse']:.2e}")
            print(f"    MSE reduction: {res['mse_reduction']:.2f}x  "
                  f"Bias reduction: {res['bias_reduction']:.2f}x")
            all_results[obs_name].append(res)

            del g_net
            torch.cuda.empty_cache()

        # ============ Part C: Score-Informed CV ============
        for obs_name, f_vals, gt in [("energy", e_vals, ref_e),
                                      ("dist", d_vals, ref_d)]:
            cv_seed = seed * 100 + 50 + (0 if obs_name == "energy" else 1)
            torch.manual_seed(cv_seed)
            torch.cuda.manual_seed(cv_seed)

            g_res = SteinEGNN_LN(
                n_particles=N_PARTICLES, spatial_dim=SPATIAL_DIM,
                hidden_nf=128, n_layers=5, tanh=False,
            ).to(device)

            sicv = ScoreInformedSteinCV(
                g_res, ism,
                use_hutchinson=True, n_probes=4,
            )

            print(f"\n  [{obs_name}] Score-Informed CV Training ...")
            t1 = time.time()
            sicv.fit(
                samples, energy, f_vals,
                lr=1e-3, n_iters=3000, batch_size=256,
            )
            print(f"  [{obs_name}] Score CV training done in {time.time()-t1:.1f}s")
            print(f"    Final α = {sicv.alpha.item():.4e}")

            print(f"  [{obs_name}] Score CV Evaluating ...")
            t2 = time.time()
            res = evaluate_score_cv(
                sicv, samples, grad_E, f_vals, sicv._Tg_init, gt,
            )
            print(f"  [{obs_name}] Score CV Eval done in {time.time()-t2:.1f}s")

            print(f"    Naive:    mean={res['naive_mean']:.5f}  "
                  f"bias={res['naive_bias']:.5f}  var={res['naive_var']:.2e}  "
                  f"MSE={res['naive_mse']:.2e}")
            print(f"    ScoreCV:  mean={res['cv_mean']:.5f}  "
                  f"bias={res['cv_bias']:.5f}  var={res['cv_var']:.2e}  "
                  f"MSE={res['cv_mse']:.2e}")
            print(f"    MSE reduction: {res['mse_reduction']:.2f}x  "
                  f"Bias reduction: {res['bias_reduction']:.2f}x")
            print(f"    α={res['alpha']:.4e}  "
                  f"T_ν g_init: mean={res['Tg_init_mean']:.2f}  "
                  f"std={res['Tg_init_std']:.2f}")
            all_score_results[obs_name].append(res)

            del g_res, sicv
            torch.cuda.empty_cache()

    # Final summary — Basic CV
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — Basic SteinCV (mean ± std over {len(SEEDS)} seeds)")
    print(f"{'='*70}")

    for obs_name in ["energy", "dist"]:
        results = all_results[obs_name]
        gt = ref_e if obs_name == "energy" else ref_d
        print(f"\n  ── {obs_name} (GT={gt:.4f}) ──")
        for key in ["naive_mean", "naive_bias", "naive_var", "naive_mse",
                     "cv_mean", "cv_bias", "cv_var", "cv_mse",
                     "mse_reduction", "bias_reduction"]:
            vals = [r[key] for r in results]
            m, s = np.mean(vals), np.std(vals)
            if "mse" in key.lower() and "reduction" not in key:
                print(f"    {key:18s}: {m:.2e} ± {s:.2e}")
            elif "var" in key and "reduction" not in key:
                print(f"    {key:18s}: {m:.2e} ± {s:.2e}")
            else:
                print(f"    {key:18s}: {m:.5f} ± {s:.5f}")

    # Final summary — Score-Informed CV
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — Score-Informed CV (mean ± std over {len(SEEDS)} seeds)")
    print(f"{'='*70}")

    for obs_name in ["energy", "dist"]:
        results = all_score_results[obs_name]
        gt = ref_e if obs_name == "energy" else ref_d
        print(f"\n  ── {obs_name} (GT={gt:.4f}) ──")
        for key in ["naive_mean", "naive_bias", "naive_var", "naive_mse",
                     "cv_mean", "cv_bias", "cv_var", "cv_mse",
                     "mse_reduction", "bias_reduction",
                     "alpha", "Tg_init_mean", "Tg_init_std"]:
            vals = [r[key] for r in results]
            m, s = np.mean(vals), np.std(vals)
            if "mse" in key.lower() and "reduction" not in key:
                print(f"    {key:18s}: {m:.2e} ± {s:.2e}")
            elif "var" in key and "reduction" not in key:
                print(f"    {key:18s}: {m:.2e} ± {s:.2e}")
            elif key in ("alpha", "Tg_init_mean", "Tg_init_std"):
                print(f"    {key:18s}: {m:.4e} ± {s:.4e}")
            else:
                print(f"    {key:18s}: {m:.5f} ± {s:.5f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
