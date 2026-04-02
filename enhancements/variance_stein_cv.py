"""
enhancements/variance_stein_cv.py

Variance-loss Stein Control Variates (KSH-style).

Contains:
  - stein_operator_on_net:   Flat-space Stein operator T_ν g(x) = score·g + div(g)
  - SteinBiasCorrector:      Basic CV trainer minimizing Var[f + T_ν g]
  - ScoreInformedSteinCV:    Score-decomposition: g = α·g_init + g_res

Mathematical background:
  For target ν ∝ exp(-E), the Stein operator is:
    T_ν g(x) = -∇E(x)·g(x) + div(g)(x)

  Stein's identity: E_ν[T_ν g(X)] = 0 for any smooth g.
  So h(x) = f(x) + T_ν g(x) is an unbiased estimator of E_ν[f].

  SteinBiasCorrector minimizes Var[h] directly.
  ScoreInformedSteinCV decomposes g = α·g_init + g_res where
    g_init = -(∇E + s_ϕ) uses a pre-trained score model s_ϕ ≈ ∇log q_θ.

Key differences from SML's neural_stein_cv.py (PDE-loss):
  - Loss: Var[h] instead of ||∇h||²
  - Does NOT need ∇f (simpler, works for non-differentiable observables)
  - Supports validation split + early stopping
  - Compatible with any g-network (MLP, EGNN, SteinEGNN_LN)

Ported from KSH_ASBS/stein_cv/neural_cv.py
"""

import copy
import math
import torch
import torch.nn as nn
from torch import Tensor


# ────────────────────────────────────────────────────────────
# Core: flat-space Stein operator
# ────────────────────────────────────────────────────────────

def stein_operator_on_net(g_net, x_flat, score_flat,
                          use_hutchinson=False, n_probes=1):
    """Flat-space Stein operator: T_ν g(x) = score · g + div(g).

    Args:
        g_net: callable, (B, dim) → (B, dim)
        x_flat: (B, dim) system configurations
        score_flat: (B, dim) — score of target = -∇E(x)
        use_hutchinson: if True, estimate div(g) via Hutchinson trace estimator
        n_probes: number of random probes for Hutchinson (default 1)

    Returns:
        (B,) T_ν g per system
    """
    B, dim = x_flat.shape
    x = x_flat.detach().requires_grad_(True)
    g = g_net(x)

    score_dot_g = (score_flat * g).sum(-1)  # (B,)

    if use_hutchinson:
        div_g = torch.zeros(B, device=x_flat.device)
        for _ in range(n_probes):
            v = torch.randn_like(x)
            vJv = torch.autograd.grad(
                (g * v).sum(), x, create_graph=True, retain_graph=True
            )[0]
            div_g = div_g + (v * vJv).sum(-1)
        div_g = div_g / n_probes
    else:
        div_g = torch.zeros(B, device=x_flat.device)
        for d in range(dim):
            grad_d = torch.autograd.grad(
                g[:, d].sum(), x, create_graph=True, retain_graph=True
            )[0]
            div_g = div_g + grad_d[:, d]

    return score_dot_g + div_g


# ────────────────────────────────────────────────────────────
# SteinBiasCorrector: minimize Var[f + T_ν g]
# ────────────────────────────────────────────────────────────

class SteinBiasCorrector:
    """Basic Stein CV: minimize Var[f + T_ν g_ψ].

    Trains any g-network (MLP, SteinEGNN, SteinEGNN_LN) by directly
    minimizing the sample variance of the corrected estimator h = f + T_ν g.

    Features:
      - Validation: 3 modes (fresh sampler, fixed split, train-only)
      - Early stopping with patience
      - Cosine LR annealing
      - Optional bias penalty: λ * E[T_ν g]²

    Usage:
        g_net = SteinEGNN_LN(n_particles=4, spatial_dim=2, hidden_nf=64, n_layers=4)
        corrector = SteinBiasCorrector(g_net=g_net)
        corrector.fit(samples, grad_E, f_vals, n_iters=10000)
        mean_est, Tg = corrector.estimate(samples, grad_E, f_vals)
    """

    def __init__(self, g_net, use_hutchinson=False, n_probes=4):
        self.g_net = g_net
        self.use_hutchinson = use_hutchinson
        self.n_probes = n_probes

    def _compute_h(self, samples, score, f_vals, batch_size=1000,
                    return_Tg=False):
        """Compute h = f + T_ν g for a set of samples (batched, no grad on output)."""
        self.g_net.eval()
        N = samples.shape[0]
        h_all, Tg_all = [], []
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            with torch.enable_grad():
                Tg = stein_operator_on_net(
                    self.g_net, samples[s:e], score[s:e],
                    use_hutchinson=self.use_hutchinson, n_probes=self.n_probes,
                )
            h_all.append((f_vals[s:e] + Tg).detach())
            if return_Tg:
                Tg_all.append(Tg.detach())
        self.g_net.train()
        if return_Tg:
            return torch.cat(h_all), torch.cat(Tg_all)
        return torch.cat(h_all)

    def fit(self, samples, grad_E, f_vals,
            lr=1e-3, n_iters=5000, batch_size=512, verbose=True,
            cosine_lr=True, weight_decay=0.0, val_fraction=0.2,
            patience=6, bias_penalty=0.0, val_sampler=None):
        """Train Stein CV with early stopping.

        Validation modes (checked in priority order):
          1. val_sampler: callable() -> (samples, grad_E, f_vals).
             Fresh samples every eval — best for detecting overfitting.
          2. val_fraction > 0: fixed train/val split.
          3. val_fraction == 0: no validation, use train variance.

        Args:
            samples: (N, dim) terminal ASBS samples
            grad_E: (N, dim) energy gradients ∇E(x)
            f_vals: (N,) observable values f(x)
            lr: learning rate (default 1e-3)
            n_iters: training iterations (default 5000)
            batch_size: mini-batch size (default 512)
            cosine_lr: use cosine annealing LR schedule
            weight_decay: L2 regularization
            val_fraction: fraction of data for validation (0 = no split)
            patience: early stopping patience (0 = disabled)
            bias_penalty: λ for mean(T_ν g)² penalty term
            val_sampler: optional callable for fresh validation samples
        """
        device = samples.device
        score = -grad_E
        N = samples.shape[0]

        use_fresh_val = val_sampler is not None

        if use_fresh_val:
            train_samples, train_score, train_f = samples, score, f_vals
            N_train = N
            if verbose:
                print(f"  [SteinCV] Train: {N} (fresh re-sample validation)")
        elif val_fraction > 0:
            perm = torch.randperm(N, device=device)
            n_train = int(N * (1 - val_fraction))
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]
            train_samples = samples[train_idx]
            train_score = score[train_idx]
            train_f = f_vals[train_idx]
            val_samples = samples[val_idx]
            val_score = score[val_idx]
            val_f = f_vals[val_idx]
            N_train = n_train
            if verbose:
                print(f"  [SteinCV] Train/Val split: {n_train}/{N - n_train}")
        else:
            train_samples, train_score, train_f = samples, score, f_vals
            N_train = N

        optimizer = torch.optim.Adam(
            self.g_net.parameters(), lr=lr, weight_decay=weight_decay)
        if cosine_lr:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_iters, eta_min=lr * 0.01)
        else:
            sched = None

        best_var = float("inf")
        best_state = None
        best_step = 0
        no_improve = 0

        for step in range(n_iters):
            if batch_size >= N_train:
                idx = torch.arange(N_train, device=device)
            else:
                idx = torch.randint(0, N_train, (batch_size,), device=device)
            Tg = stein_operator_on_net(
                self.g_net, train_samples[idx], train_score[idx],
                use_hutchinson=self.use_hutchinson, n_probes=self.n_probes,
            )
            h = train_f[idx] + Tg
            loss = h.var()
            if bias_penalty > 0:
                loss = loss + bias_penalty * (Tg.mean() ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.g_net.parameters(), 5.0)
            optimizer.step()
            if sched:
                sched.step()

            # Evaluate every 500 steps
            if verbose and (step + 1) % 500 == 0:
                h_train, Tg_train = self._compute_h(
                    train_samples, train_score, train_f, return_Tg=True)
                train_var = h_train.var().item()
                train_mean = h_train.mean().item()
                Tg_mean_train = Tg_train.mean().item()
                cur_lr = optimizer.param_groups[0]['lr']

                has_val = False
                if use_fresh_val:
                    vs, vg, vf = val_sampler()
                    v_score = -vg
                    h_val, Tg_val = self._compute_h(
                        vs, v_score, vf, return_Tg=True)
                    val_var = h_val.var().item()
                    val_mean = h_val.mean().item()
                    Tg_mean_val = Tg_val.mean().item()
                    check_var = val_var
                    has_val = True
                    del vs, vg, vf, v_score, h_val, Tg_val
                elif not use_fresh_val and val_fraction > 0:
                    h_val, Tg_val = self._compute_h(
                        val_samples, val_score, val_f, return_Tg=True)
                    val_var = h_val.var().item()
                    val_mean = h_val.mean().item()
                    Tg_mean_val = Tg_val.mean().item()
                    check_var = val_var
                    has_val = True
                else:
                    check_var = train_var

                marker = ""
                if check_var < best_var:
                    best_var = check_var
                    best_state = copy.deepcopy(self.g_net.state_dict())
                    best_step = step + 1
                    no_improve = 0
                    marker = " *best*"
                else:
                    no_improve += 1

                if has_val:
                    print(f"  [SteinCV] step {step+1}/{n_iters}  "
                          f"t_var={train_var:.6f} t_mean={train_mean:.4f} "
                          f"t_Tg={Tg_mean_train:+.4f}  "
                          f"v_var={val_var:.6f} v_mean={val_mean:.4f} "
                          f"v_Tg={Tg_mean_val:+.4f}  "
                          f"lr={cur_lr:.1e}{marker}")
                else:
                    print(f"  [SteinCV] step {step+1}/{n_iters}  "
                          f"var={train_var:.6f} mean={train_mean:.4f} "
                          f"Tg_mean={Tg_mean_train:+.4f}  "
                          f"lr={cur_lr:.1e}{marker}")

                if patience > 0 and no_improve >= patience:
                    if verbose:
                        print(f"  [SteinCV] Early stopping at step {step+1} "
                              f"(no improvement for {patience} evals)")
                    break

        if best_state is not None:
            self.g_net.load_state_dict(best_state)
        if verbose:
            tag = "fresh_val_var" if use_fresh_val else (
                "val_var" if val_fraction > 0 else "full_var")
            print(f"  [SteinCV] Best {tag}={best_var:.6f} at step {best_step}")

    def estimate(self, samples, grad_E, f_vals):
        """Corrected estimate for a (sub)sample.

        Args:
            samples: (N, dim) samples
            grad_E: (N, dim) energy gradients
            f_vals: (N,) observable values

        Returns:
            (mean_estimate: float, Tg: (N,) tensor)
        """
        self.g_net.eval()
        score = -grad_E
        with torch.enable_grad():
            Tg = stein_operator_on_net(
                self.g_net, samples, score,
                use_hutchinson=self.use_hutchinson, n_probes=self.n_probes,
            )
        h = f_vals + Tg.detach()
        return h.mean().item(), Tg.detach()


# ────────────────────────────────────────────────────────────
# ScoreInformedSteinCV: g = α·g_init + g_res
# ────────────────────────────────────────────────────────────

class ScoreInformedSteinCV:
    """Score-informed Stein CV: g = α·g_init + g_res.

    Decomposes the CV vector field into:
      g_init(x) = -(∇E(x) + s_ϕ(x))   [from pre-trained score model]
      g_res(x)                           [trainable residual]
      α                                  [learnable scalar, log-parameterized]

    When q_θ ≈ π: s_ϕ ≈ -∇E → g_init ≈ 0 → T_ν g_init ≈ 0.
    The residual g_res only corrects the remaining mismatch.

    T_ν g_init is precomputed once and frozen during training.
    Only g_res parameters and α are optimized.

    Loss: mean((h - running_mean)²) where h = f + α·T_ν g_init + T_ν g_res
    The running mean (EMA, decay=0.99) provides stable centering.

    Usage:
        # Phase 1: Train score model
        ism = ImplicitScoreModel(ScoreEGNN(...), ...)
        ism.fit(samples)

        # Phase 2: Train score-informed CV
        g_res = SteinEGNN_LN(...)
        sicv = ScoreInformedSteinCV(g_res, ism, use_hutchinson=True, n_probes=4)
        sicv.fit(samples, energy, f_vals)
        h_all = sicv.eval_all(samples, grad_E, f_vals, sicv._Tg_init)
    """

    def __init__(self, g_residual_net, score_model,
                 use_hutchinson=False, n_probes=4):
        self.g_res = g_residual_net
        self.score_model = score_model
        self.use_hutchinson = use_hutchinson
        self.n_probes = n_probes
        self.log_alpha = None
        self._Tg_init = None

    @property
    def alpha(self):
        """Current α value (exp of log_alpha)."""
        return torch.exp(self.log_alpha)

    def _precompute_Tg_init(self, samples, energy, score_net,
                             n_particles, spatial_dim, batch_size=256):
        """Compute T_ν g_init for all samples where g_init = -(∇E + s_ϕ).

        This is computed once and frozen — only α and g_res are trained.

        Args:
            samples: (N, dim) terminal samples
            energy: energy object with .eval() and .grad_E()
            score_net: trained ScoreEGNN network
            n_particles: number of particles
            spatial_dim: spatial dimension per particle

        Returns:
            (N,) tensor of T_ν g_init values
        """
        from .score_matching import _build_edges

        N, dim = samples.shape
        device = samples.device
        Tg = torch.zeros(N, device=device)

        grad_E_all = energy.grad_E(samples)
        score_all = -grad_E_all

        score_net.eval()
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start

            with torch.enable_grad():
                x_req = samples[start:end].detach().requires_grad_(True)

                e = energy.eval(x_req)
                grad_e = torch.autograd.grad(
                    e.sum(), x_req, create_graph=True
                )[0]

                x_parts = x_req.view(B * n_particles, spatial_dim)
                edge_index, batch_idx = _build_edges(n_particles, B, device)
                s_phi = score_net(x_parts, batch_idx, edge_index).view(B, dim)

                g = -(grad_e + s_phi)

                score_b = score_all[start:end]
                score_dot_g = (score_b * g).sum(-1)

                if self.use_hutchinson:
                    div_g = torch.zeros(B, device=device)
                    for _ in range(self.n_probes):
                        v = torch.randn_like(x_req)
                        vJv = torch.autograd.grad(
                            (g * v).sum(), x_req, retain_graph=True
                        )[0]
                        div_g = div_g + (v * vJv).sum(-1)
                    div_g = div_g / self.n_probes
                else:
                    div_g = torch.zeros(B, device=device)
                    for d in range(dim):
                        grad_d = torch.autograd.grad(
                            g[:, d].sum(), x_req, retain_graph=True
                        )[0]
                        div_g = div_g + grad_d[:, d]

                Tg[start:end] = (score_dot_g + div_g).detach()

        return Tg

    def fit(self, samples, energy, f_vals,
            lr=1e-3, n_iters=5000, batch_size=512, verbose=True):
        """Train score-informed Stein CV.

        1. Precomputes T_ν g_init (frozen)
        2. Optimizes α (log-parameterized) and g_res to minimize
           mean((h - running_mean)²) where h = f + α·T_init + T_res

        Args:
            samples: (N, dim) terminal samples
            energy: energy object with .eval(), .grad_E()
            f_vals: (N,) observable values
            lr: learning rate
            n_iters: training iterations
            batch_size: mini-batch size
        """
        device = samples.device
        n_particles = self.g_res.n_particles
        spatial_dim = self.g_res.spatial_dim

        # Get the underlying ScoreEGNN network from the ISM model
        s_net = getattr(self.score_model, 's_net',
                        getattr(self.score_model, 'score_net', None))

        print("    Precomputing T_ν g_init ...")
        self._Tg_init = self._precompute_Tg_init(
            samples, energy, s_net, n_particles, spatial_dim
        )

        Tg_std = self._Tg_init.std().item()
        alpha_init = 0.01
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha_init), device=device)
        )
        print(f"    T_ν g_init: mean={self._Tg_init.mean():.2f}  "
              f"std={Tg_std:.2f}  α_init={alpha_init:.2e}")

        grad_E = energy.grad_E(samples)
        score = -grad_E

        optimizer = torch.optim.Adam(
            list(self.g_res.parameters()) + [self.log_alpha], lr=lr
        )
        N = samples.shape[0]
        running_mean = 0.0

        for step in range(n_iters):
            idx = torch.randint(0, N, (batch_size,), device=device)
            Tg_res = stein_operator_on_net(
                self.g_res, samples[idx], score[idx],
                use_hutchinson=self.use_hutchinson, n_probes=self.n_probes,
            )
            alpha = torch.exp(self.log_alpha)
            h = f_vals[idx] + alpha * self._Tg_init[idx] + Tg_res

            with torch.no_grad():
                running_mean = 0.99 * running_mean + 0.01 * h.mean().item()
            loss = ((h - running_mean) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.g_res.parameters()) + [self.log_alpha], 5.0
            )
            optimizer.step()

            if verbose and (step + 1) % 500 == 0:
                print(f"  [ScoreSteinCV] step {step+1}/{n_iters}  "
                      f"var={loss.item():.6f}  "
                      f"α={torch.exp(self.log_alpha).item():.2e}")

    def eval_all(self, samples, grad_E, f_vals, Tg_init):
        """Compute h = f + α·T_init + T_res for all samples (batched).

        Args:
            samples: (N, dim)
            grad_E: (N, dim)
            f_vals: (N,)
            Tg_init: (N,) precomputed T_ν g_init values

        Returns:
            (N,) corrected h values
        """
        self.g_res.eval()
        score = -grad_E
        N = samples.shape[0]
        bs = 256
        Tg_res = torch.zeros(N, device=samples.device)

        for start in range(0, N, bs):
            end = min(start + bs, N)
            with torch.enable_grad():
                tr = stein_operator_on_net(
                    self.g_res, samples[start:end], score[start:end],
                    use_hutchinson=self.use_hutchinson, n_probes=self.n_probes,
                )
            Tg_res[start:end] = tr.detach()

        alpha = torch.exp(self.log_alpha).item()
        return f_vals + alpha * Tg_init + Tg_res
