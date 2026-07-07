"""Compare the three section-6 prediction losses in the linear LQE testbed.

The DESIGN.md (section 6) world-model objective has three prediction terms,
indexed by the {a-priori vs a-posteriori} x {observation vs latent} axes:

- ``L_prior_obs = 0.5||y_hat  - y||^2``           a-priori observation (grounding; M4 today)
- ``L_post_obs  = 0.5||y_post - y||^2``           a-posteriori observation (direct-K; section 6.1)
- ``L_latent_sd = 0.5||sg(x_t^+) - F x_{t-1}^+||^2``  latent self-distillation (section 4.2/6)

All three live in :class:`kf_rnn.model.sequential.RnnSelfDistillPredictor`, a thin
override of the test-time-training (TTT) filter ``RnnInContextPredictor``. By
choosing the loss weights ``(alpha, beta0, beta2)`` and which fast-weights adapt
(``adapt_keys``) this single model recovers the four canvas methods M1-M4 (and the
M3 "bridge"), so we can put them on one footing and read them out two ways:

- SD story (the self-distillation / deadly-triad view): eigen-phases and |lambda|
  of the adapted ``F`` vs the true system spectrum -- does grounding stop the
  M1/M2 magnitude blow-up while the phases still lock onto the true eigen-phases?
- TTT story (the filtering view): impulse response of the adapted filter vs the
  ground-truth steady-state Kalman filter (the *irreducible* optimum), plus the
  project's exact closed-form analytical error of the adapted filter as it adapts,
  reported as the **excess over the irreducible floor** ``(err - floor)/floor``
  (0 == reaches the irreducible optimum). The analytical error is an expectation
  over the noise (no sampling, no in-sample bias), unlike a finite-trace MSE.

Examples
--------
Default sweep (M1-M4 + a-posteriori probe) on a 6-state, 2-obs system::

    python scripts/self_distillation_losses.py --s-d 6 --o-d 2 --seed 0

Fully observed, larger state::

    python scripts/self_distillation_losses.py --s-d 8 --o-d 8 -L 400
"""

import argparse
import os
import tempfile
from types import SimpleNamespace
from typing import NamedTuple, Optional

# Writable matplotlib cache (the user's ~/.config may be read-only on the cluster).
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "kf_rnn_mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

# The compute device is selected from ``--device`` in ``main`` (default: cuda if
# available, else cpu). The ``kf_rnn`` import below sets the project default to
# cuda:0 via ``ecliseutils.configure``; ``main`` has the final say. Plotting uses
# ``.numpy(force=True)`` / ``.cpu()`` so figures render on either device.

import ecliseutils as eu
from tensordict import TensorDict

from kf_rnn.infrastructure.config import EnvironmentShape, ProblemShape, SystemConfig
from kf_rnn.infrastructure.config.schema import TrainConfig
from kf_rnn.infrastructure.settings import OUTPUT_PATH
from kf_rnn.model.sequential import RnnSelfDistillPredictor
from kf_rnn.model.sequential.base import SequentialPredictor
from kf_rnn.model.convolutional import CnnAnalyticalLeastSquaresPredictor, CnnLeastSquaresPredictor
from kf_rnn.model.convolutional.base import ConvolutionalPredictor
from kf_rnn.system.linear_time_invariant import ContinuousDistribution

# Reuse the existing readout helpers (both scripts are import-safe -- their work is
# guarded by ``if __name__ == "__main__"``); the scripts dir is on sys.path[0].
from ttt_impulse import kalman_ir
from self_distillation_eigenphase import active_phase_match_error, _draw_complex_plane


# SECTION: Method catalogue (weights + which fast-weights adapt + decoder init)

class Method(NamedTuple):
    label: str
    cfg: dict        # extra RnnSelfDistillPredictor.Config kwargs (alpha/beta0/beta2/adapt_keys/weight_decay)
    h_init: str      # "random" (kaiming default) or "zero" (M1: innovation == y -> raw K y injection)


def default_methods(anchor: float, post: float, wd: float) -> list[Method]:
    """The M1-M4 ladder plus an a-posteriori probe.

    ``anchor`` is the small a-priori grounding weight ``beta0`` for the M3 bridge,
    ``post`` the a-posteriori weight ``beta2`` for the probe, ``wd`` the decoupled
    weight decay paired with the a-posteriori term (section 6.1 rate analog).
    """
    return [
        Method("M1 raw-injection\n(latent, H=0)",
               dict(alpha=1.0, beta0=0.0, beta2=0.0, adapt_keys=("F",)), "zero"),
        Method("M2 error-correct\n(latent, fixed H)",
               dict(alpha=1.0, beta0=0.0, beta2=0.0, adapt_keys=("F",)), "random"),
        Method("M3 bridge\n(latent + anchor)",
               dict(alpha=1.0, beta0=anchor, beta2=0.0, adapt_keys=("F", "H")), "random"),
        Method("M4 full obs\n(a-priori)",
               dict(alpha=0.0, beta0=1.0, beta2=0.0, adapt_keys=("F", "H", "K")), "random"),
        Method("M3+post\n(latent+anchor+post)",
               dict(alpha=1.0, beta0=anchor, beta2=post, adapt_keys=("F", "H", "K"), weight_decay=wd), "random"),
    ]


def m3m4_methods() -> list[Method]:
    """Clean 4-way ``(alpha, beta0, beta2)`` comparison for the m3-vs-m4 study.

    All four adapt every fast-weight ``(F, H, K)`` (K starts at the ``K=0``
    default) and use **no weight decay**, so the only independent variable is the
    loss-weight triple -- this re-isolates the three section-6 components without
    the historical frozen-K / weight-decay plumbing (study manifest section 5).
    The grid is ``alpha`` on/off x ``beta2`` on/off, all sharing the ``beta0``
    grounding:

    - ``(1.0, 0.05, 0.05)`` latent SD + anchor + a-posteriori (M3+post, no wd)
    - ``(1.0, 0.05, 0.0)``  latent SD + anchor (M3 bridge)
    - ``(0.0, 1.0, 1.0)``   a-priori + a-posteriori obs (M4 + post)
    - ``(0.0, 1.0, 0.0)``   pure a-priori obs (M4)
    - ``(1.0, 1.0, 1.0)``   latent SD + full a-priori + a-posteriori (all three on)
    - ``(1.0, 1.0, 0.0)``   latent SD + full a-priori (SD with strong grounding)
    """
    akeys = ("F", "H", "K")
    return [
        Method("M3+post no-wd\n(SD+anchor+post)",
               dict(alpha=1.0, beta0=0.05, beta2=0.05, adapt_keys=akeys), "random"),
        Method("M3 bridge\n(SD+anchor)",
               dict(alpha=1.0, beta0=0.05, beta2=0.0, adapt_keys=akeys), "random"),
        Method("M4+post\n(a-priori+post)",
               dict(alpha=0.0, beta0=1.0, beta2=1.0, adapt_keys=akeys), "random"),
        Method("M4\n(a-priori)",
               dict(alpha=0.0, beta0=1.0, beta2=0.0, adapt_keys=akeys), "random"),
        Method("SD+M4+post\n(1,1,1)",
               dict(alpha=1.0, beta0=1.0, beta2=1.0, adapt_keys=akeys), "random"),
        Method("SD+M4\n(1,1,0)",
               dict(alpha=1.0, beta0=1.0, beta2=0.0, adapt_keys=akeys), "random"),
    ]


def minimal_methods(step_decay: float = 0.67, polyak_burnin: float = 0.1) -> list[Method]:
    """The minimal vanishing-gain method (and its constant-step control).

    Both curves are the pure a-priori M4 objective ``(alpha, beta0, beta2) =
    (0, 1, 0)`` adapting every fast-weight ``(F, H, K)`` with the truncated
    ``window`` gradient; the *only* difference is the gain:

    - "M4 constant" -- the current constant-step SGD (``step_decay=0``, no
      averaging): reproduces the ~0.2%-excess plateau as the control. It is
      *pinned* constant (``step_decay=0``, ``polyak_burnin=-1``) so it stays a
      faithful control even when the global ``--step-decay`` / ``--polyak-burnin``
      are set for a sweep.
    - "M4 decay+avg" -- a vanishing gain ``eta_t = step_size * (t+1)^(-a)`` with
      Polyak-Ruppert tail averaging from ``polyak_burnin * L``: the minimal change
      that targets the constant-gain noise ball, the load-bearing cause of the
      plateau (no stability projection / warm start / RTRL). ``a`` and the burn-in
      come from the arguments (wired to the CLI in ``main``), so the knob-only
      isolation sweep is a single ``--step-decay`` / ``--polyak-burnin`` per run.
    """
    akeys = ("F", "H", "K")
    return [
        Method("M4 constant\n(baseline)",
               dict(alpha=0.0, beta0=1.0, beta2=0.0, adapt_keys=akeys,
                    step_decay=0.0, polyak_burnin=-1.0), "random"),
        Method(f"M4 decay+avg\n(a={step_decay:g}, n0={polyak_burnin:g})",
               dict(alpha=0.0, beta0=1.0, beta2=0.0, adapt_keys=akeys,
                    step_decay=step_decay, polyak_burnin=polyak_burnin), "random"),
    ]


def mech_methods(step_decay: float = 0.51, polyak_burnin: float = 0.5) -> list[Method]:
    """Three-way, constant-gain-centric mechanism comparison on stationary LQE.

    The reframing: a vanishing gain cannot track a moving target, so the
    decay+avg method is a stationary *oracle* / lower bound, not a deployable
    filter. The real contest is among **constant-gain** schemes, judged by their
    steady noise-ball size *and* tail stability:

    - "M4 constant" -- pure a-priori ``(0, 1, 0)`` at constant gain: the honest
      deployable baseline (asymptotes to a small ball ``~ sqrt(eta * grad_var)``).
    - "SD+anchor constant" -- latent SD + small a-priori anchor ``(1, 0.05, 0)``
      at constant gain: the candidate. If the smooth model-generated SD target
      reduces the gradient variance, its ball shrinks at the *same* ``eta`` -- a
      directly transferable win, even if it never reaches exactly zero excess.
    - "M4 decay+avg (oracle)" -- pure a-priori with the vanishing-gain + Polyak
      recipe (``step_decay``/``polyak_burnin`` from the global knobs): the
      stationary lower bound, shown only to price how far the deployable constant
      methods sit from the optimum. Across a ``--step-size`` sweep its non-recipe
      points also document the decaying-gain brittleness.

    Fairness is per-method-best: sweep ``--step-size`` and read each method at its
    own best step (the constant and decay schedules have opposite optimal-step
    regimes), rather than pinning a single shared initial gain.
    """
    akeys = ("F", "H", "K")
    return [
        Method("M4 constant\n(a-priori)",
               dict(alpha=0.0, beta0=1.0, beta2=0.0, adapt_keys=akeys,
                    step_decay=0.0, polyak_burnin=-1.0), "random"),
        Method("SD+anchor constant\n(alpha=1, beta0=0.05)",
               dict(alpha=1.0, beta0=0.05, beta2=0.0, adapt_keys=akeys,
                    step_decay=0.0, polyak_burnin=-1.0), "random"),
        Method(f"M4 decay+avg oracle\n(a={step_decay:g}, n0={polyak_burnin:g})",
               dict(alpha=0.0, beta0=1.0, beta2=0.0, adapt_keys=akeys,
                    step_decay=step_decay, polyak_burnin=polyak_burnin), "random"),
    ]


def nstep_methods(max_depth: int = 4, anchor: float = 0.05) -> list[Method]:
    """Depth sweep of the n-step latent self-distillation ladder on constant gain.

    Isolates the *only* new knob -- the SD horizon -- against a common pure
    a-priori control, one arm per ladder depth ``n = 1..max_depth`` (``max_depth``
    is normally the window length; a horizon deeper than the window clamps to the
    detached root, so it saturates there and there is no reason to sweep past it):

    - "M4 constant"   -- pure a-priori ``(0, 1, 0)``: the deployable baseline.
    - "SD+anchor n=k"  -- latent SD + small a-priori anchor ``(1, anchor, 0)`` with
      the ladder set to ``sd_horizon=k`` (autonomous ``F`` roll-out over horizons
      ``1..k``), for ``k = 1..max_depth``. ``n=1`` is the original single-step SD
      term -- i.e. the previous default behavior -- so the sweep always contains
      the headline as its first SD arm.

    All arms are pinned constant-gain (``step_decay=0``, ``polyak_burnin=-1``) so
    the horizon is the only independent variable; sweep ``--step-size`` and read
    each at its own best step, as in the ``mech`` study.
    """
    akeys = ("F", "H", "K")
    methods = [
        Method("M4 constant\n(a-priori)",
               dict(alpha=0.0, beta0=1.0, beta2=0.0, adapt_keys=akeys,
                    step_decay=0.0, polyak_burnin=-1.0, sd_horizon=1), "random"),
    ]
    for k in range(1, max(1, max_depth) + 1):
        methods.append(
            Method(f"SD+anchor n={k}\n(alpha=1, beta0={anchor:g})",
                   dict(alpha=1.0, beta0=anchor, beta2=0.0, adapt_keys=akeys,
                        step_decay=0.0, polyak_burnin=-1.0, sd_horizon=k), "random"))
    return methods


# SECTION: Metrics

# Magnitude above which a matrix is treated as diverged WITHOUT calling an
# eigensolver: a diverging filter's entries grow geometrically and pass through
# huge-but-finite values before becoming inf, and LAPACK (MKL) can corrupt memory
# / hard-abort on such inputs -- which no Python ``try/except`` can catch. A stable
# filter has O(1) entries, so 1e6 is a safe, generous divergence cutoff.
_EIG_MAX_ABS = 1e6


def _eig_unsafe(M: torch.Tensor) -> bool:
    """True if ``M`` must not be fed to an eigensolver (non-finite or huge)."""
    return (not torch.isfinite(M).all()) or (M.abs().max().item() > _EIG_MAX_ABS)


def _spectral_radius(M: torch.Tensor) -> float:
    """Spectral radius, robust to a diverged (non-finite/huge) adapted F (returns inf)."""
    if _eig_unsafe(M):
        return float("inf")
    try:
        return torch.linalg.eigvals(M).abs().max().item()
    except RuntimeError:
        return float("inf")


@torch.no_grad()
def filter_analytical_error(F_hat: torch.Tensor, H_hat: torch.Tensor,
                            K_hat: torch.Tensor, lsg_td: TensorDict) -> float:
    """Exact steady-state observation prediction error of the LTI filter
    ``(F_hat, H_hat, K_hat)`` against the true system (``lsg_td = lsg.td()``).

    This is the project's closed-form analytical error -- an expectation over the
    process/observation noise, not a finite-sample MSE -- so it has no in-sample
    bias and no noise-floor sampling jitter.

    Returns ``+inf`` when the filter's closed loop ``F_hat (I - K_hat H_hat)`` is
    unstable (spectral radius ``>= 1``): the closed-form geometric series only
    converges for a contractive closed loop, which is exactly the M1/M2 (and the
    early-adaptation) regime, so an unstable filter has no finite analytical error.
    """
    if any(_eig_unsafe(M) for M in (F_hat, H_hat, K_hat)):
        return float("inf")
    S = F_hat.shape[-1]
    eye = torch.eye(S, dtype=F_hat.dtype, device=F_hat.device)
    closed = F_hat @ (eye - K_hat @ H_hat)                       # filter-side closed loop
    if _eig_unsafe(closed):
        return float("inf")
    try:
        rho = torch.linalg.eigvals(closed).abs().max().item()
    except RuntimeError:
        return float("inf")
    if not np.isfinite(rho) or rho >= 1.0:
        return float("inf")
    kfs = TensorDict({"F": F_hat, "H": H_hat, "K": K_hat}, batch_size=torch.Size([]))
    err = SequentialPredictor.analytical_error(kfs, lsg_td)["environment", "observation"].item()
    return err if np.isfinite(err) and err >= 0.0 else float("inf")


@torch.no_grad()
def batched_filter_analytical_error(F: torch.Tensor, H: torch.Tensor, K: torch.Tensor,
                                    lsg_td: TensorDict) -> np.ndarray:
    """Vectorized :func:`filter_analytical_error` over a leading ``[N x ...]`` batch
    of filters; returns an ``[N]`` numpy array (``nan`` for each filter that the
    scalar version would reject as ``+inf``).

    Same closed-form analytical error and the same divergence/contractivity guard
    as the scalar path, but computed for all ``N`` trajectories in one shot. The
    crucial detail is that any non-finite / huge / non-contractive trajectory is
    replaced by a trivially stable dummy (``F = H = K = 0`` -> closed loop ``0``)
    *before* any eigensolver runs, so LAPACK/cuSOLVER never sees a diverged matrix
    (the memory-corruption hazard the scalar guard exists to avoid); those entries
    are masked back to ``nan`` afterwards. ``nan`` (not ``+inf``) matches what the
    callers store for an unstable snapshot."""
    N, S = F.shape[0], F.shape[-1]
    eye = torch.eye(S, dtype=F.dtype, device=F.device)

    def _safe(M: torch.Tensor) -> torch.Tensor:                     # [N] bool
        flat = M.reshape(N, -1)
        return torch.isfinite(flat).all(dim=1) & (flat.abs().amax(dim=1) <= _EIG_MAX_ABS)

    def _gate(M: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        return torch.where(keep.reshape(N, *([1] * (M.dim() - 1))), M, torch.zeros_like(M))

    ok = _safe(F) & _safe(H) & _safe(K)
    Fs, Hs, Ks = _gate(F, ok), _gate(H, ok), _gate(K, ok)
    closed = Fs @ (eye - Ks @ Hs)                                   # [N x S x S], safe to eig
    rho = torch.linalg.eigvals(closed).abs().amax(dim=-1)           # [N]
    stable = ok & torch.isfinite(rho) & (rho < 1.0)

    kfs = TensorDict({"F": _gate(F, stable), "H": _gate(H, stable), "K": _gate(K, stable)},
                     batch_size=torch.Size([N]))
    err = SequentialPredictor.analytical_error(kfs, lsg_td)["environment", "observation"].reshape(N)
    valid = stable & torch.isfinite(err) & (err >= 0.0)
    out = torch.where(valid, err, torch.full_like(err, float("nan")))
    return out.numpy(force=True)


@torch.no_grad()
def _adapt_and_measure(model: RnnSelfDistillPredictor, state0: torch.Tensor,
                       observations: torch.Tensor, lsg_td: TensorDict,
                       sampled_set: set[int]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor],
                                                       dict[int, float], dict[int, float]]:
    """Replay the online TTT loop for one trajectory (mirrors ``adapted_theta``),
    snapshot the analytical error at each step in ``sampled_set``, and return the
    final *reported* fast-weights, the final *raw* iterate, and the per-step
    analytical error of each (``{step: analytical_error}``).

    When ``model.polyak_burnin >= 0`` the reported filter is the Polyak-Ruppert
    tail average ``theta_bar = mean_{t >= n0} theta_t`` (``n0 = polyak_burnin * L``)
    rather than the raw iterate, so the snapshots and the returned filter both see
    the averaged (asymptotically efficient) estimate. ``polyak_burnin < 0`` keeps
    the raw iterate (``theta_bar is theta``), i.e. the original behaviour.

    Diagnostic: ``raw_step_errs`` is the analytical error of the *raw* iterate
    (no averaging) at the same sampled steps. Comparing it to the averaged
    ``step_errs`` separates two failure modes -- a raw iterate that is itself
    far/unstable (under-convergence) versus a raw iterate that is stable but whose
    Polyak average ``theta_bar`` tips over the unit circle (a parameter-space
    averaging artifact, since stability is not preserved under averaging)."""
    theta = {k: v.detach().clone() for k, v in eu.td_items(eu.parameter_td(model)).items()}
    s = state0
    w = model.window
    ent_win = s[None].expand(w, *s.shape).clone()
    L = observations.shape[0]
    n0 = int(model.polyak_burnin * L) if model.polyak_burnin >= 0.0 else None
    theta_sum: dict[str, torch.Tensor] | None = None
    avg_count = 0
    step_errs: dict[int, float] = {}
    raw_step_errs: dict[int, float] = {}
    for t in range(L):
        out = torch.func.functional_call(model.cell, theta, (s, {}, observations[t]))
        s_post = out["environment", "state"]
        ent_win = torch.cat([ent_win[1:], s[None]], dim=0)
        s_start = ent_win[0].detach()
        w0 = max(0, t - w + 1)
        win_obs = observations[w0:t + 1]
        for _ in range(model.n_steps):
            grads = model._compute_grads(theta, s_start, {}, win_obs)
            theta = model._optimizer_step(theta, grads, t)
        s = s_post

        # Reported filter: raw iterate, or the Polyak tail average once past burn-in.
        if n0 is not None and t >= n0:
            if theta_sum is None:
                theta_sum = {k: v.clone() for k, v in theta.items()}
            else:
                for k, v in theta.items():
                    theta_sum[k] += v
            avg_count += 1
            theta_bar = {k: v / avg_count for k, v in theta_sum.items()}
        else:
            theta_bar = theta

        if t in sampled_set:
            step_errs[t] = filter_analytical_error(theta_bar["F"], theta_bar["H"], theta_bar["K"], lsg_td)
            # Raw-iterate error (no averaging) for the under-convergence vs
            # averaging-artifact diagnostic. When averaging is disabled
            # ``theta_bar is theta``, so this simply duplicates ``step_errs``.
            raw_step_errs[t] = (step_errs[t] if theta_bar is theta
                                else filter_analytical_error(theta["F"], theta["H"], theta["K"], lsg_td))
    return theta_bar, theta, step_errs, raw_step_errs


@torch.no_grad()
def _adapt_and_measure_batched(model: RnnSelfDistillPredictor, state0: torch.Tensor,
                               observations: torch.Tensor, lsg_td: TensorDict,
                               sampled_steps: list[int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    """Vectorized replay of the online TTT loop over ALL ``N`` trajectories at once.

    Semantically identical to running :func:`_adapt_and_measure` for each of the
    ``N`` trajectories (same prediction, sliding window, gradient step, Polyak
    tail-averaging, and per-step analytical-error snapshots), but every operation
    carries a leading ``[N x ...]`` batch and the per-trajectory prediction /
    gradient are ``torch.vmap``-ed -- so the length-``L`` Python loop launches one
    set of GPU kernels per step instead of ``N``. All ``N`` trajectories share the
    same initial fast-weights (the model's parameters), exactly as the per-trace
    version did (it re-cloned the same ``theta`` each call).

    Returns the final *reported* (Polyak-averaged or raw) and the final *raw*
    fast-weights as ``[N x ...]`` batches, plus the ``[N x n_sampled]`` analytical
    error matrices for the reported and raw iterates (``nan`` where unstable).
    """
    N, L = observations.shape[0], observations.shape[1]
    w = model.window
    theta = {k: v.detach()[None].expand(N, *v.shape).clone()
             for k, v in eu.td_items(eu.parameter_td(model)).items()}
    s = state0                                                       # [N x S_D]
    ent_win = s[:, None].expand(N, w, *s.shape[1:]).clone()          # [N x w x S_D]
    n0 = int(model.polyak_burnin * L) if model.polyak_burnin >= 0.0 else None
    theta_sum: dict[str, torch.Tensor] | None = None
    avg_count = 0
    sampled_set = set(sampled_steps)
    idx_of = {t: j for j, t in enumerate(sampled_steps)}
    err_mat = np.full((N, len(sampled_steps)), np.nan)
    raw_err_mat = np.full((N, len(sampled_steps)), np.nan)

    # Per-trajectory prediction (only the posterior state is needed downstream)
    # and gradient, mapped over the batch dim. ``win_actions`` is the empty dict,
    # so it is the single un-mapped argument of ``_compute_grads``.
    cell_state = torch.vmap(
        lambda th, st, ob: torch.func.functional_call(model.cell, th, (st, {}, ob))["environment", "state"])
    grad_step = torch.vmap(model._compute_grads, in_dims=(0, 0, None, 0))

    for t in range(L):
        s_post = cell_state(theta, s, observations[:, t])           # [N x S_D]
        ent_win = torch.cat([ent_win[:, 1:], s[:, None]], dim=1)    # push entering state s
        s_start = ent_win[:, 0].detach()                            # [N x S_D]
        w0 = max(0, t - w + 1)
        win_obs = observations[:, w0:t + 1]                         # [N x w' x O_D]
        for _ in range(model.n_steps):
            grads = grad_step(theta, s_start, {}, win_obs)
            theta = model._optimizer_step(theta, grads, t)
        s = s_post

        # Reported filter: raw iterate, or the Polyak tail average past burn-in.
        if n0 is not None and t >= n0:
            if theta_sum is None:
                theta_sum = {k: v.clone() for k, v in theta.items()}
            else:
                for k, v in theta.items():
                    theta_sum[k] += v
            avg_count += 1
            theta_bar = {k: v / avg_count for k, v in theta_sum.items()}
        else:
            theta_bar = theta

        if t in sampled_set:
            j = idx_of[t]
            err_mat[:, j] = batched_filter_analytical_error(
                theta_bar["F"], theta_bar["H"], theta_bar["K"], lsg_td)
            raw_err_mat[:, j] = (err_mat[:, j] if theta_bar is theta else
                                 batched_filter_analytical_error(
                                     theta["F"], theta["H"], theta["K"], lsg_td))
    return theta_bar, theta, err_mat, raw_err_mat


class MethodResult(NamedTuple):
    label: str
    F_hat_mean: torch.Tensor          # [S_Dh x S_Dh] trajectory-averaged adapted F (finite trajectories)
    mean_ir: torch.Tensor             # [R x O_D x O_D] trajectory-averaged impulse response
    rel_ir_err: float                 # ||mean_ir - truth_ir|| / ||truth_ir||
    radius_med: float                 # median over trajectories of spectral radius of adapted F (robust)
    radius_max: float                 # max over trajectories (sensitive to a single diverging run)
    n_diverged: int                   # number of trajectories whose adapted F went non-finite
    mean_phase_err: float             # active-mode phase error of F_hat_mean vs true eigen-phases (rad)
    n_active: int
    err_curve: np.ndarray             # [n_sampled] median-over-trajectories analytical error per sampled step
    frac_stable: np.ndarray           # [n_sampled] fraction of trajectories whose filter is stable (finite)
    steady_err: float                 # steady analytical error (median of the last sampled steps)
    final_excess: float               # (steady_err - irreducible) / irreducible -- the sensitive gap metric
    # Diagnostic: the same readouts for the RAW iterate (no Polyak averaging), to
    # separate under-convergence from a parameter-averaging stability artifact.
    # Defaulted so sweeps recorded before this diagnostic still reload.
    raw_err_curve: np.ndarray = np.empty(0)
    raw_frac_stable: np.ndarray = np.empty(0)
    raw_rel_ir_err: float = float("inf")
    raw_steady_err: float = float("inf")
    raw_final_excess: float = float("inf")


def _build_method_theta(method: Method, args: argparse.Namespace,
                        problem_shape: ProblemShape, O_D: int,
) -> tuple[dict[str, torch.Tensor], RnnSelfDistillPredictor, tuple[str, ...]]:
    """Reproduce ONE method's initial fast-weights exactly as the (former)
    per-method path did: re-seed to ``args.seed``, build the model (kaiming-normal
    H draw), then apply the M1 (H=0) / frozen-K (random gain) init tweaks. Returns
    the initial ``theta`` (a dict of 2-D weights), the model (used only as the
    shared prediction cell / for window & step hyper-params), and ``adapt_keys``.

    Re-seeding here means every method draws the SAME initialization, so the only
    inter-method differences are the loss weights / adapt set -- identical to the
    old ``run_method`` semantics, just factored out so all methods can be stacked
    into one batched online loop."""
    torch.manual_seed(args.seed)
    cfg_kwargs = dict(step_decay=args.step_decay, polyak_burnin=args.polyak_burnin,
                      sd_horizon=args.sd_horizon, keep_launch=args.keep_launch)
    cfg_kwargs.update(method.cfg)
    model = RnnSelfDistillPredictor(RnnSelfDistillPredictor.Config(
        problem_shape=problem_shape, S_D=args.s_d, n_steps=args.n_steps,
        step_size=args.step_size, window=args.window,
        **cfg_kwargs,
    ))
    adapt_keys = tuple(method.cfg.get("adapt_keys", ("F", "H", "K")))
    with torch.no_grad():
        if "K" not in adapt_keys:
            model.K.copy_(torch.randn((args.s_d, O_D)) / (O_D ** 0.5))
        if method.h_init == "zero":
            model.H.zero_()
    model.eval()
    theta0 = {k: v.detach().clone() for k, v in eu.td_items(eu.parameter_td(model)).items()}
    return theta0, model, adapt_keys


def _make_sd_grad_fn(cell, h_max: int, keep_launch: bool):
    """Build the per-trajectory ``grad(window_loss)`` used by the multi-method
    batched loop, vmapped over the trajectory dim.

    The window loss is the same weighted section-6 objective as
    :meth:`RnnSelfDistillPredictor._window_loss`, but the loss weights
    ``(alpha, beta0, beta2)`` and the SD horizon are passed in as *per-trajectory*
    scalars so every stacked method shares one loss body. Two batching details
    keep it numerically identical to running each method on its own:

    - The a-priori / a-posteriori / ladder residuals are gated with ``torch.where``
      on a *finite* fallback (``0``) so a term a given method does not use (weight
      ``0``, or a ladder rung ``k`` beyond its horizon) contributes exactly ``0``
      and ``0`` gradient -- and, crucially, never turns a *diverged* trajectory's
      ``inf`` residual into a ``0 * inf = nan`` that would pollute the shared step
      (the old per-method path simply skipped those terms).
    - The autonomous ladder roll-out ``F^k x_{j-k}^+`` gates the launch state and
      ``F`` themselves to ``0`` when the rung is inactive, so the matmul chain stays
      finite in the backward pass for methods that do not use that rung.

    For an *active* term the gates are the identity, so the gradient matches the
    old single-method computation bit for bit (including its ``inf``/``nan`` on a
    genuinely diverged trajectory)."""

    def loss(theta, s_start, win_obs, alpha, beta0, beta2, sd_horizon):
        s = s_start
        w = win_obs.shape[0]
        total = win_obs.new_zeros(())
        H_t, F_t = theta["H"], theta["F"]
        # Launch buffer: posts[0] is the detached root (logical index -1),
        # posts[j+1] is the a-posteriori state x_j^+ of window step j.
        posts = [s_start.detach()]
        for j in range(w):
            obs_j = win_obs[j]
            out = torch.func.functional_call(cell, theta, (s, {}, obs_j))
            y_hat = out["environment", "observation"]       # H F x_{t-1}^+
            s_post = out["environment", "state"]            # x_t^+
            # a-priori observation prediction (beta0)
            r0 = torch.where(beta0 > 0, y_hat - obs_j, torch.zeros_like(y_hat))
            term = beta0 * 0.5 * r0.pow(2).sum(-1)
            # a-posteriori observation prediction (beta2)
            y_post = s_post @ H_t.mT
            r2 = torch.where(beta2 > 0, y_post - obs_j, torch.zeros_like(y_post))
            term = term + beta2 * 0.5 * r2.pow(2).sum(-1)
            # n-step latent self-distillation ladder (alpha) onto sg(x_j^+)
            target = s_post.detach()
            for k in range(1, min(h_max, j + 1) + 1):
                li = j - k                                  # -1 selects the detached root
                launch = posts[li + 1]
                if li < 0 or not keep_launch:
                    launch = launch.detach()
                gate = (alpha > 0) & (k <= sd_horizon)
                # Gate launch + F to a finite dummy on inactive rungs so the
                # F^k roll-out cannot backprop 0 * inf into the shared step.
                F_g = torch.where(gate, F_t, torch.zeros_like(F_t))
                pred = torch.where(gate, launch, torch.zeros_like(launch))
                for _ in range(k):
                    pred = pred @ F_g.mT
                rk = torch.where(gate, target - pred, torch.zeros_like(target))
                term = term + alpha * 0.5 * rk.pow(2).sum(-1)
            total = total + term
            posts.append(s_post)
            s = s_post
        return total / w

    return torch.vmap(torch.func.grad(loss, argnums=0), in_dims=(0, 0, 0, 0, 0, 0, 0))


@torch.no_grad()
def _adapt_and_measure_batched_multi(
        cell, window: int, n_steps: int, step_size: float,
        theta: dict[str, torch.Tensor], state0: torch.Tensor, observations: torch.Tensor,
        lsg_td: TensorDict, sampled_steps: list[int],
        alpha_v: torch.Tensor, beta0_v: torch.Tensor, beta2_v: torch.Tensor,
        wd_v: torch.Tensor, sd_horizon_v: torch.Tensor, step_decay_v: torch.Tensor,
        polyak_burnin_v: torch.Tensor, adapt_mask: dict[str, torch.Tensor],
        h_max: int, keep_launch: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    """Replay the online TTT loop for ALL stacked (method x trajectory) rows at
    once. Identical semantics to :func:`_adapt_and_measure_batched`, but every
    per-method knob is a per-row tensor over the leading ``M*N`` batch:
    ``alpha_v / beta0_v / beta2_v`` (loss weights), ``sd_horizon_v`` (ladder
    depth), ``wd_v`` (weight decay), ``step_decay_v`` (gain exponent),
    ``polyak_burnin_v`` (tail-average start, ``<0`` disables), and ``adapt_mask``
    (a ``{F,H,K}`` -> per-row 0/1 mask of which fast-weight is updated).

    This turns the former ``M`` sequential length-``L`` loops (one per method) into
    a single length-``L`` loop over ``M*N`` rows. Because the loop is entirely
    launch/dispatch bound (its per-step cost is flat in the batch size), stacking
    the methods is close to free and cuts wall time by roughly the method count.

    Returns the final reported (Polyak-averaged or raw) and raw fast-weights as
    ``[M*N x ...]`` batches, plus the ``[M*N x n_sampled]`` reported / raw
    analytical-error matrices (``nan`` where a row's filter is unstable)."""
    Ntot, L = observations.shape[0], observations.shape[1]
    w = window
    s = state0                                                       # [Ntot x S_D]
    ent_win = s[:, None].expand(Ntot, w, *s.shape[1:]).clone()       # [Ntot x w x S_D]
    sampled_set = set(sampled_steps)
    idx_of = {t: j for j, t in enumerate(sampled_steps)}
    err_mat = np.full((Ntot, len(sampled_steps)), np.nan)
    raw_err_mat = np.full((Ntot, len(sampled_steps)), np.nan)

    cell_state = torch.vmap(
        lambda th, st, ob: torch.func.functional_call(cell, th, (st, {}, ob))["environment", "state"])
    grad_step = _make_sd_grad_fn(cell, h_max, keep_launch)

    # Polyak tail averaging: which rows average, and their (per-row) start step.
    avg_active = polyak_burnin_v >= 0.0                              # [Ntot] bool
    avg_any = bool(avg_active.any().item())
    n0_v = (polyak_burnin_v * L).long()                             # [Ntot]
    theta_sum: dict[str, torch.Tensor] | None = None
    count = torch.zeros(Ntot)

    def _rowwise(vec: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return vec.view(Ntot, *([1] * (ref.dim() - 1)))

    for t in range(L):
        s_post = cell_state(theta, s, observations[:, t])           # [Ntot x S_D]
        ent_win = torch.cat([ent_win[:, 1:], s[:, None]], dim=1)
        s_start = ent_win[:, 0].detach()                            # [Ntot x S_D]
        w0 = max(0, t - w + 1)
        win_obs = observations[:, w0:t + 1]                        # [Ntot x w' x O_D]
        eta = step_size * ((t + 1) ** (-step_decay_v))              # [Ntot]
        for _ in range(n_steps):
            grads = grad_step(theta, s_start, win_obs, alpha_v, beta0_v, beta2_v, sd_horizon_v)
            new_theta: dict[str, torch.Tensor] = {}
            for k, v in theta.items():
                cand = v - _rowwise(eta, v) * (grads[k] + _rowwise(wd_v, v) * v)
                new_theta[k] = torch.where(_rowwise(adapt_mask[k], v) > 0, cand, v)
            theta = new_theta
        s = s_post

        # Reported filter: raw iterate, or the per-row Polyak tail average.
        if avg_any:
            active = avg_active & (t >= n0_v)                       # [Ntot] bool
            if theta_sum is None:
                theta_sum = {k: torch.zeros_like(v) for k, v in theta.items()}
            am = active.to(count.dtype)
            count = count + am
            for k in theta:
                theta_sum[k] = theta_sum[k] + _rowwise(am, theta[k]) * theta[k]
            cnt_safe = torch.where(count > 0, count, torch.ones_like(count))
            theta_bar = {k: torch.where(_rowwise(count > 0, v), theta_sum[k] / _rowwise(cnt_safe, v), v)
                         for k, v in theta.items()}
        else:
            theta_bar = theta

        if t in sampled_set:
            j = idx_of[t]
            err_mat[:, j] = batched_filter_analytical_error(
                theta_bar["F"], theta_bar["H"], theta_bar["K"], lsg_td)
            raw_err_mat[:, j] = (err_mat[:, j] if not avg_any else
                                 batched_filter_analytical_error(
                                     theta["F"], theta["H"], theta["K"], lsg_td))
    return theta_bar, theta, err_mat, raw_err_mat


def finalize_method_result(label: str, args: argparse.Namespace,
                           theta_bar_b: dict[str, torch.Tensor],
                           theta_raw_b: dict[str, torch.Tensor],
                           err_mat: np.ndarray, raw_err_mat: np.ndarray,
                           truth_ir: torch.Tensor, floor: float,
                           sampled_steps: list[int]) -> MethodResult:
    """Per-method final readouts from one method's slice of the batched replay.

    Spectral radius + impulse response are a cheap N-length loop over single 2-D
    matrices (kalman_ir / the eigensolver), run once per method -- not on the
    length-``L`` hot path that the batched loop vectorized away. Only finite
    (non-diverged) trajectories contribute to the averaged spectrum / IR; a
    diverged trajectory is recorded as radius == inf."""
    N = err_mat.shape[0]
    R = truth_ir.shape[0]
    F_hats, irs, radii = [], [], []
    raw_irs = []   # raw-iterate IRs (no averaging), for the diagnostic relIR
    for n in range(N):
        F_hat, H_hat, K_hat = theta_bar_b["F"][n], theta_bar_b["H"][n], theta_bar_b["K"][n]
        rad = _spectral_radius(F_hat)
        radii.append(rad)
        if np.isfinite(rad):
            F_hats.append(F_hat)
            irs.append(kalman_ir(F_hat, H_hat, K_hat, R))
        rF, rH, rK = theta_raw_b["F"][n], theta_raw_b["H"][n], theta_raw_b["K"][n]
        if np.isfinite(_spectral_radius(rF)):
            raw_irs.append(kalman_ir(rF, rH, rK, R))

    if F_hats:
        F_hat_mean = torch.stack(F_hats, dim=0).mean(dim=0).detach()
        mean_ir = torch.stack(irs, dim=0).mean(dim=0).detach()
        rel_ir_err = ((mean_ir - truth_ir).norm() / truth_ir.norm()).item()
    else:
        F_hat_mean = torch.full((args.s_d, args.s_d), float("nan"))
        mean_ir = torch.full_like(truth_ir, float("nan"))
        rel_ir_err = float("inf")
    finite_radii = [r for r in radii if np.isfinite(r)]
    radius_med = float(np.median(finite_radii)) if finite_radii else float("inf")
    raw_rel_ir_err = (((torch.stack(raw_irs, dim=0).mean(dim=0).detach() - truth_ir).norm()
                       / truth_ir.norm()).item()) if raw_irs else float("inf")

    tail = max(1, len(sampled_steps) // 5)

    def _curve_steady_excess(mat):
        with np.errstate(invalid="ignore"):
            curve = np.nanmedian(mat, axis=0)                   # nan where every trajectory is unstable
        frac = np.mean(np.isfinite(mat), axis=0)
        steady_tail = curve[-tail:]
        steady = float(np.nanmedian(steady_tail)) if np.any(np.isfinite(steady_tail)) else float("inf")
        excess = ((steady - floor) / floor) if np.isfinite(steady) else float("inf")
        return curve, frac, steady, excess

    err_curve, frac_stable, steady_err, final_excess = _curve_steady_excess(err_mat)
    raw_err_curve, raw_frac_stable, raw_steady_err, raw_final_excess = _curve_steady_excess(raw_err_mat)

    return MethodResult(
        label=label,
        F_hat_mean=F_hat_mean,
        mean_ir=mean_ir,
        rel_ir_err=rel_ir_err,
        radius_med=radius_med,
        radius_max=float(np.max(radii)),
        n_diverged=int(sum(1 for r in radii if not np.isfinite(r))),
        mean_phase_err=float("nan"),    # filled by the caller (needs true_eig)
        n_active=0,
        err_curve=err_curve,
        frac_stable=frac_stable,
        steady_err=steady_err,
        final_excess=final_excess,
        raw_err_curve=raw_err_curve,
        raw_frac_stable=raw_frac_stable,
        raw_rel_ir_err=raw_rel_ir_err,
        raw_steady_err=raw_steady_err,
        raw_final_excess=raw_final_excess,
    )


def run_methods(methods: list[Method], args: argparse.Namespace, problem_shape: ProblemShape,
                observations: torch.Tensor, state0: torch.Tensor, truth_ir: torch.Tensor,
                lsg_td: TensorDict, floor: float, sampled_steps: list[int],
                true_eig: torch.Tensor) -> list[MethodResult]:
    """Run every method in ONE batched online loop and return their results.

    All methods share the same system, observations and (re-seeded) initial
    fast-weights, so they are stacked along the batch dim -- ``M`` methods x ``N``
    trajectories -> ``M*N`` rows -- and adapted together in a single length-``L``
    replay. Since that loop is dispatch bound (flat in batch size), this is ~M
    times faster than the former one-loop-per-method approach while producing the
    same per-method results. Rows for method ``i`` occupy ``[i*N : (i+1)*N]``."""
    N = observations.shape[0]
    O_D = observations.shape[-1]
    M = len(methods)

    # Build each method's initial theta (identical seed => identical base init;
    # only the M1 H=0 / frozen-K tweaks differ) and its per-method scalar knobs.
    thetas: list[dict[str, torch.Tensor]] = []
    scal: dict[str, list[float]] = {k: [] for k in
                                    ("alpha", "beta0", "beta2", "wd", "sd_horizon", "step_decay", "polyak_burnin")}
    mask_cols: dict[str, list[float]] = {k: [] for k in ("F", "H", "K")}
    template = None
    for m in methods:
        theta0, model, adapt_keys = _build_method_theta(m, args, problem_shape, O_D)
        template = model
        thetas.append(theta0)
        cfg = m.cfg
        scal["alpha"].append(float(cfg.get("alpha", 0.0)))
        scal["beta0"].append(float(cfg.get("beta0", 1.0)))
        scal["beta2"].append(float(cfg.get("beta2", 0.0)))
        scal["wd"].append(float(cfg.get("weight_decay", 0.0)))
        scal["sd_horizon"].append(float(cfg.get("sd_horizon", args.sd_horizon)))
        scal["step_decay"].append(float(cfg.get("step_decay", args.step_decay)))
        scal["polyak_burnin"].append(float(cfg.get("polyak_burnin", args.polyak_burnin)))
        for key in ("F", "H", "K"):
            mask_cols[key].append(1.0 if key in adapt_keys else 0.0)

    # Stack: each method's theta broadcast over its N trajectories, then concat.
    theta_stacked = {k: torch.cat([th[k][None].expand(N, *th[k].shape).clone() for th in thetas], dim=0)
                     for k in thetas[0]}
    observations_stacked = observations.repeat(M, 1, 1)             # [M*N x L x O_D]
    state0_stacked = state0.repeat(M, 1)                            # [M*N x S_D]

    def _vec(vals: list[float]) -> torch.Tensor:
        return torch.tensor(vals, dtype=torch.get_default_dtype()).repeat_interleave(N)

    alpha_v, beta0_v, beta2_v = _vec(scal["alpha"]), _vec(scal["beta0"]), _vec(scal["beta2"])
    wd_v, sd_horizon_v = _vec(scal["wd"]), _vec(scal["sd_horizon"])
    step_decay_v, polyak_burnin_v = _vec(scal["step_decay"]), _vec(scal["polyak_burnin"])
    adapt_mask = {k: _vec(mask_cols[k]) for k in ("F", "H", "K")}
    h_max = int(max(scal["sd_horizon"])) if scal["sd_horizon"] else 1

    theta_bar_b, theta_raw_b, err_mat, raw_err_mat = _adapt_and_measure_batched_multi(
        template.cell, template.window, template.n_steps, template.step_size,
        theta_stacked, state0_stacked, observations_stacked, lsg_td, sampled_steps,
        alpha_v, beta0_v, beta2_v, wd_v, sd_horizon_v, step_decay_v, polyak_burnin_v,
        adapt_mask, h_max, args.keep_launch)

    results: list[MethodResult] = []
    for i, m in enumerate(methods):
        sl = slice(i * N, (i + 1) * N)
        tb = {k: v[sl] for k, v in theta_bar_b.items()}
        tr = {k: v[sl] for k, v in theta_raw_b.items()}
        r = finalize_method_result(m.label, args, tb, tr, err_mat[sl], raw_err_mat[sl],
                                   truth_ir, floor, sampled_steps)
        # Fill the eigen-phase metric now that we have true_eig.
        if not _eig_unsafe(r.F_hat_mean):
            est_eig = torch.linalg.eigvals(r.F_hat_mean)
            mean_err, _, n_active = active_phase_match_error(true_eig, est_eig, args.active_threshold)
        else:
            mean_err, n_active = float("nan"), 0
        r = r._replace(mean_phase_err=mean_err, n_active=n_active)
        results.append(r)

        label = m.label.replace("\n", " ")
        print(f"\n=== {label} ===")
        print(f"  adapt_keys={m.cfg.get('adapt_keys')}, alpha={m.cfg.get('alpha', 0.0)}, "
              f"beta0={m.cfg.get('beta0', 0.0)}, beta2={m.cfg.get('beta2', 0.0)}")
        print(f"  median|lambda(F_hat)|  : {r.radius_med:.3f}   (-> 1.0 is the unit circle; "
              f"{r.n_diverged}/{args.traces} diverged, max={r.radius_max:.3g})")
        print(f"  active-mode phase err  : {r.mean_phase_err:.3f} rad  ({r.n_active} active modes)")
        print(f"  relative IR error      : {r.rel_ir_err:.3f}   (0 == matches irreducible filter)")
        tail_n = max(1, len(sampled_steps) // 5)
        steady_stable = r.frac_stable[-tail_n:].mean()
        print(f"  analytical error       : {r.steady_err:.4f}   (floor={floor:.4f}, "
              f"excess={r.final_excess * 100:.1f}%, {steady_stable * 100:.0f}% stable at tail)")
        if r.raw_frac_stable.size:
            raw_stable = r.raw_frac_stable[-tail_n:].mean()
            print(f"  raw iterate (no avg)   : relIR={r.raw_rel_ir_err:.3f}, "
                  f"excess={r.raw_final_excess * 100:.1f}%, {raw_stable * 100:.0f}% stable at tail")
    return results


# SECTION: FIR benchmarks (online least-squares + optimal analytical FIR)
#
# Two reference filters of a fixed finite length R, computed *without* TTT, so we
# can see where the adapting RNN sits relative to the classical FIR baselines:
#   - optimal analytical FIR: the exact length-R minimiser of the closed-form
#     observation error -- reuses ``CnnAnalyticalLeastSquaresPredictor`` (one Newton
#     step; the objective is quadratic in the FIR taps), NOT reimplemented here.
#   - online least-squares FIR: ``CnnLeastSquaresPredictor`` fit incrementally on
#     the same trajectories; we read its exact analytical error as context grows.
# Both are reported with the same closed-form ``analytical_error`` as the methods.


class FirBenchmarks(NamedTuple):
    opt_err: dict          # {R: exact analytical error of the optimal length-R FIR}
    online_lengths: list   # IR lengths for the online-LS error-vs-step curves
    online_curves: dict    # {R: np.ndarray aligned to the TTT sampled_steps}


def _parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.replace(" ", "").split(",") if x] if spec else []


def _fir_pair(model_cls, problem_shape: ProblemShape, R: int, device, **cfg_kwargs):
    """Build an ``[N=1 x E=1]`` ensemble pair for one length-``R`` FIR model with
    leaf, grad-carrying params on ``device``. ``stack_module_arr`` can land the
    stacked params on a different device and as non-leaf views, which the Newton
    optimizer in ``newton_initialization`` rejects -- so re-anchor them here."""
    cfg = model_cls.Config(problem_shape=problem_shape, ir_length=R, **cfg_kwargs)
    models = eu.multi_map(lambda _: model_cls(cfg), np.empty((1, 1)), dtype=nn.Module)
    ref, stacked = eu.stack_module_arr(models)
    stacked = stacked.to(device).apply(lambda t: t.detach().clone().requires_grad_(t.requires_grad))
    return ref, stacked


def optimal_fir_error(problem_shape: ProblemShape, lsg, lsg_td: TensorDict,
                      R: int, device) -> float:
    """Exact analytical error of the *optimal* length-``R`` FIR for this system,
    via ``CnnAnalyticalLeastSquaresPredictor.newton_initialization`` (the project's
    optimal-FIR routine -- not reimplemented). Returns ``+inf`` on a non-finite
    result so it degrades gracefully on the plots."""
    ref, stacked = _fir_pair(CnnAnalyticalLeastSquaresPredictor, problem_shape, R, device)
    exclusive = SimpleNamespace(n_train_systems=1, train_info=SimpleNamespace(systems=lsg))
    init, _ = ref.newton_initialization(stacked, exclusive)
    with torch.no_grad():
        err = ConvolutionalPredictor.analytical_error(init, lsg_td)["environment", "observation"]
    e = float(err.reshape(-1)[0].item())
    return e if np.isfinite(e) and e >= 0.0 else float("inf")


@torch.no_grad()
def online_fir_curve(problem_shape: ProblemShape, lsg, lsg_td: TensorDict,
                     observations: torch.Tensor, R: int, sampled_steps: list[int],
                     device, ridge: float) -> np.ndarray:
    """Online least-squares length-``R`` FIR fit incrementally (one timestep at a
    time, all ``N`` traces in parallel) on the same observations the methods see;
    return its exact analytical error at each step in ``sampled_steps`` (reuses
    ``CnnLeastSquaresPredictor.train_least_squares_online``)."""
    ref, stacked = _fir_pair(CnnLeastSquaresPredictor, problem_shape, R, device, ridge=ridge)
    N, L, _ = observations.shape
    # engine layout: [N_model x E x n_sys x n_traces x L]; flatten(2,-2) folds the
    # (n_sys, n_traces) dims into the batch B, leaving bsz=[N_model,E], (B,L)=[-2:].
    train_ds = TensorDict.from_dict(
        {"environment": {"observation": observations[None, None, None]}, "controller": {}},
        batch_size=(1, 1, 1, N, L),
    )
    exclusive = SimpleNamespace(
        n_train_systems=1,
        train_info=SimpleNamespace(systems=lsg, dataset=train_ds),
    )
    thp = TrainConfig()
    thp.sampling.batch_size = 1
    model_pair = (ref, stacked)
    cache = SimpleNamespace(t=0)
    sampled_set = set(sampled_steps)
    idx_of = {t: j for j, t in enumerate(sampled_steps)}
    curve = np.full(len(sampled_steps), np.nan)
    t = -1   # 0-based index of the timestep consumed by the current LS update
    while not ref.terminate_least_squares_online(thp, exclusive, model_pair, cache):
        ref.train_least_squares_online(thp, exclusive, model_pair, cache)
        t += 1
        if t in sampled_set:
            al = ConvolutionalPredictor.analytical_error(stacked, lsg_td)["environment", "observation"]
            e = float(al.reshape(-1)[0].item())
            curve[idx_of[t]] = e if np.isfinite(e) and e >= 0.0 else np.nan
    return curve


def compute_fir_benchmarks(args: argparse.Namespace, problem_shape: ProblemShape, lsg,
                           lsg_td: TensorDict, observations: torch.Tensor,
                           sampled_steps: list[int], floor: float) -> FirBenchmarks:
    dev = lsg_td["environment", "F"].device
    lengths = _parse_int_list(args.fir_lengths)
    online_lengths = _parse_int_list(args.online_fir_lengths)
    # The optimal-FIR sweep covers both the standalone length sweep and any length
    # used for an online curve (so each online curve has its optimal-FIR target).
    opt_err = {R: optimal_fir_error(problem_shape, lsg, lsg_td, R, dev)
               for R in sorted(set(lengths) | set(online_lengths))}
    online_curves = {R: online_fir_curve(problem_shape, lsg, lsg_td, observations, R,
                                          sampled_steps, dev, args.fir_ridge)
                     for R in online_lengths}
    print("\nFIR benchmarks (exact analytical error):")
    for R in sorted(opt_err):
        print(f"  optimal FIR R={R:3d}: err={opt_err[R]:.5f}  excess={opt_err[R] - floor:.5f}")
    for R in online_lengths:
        tail = online_curves[R][np.isfinite(online_curves[R])]
        final = tail[-1] if tail.size else float("nan")
        print(f"  online  FIR R={R:3d}: final err={final:.5f}")
    return FirBenchmarks(opt_err=opt_err, online_lengths=online_lengths, online_curves=online_curves)


# SECTION: Plotting

def plot_complex_planes(results: list[MethodResult], true_eig: torch.Tensor,
                        out_dir: str, tag: str) -> str:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8), squeeze=False)
    true_eig = true_eig.detach().cpu()
    theta_grid = np.linspace(0, 2 * np.pi, 400)
    for ax, r in zip(axes[0], results):
        if not _eig_unsafe(r.F_hat_mean):
            est_eig = torch.linalg.eigvals(r.F_hat_mean).detach().cpu()
            _draw_complex_plane(ax, true_eig, est_eig,
                                f"{r.label}\nmed|lambda|={r.radius_med:.2f} "
                                f"({r.n_diverged} diverged), phase err={r.mean_phase_err:.2f}")
        else:
            # Diverged: show the unit circle + true spectrum only.
            ax.plot(np.cos(theta_grid), np.sin(theta_grid), color="0.7", lw=1.0)
            ax.scatter(true_eig.real, true_eig.imag, marker="x", s=80, color="C0", label="true F")
            ax.set_aspect("equal")
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
            ax.set_title(f"{r.label}\nDIVERGED (max|lambda|=inf)")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Adapted F spectrum: true F (x) vs mean adapted F_hat (o)")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_eig_complex_plane.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_impulse(results: list[MethodResult], truth_ir: torch.Tensor,
                 out_dir: str, tag: str) -> str:
    """Overlay each method's mean impulse response on the ground-truth (irreducible)
    Kalman IR, for the leading output<-input channel."""
    R = truth_ir.shape[0]
    lags = np.arange(1, R + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(lags, truth_ir[:, 0, 0].numpy(force=True), color="k", lw=2.2, marker="o", ms=3,
            label="ground-truth Kalman (irreducible)", zorder=5)
    for r in results:
        ax.plot(lags, r.mean_ir[:, 0, 0].numpy(force=True), lw=1.4, marker=".", ms=4,
                label=f"{r.label.splitlines()[0]} (relIR={r.rel_ir_err:.2f})")
    ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax.set_xlabel("lag")
    ax.set_ylabel("IR coefficient (out[0] <- in[0])")
    ax.set_title("Impulse response vs the irreducible Kalman filter")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_impulse.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_summary(results: list[MethodResult], out_dir: str, tag: str) -> str:
    labels = [r.label for r in results]
    x = np.arange(len(results))
    metrics = [
        ("relative IR error", [r.rel_ir_err for r in results], "C0", 0.0),
        (r"median $|\lambda(\hat F)|$", [r.radius_med for r in results], "C3", 1.0),
        ("mean phase err (rad)", [r.mean_phase_err for r in results], "C2", 0.0),
        ("excess over irreducible", [r.final_excess for r in results], "C4", 0.0),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.4))
    for ax, (name, vals, color, ref) in zip(axes, metrics):
        vplot = [v if np.isfinite(v) else np.nan for v in vals]
        ax.bar(x, vplot, color=color, alpha=0.85)
        if ref is not None:
            ax.axhline(ref, color="0.5", ls="--", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([l.splitlines()[0] for l in labels], rotation=30, ha="right", fontsize=8)
        ax.set_title(name, fontsize=10)
        # Annotate diverged (inf) bars so they are not silently missing.
        for xi, v in zip(x, vals):
            if not np.isfinite(v):
                ax.text(xi, ax.get_ylim()[1] * 0.5, "inf", ha="center", va="center",
                        rotation=90, fontsize=8, color="0.3")
    fig.suptitle("M1-M4 readouts: SD spectrum (|lambda|, phase) and filtering "
                 "(relIR, analytical excess over irreducible)")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_summary.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _overlay_fir(ax_err, ax_exc, steps: np.ndarray, floor: float,
                 fir: "Optional[FirBenchmarks]") -> None:
    """Overlay the FIR baselines on the error / excess step-panels: online-LS
    error-vs-step curves (dashed, copper) and their optimal-FIR targets (dotted
    horizontals). Drawn in a separate copper palette so they read as benchmarks,
    not as TTT methods (which use the C0.. cycle)."""
    if fir is None or not fir.online_lengths:
        return
    palette = plt.cm.copper(np.linspace(0.15, 0.8, len(fir.online_lengths)))
    for R, col in zip(fir.online_lengths, palette):
        curve = fir.online_curves[R]
        ax_err.plot(steps, curve, color=col, ls="--", lw=1.3, marker="x", ms=3,
                    label=f"online FIR R={R}")
        exc = np.where((curve - floor) / floor > 0, (curve - floor) / floor, np.nan)
        if ax_exc is not None:
            ax_exc.plot(steps, exc, color=col, ls="--", lw=1.3, marker="x", ms=3,
                        label=f"online FIR R={R}")
        oe = fir.opt_err.get(R)
        if oe is not None and np.isfinite(oe):
            ax_err.axhline(oe, color=col, ls=":", lw=1.0)
            if ax_exc is not None and (oe - floor) / floor > 0:
                ax_exc.axhline((oe - floor) / floor, color=col, ls=":", lw=1.0)


def plot_analytical_error_curve(results: list[MethodResult], steps: np.ndarray,
                                floor: float, ceiling: float,
                                out_dir: str, tag: str,
                                fir: "Optional[FirBenchmarks]" = None) -> str:
    """Two-panel readout of the exact filter error as each method adapts.

    Left: trajectory-median analytical observation error vs online step (log-y),
    with the irreducible floor and the zero-predictor ceiling as dashed lines.
    Right: excess over the irreducible floor ``(err - floor)/floor`` (log-y), the
    sensitive metric -- M3+post should descend toward 0; M1/M2/literal-M3 either
    sit far above or read unstable (gaps where every trajectory diverged).

    When ``fir`` is supplied the online-LS FIR curves and their optimal-FIR targets
    are overlaid (copper) as classical finite-length baselines.
    """
    fig, (ax_err, ax_exc) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    colors = [f"C{i}" for i in range(len(results))]

    for r, c in zip(results, colors):
        lab = r.label.splitlines()[0]
        err = r.err_curve
        ax_err.plot(steps, err, color=c, lw=1.6, marker=".", ms=4, label=lab)
        excess = (err - floor) / floor
        # Only positive excess is meaningful on a log axis; tiny negatives are numerical.
        excess = np.where(excess > 0, excess, np.nan)
        ax_exc.plot(steps, excess, color=c, lw=1.6, marker=".", ms=4, label=lab)
        # Diagnostic overlay: the raw iterate (no Polyak averaging), faint dashed.
        # Only drawn when it actually differs from the reported curve (averaging on).
        raw = r.raw_err_curve
        if raw.size == err.size and not np.allclose(np.nan_to_num(raw), np.nan_to_num(err)):
            ax_err.plot(steps, raw, color=c, lw=1.0, ls="--", alpha=0.5, label=f"{lab} raw")
            raw_exc = np.where((raw - floor) / floor > 0, (raw - floor) / floor, np.nan)
            ax_exc.plot(steps, raw_exc, color=c, lw=1.0, ls="--", alpha=0.5, label=f"{lab} raw")

    _overlay_fir(ax_err, ax_exc, steps, floor, fir)

    ax_err.axhline(floor, color="k", ls="--", lw=1.2, label="irreducible floor")
    ax_err.axhline(ceiling, color="0.5", ls=":", lw=1.2, label="zero-predictor ceiling")
    ax_err.set_yscale("log")
    ax_err.set_xlabel("online step")
    ax_err.set_ylabel("analytical observation error")
    ax_err.set_title("Filter error vs adaptation step (median over trajectories)")
    ax_err.legend(fontsize=8, loc="best")

    ax_exc.axhline(0.0, color="k", ls="--", lw=0.8)
    ax_exc.set_yscale("log")
    ax_exc.set_xlabel("online step")
    ax_exc.set_ylabel(r"excess over floor $(err-\mathrm{irr})/\mathrm{irr}$")
    ax_exc.set_title("Excess over irreducible (-> 0 means reaching the optimum)")
    ax_exc.legend(fontsize=8, loc="best")

    fig.suptitle("Analytical filter error as the methods adapt "
                 "(exact closed form; gaps = every trajectory unstable)")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_analytical_error_curve.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_excess_absolute(results: list[MethodResult], steps: np.ndarray,
                         floor: float, out_dir: str, tag: str,
                         fir: "Optional[FirBenchmarks]" = None) -> str:
    """Absolute excess over the irreducible floor ``err - floor`` vs online step,
    on a **log y-axis** (one line per method). This is the requested m3-vs-m4
    readout: it shows how far above the optimum each method settles in the same
    units as the error itself (rather than the relative ``(err-floor)/floor``).
    Gaps appear where every trajectory is unstable (``err_curve`` is ``nan``); only
    positive excess is plotted (tiny negatives are numerical and dropped).

    When ``fir`` is supplied the online-LS FIR absolute excess and the optimal-FIR
    targets are overlaid (copper) for the same lengths."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = [f"C{i}" for i in range(len(results))]
    for r, c in zip(results, colors):
        lab = r.label.splitlines()[0]
        excess = r.err_curve - floor
        excess = np.where(excess > 0, excess, np.nan)
        ax.plot(steps, excess, color=c, lw=1.7, marker=".", ms=4, label=lab)
    if fir is not None:
        palette = plt.cm.copper(np.linspace(0.15, 0.8, max(1, len(fir.online_lengths))))
        for R, col in zip(fir.online_lengths, palette):
            exc = np.where(fir.online_curves[R] - floor > 0, fir.online_curves[R] - floor, np.nan)
            ax.plot(steps, exc, color=col, ls="--", lw=1.3, marker="x", ms=3,
                    label=f"online FIR R={R}")
            oe = fir.opt_err.get(R)
            if oe is not None and np.isfinite(oe) and oe - floor > 0:
                ax.axhline(oe - floor, color=col, ls=":", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("online step")
    ax.set_ylabel(r"absolute excess over floor  $err - \mathrm{irr}$")
    ax.set_title(f"Absolute error above the irreducible optimum (irr = {floor:.4g}; "
                 "log scale, -> 0 is optimal)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_excess_absolute.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_fir_vs_length(results: list[MethodResult], fir: "Optional[FirBenchmarks]",
                       floor: float, out_dir: str, tag: str) -> Optional[str]:
    """Analytical excess over the irreducible floor as a function of FIR length R.

    The black curve is the *optimal* length-R FIR (its excess decays toward 0 as R
    grows -- a finite FIR approaches the irreducible Kalman optimum). Square markers
    are the online-LS FIR's final excess at each measured length. The dashed
    horizontals are the TTT methods' steady-state excess, so one reads off how many
    FIR taps a classical filter needs to match (or beat) each adapting method."""
    if fir is None or not fir.opt_err:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    Rs = np.array(sorted(fir.opt_err))
    opt_exc = np.array([fir.opt_err[int(R)] - floor for R in Rs])
    opt_exc = np.where(opt_exc > 0, opt_exc, np.nan)
    ax.plot(Rs, opt_exc, color="k", lw=2.0, marker="o", ms=5,
            label="optimal FIR (analytical)", zorder=5)
    for i, R in enumerate(fir.online_lengths):
        tail = fir.online_curves[R][np.isfinite(fir.online_curves[R])]
        if tail.size and tail[-1] - floor > 0:
            ax.scatter([R], [tail[-1] - floor], color="C7", marker="s", s=45, zorder=6,
                       label="online FIR (final)" if i == 0 else None)
    for r, c in zip(results, [f"C{i}" for i in range(len(results))]):
        e = r.steady_err - floor
        if np.isfinite(e) and e > 0:
            ax.axhline(e, color=c, ls="--", lw=1.2, label=f"{r.label.splitlines()[0]} (TTT steady)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("FIR length R")
    ax.set_ylabel(r"analytical excess over floor  $err - \mathrm{irr}$")
    ax.set_title("FIR excess vs length (optimal & online) with TTT steady-state methods")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_fir_vs_length.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# SECTION: Persistence (record the slow TTT sweep so plots can be regenerated)

def save_results(path: str, results: list[MethodResult], true_eig: torch.Tensor,
                 truth_ir: torch.Tensor, floor: float, ceiling: float,
                 steps_axis: np.ndarray, args: argparse.Namespace) -> None:
    """Persist the (expensive) per-method TTT results plus the system baselines, so
    a later ``--reload`` re-renders every plot without re-running the sweep. All
    tensors are moved to CPU so the checkpoint is portable across devices."""
    payload = {
        "results": [
            {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
             for k, v in r._asdict().items()}
            for r in results
        ],
        "true_eig": true_eig.detach().cpu(),
        "truth_ir": truth_ir.detach().cpu(),
        "floor": float(floor),
        "ceiling": float(ceiling),
        "steps_axis": np.asarray(steps_axis),
        "args": vars(args),
    }
    torch.save(payload, path)


def load_results(path: str) -> tuple[list[MethodResult], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    results = [MethodResult(**d) for d in payload["results"]]
    return results, payload


# SECTION: Driver

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_sys = p.add_argument_group("system")
    g_sys.add_argument("--s-d", type=int, default=6, help="state dimension S_D")
    g_sys.add_argument("--o-d", type=int, default=2, help="observation dimension O_D")
    g_sys.add_argument("--eps", type=float, default=0.1, help="continuous discretization step")
    g_sys.add_argument("--w-std", type=float, default=1.0, help="process-noise std (scaled by eps)")
    g_sys.add_argument("--v-std", type=float, default=1.0, help="observation-noise std (scaled by eps)")
    g_sys.add_argument("--f-mode", type=str, default="gaussian", choices=("gaussian", "uniform"))
    g_sys.add_argument("--h-mode", type=str, default="gaussian", choices=("gaussian", "uniform"))

    g_data = p.add_argument_group("data")
    g_data.add_argument("-N", "--traces", type=int, default=16, help="number of trajectories")
    g_data.add_argument("-L", "--length", type=int, default=200, help="trace length")
    g_data.add_argument("-R", "--lags", type=int, default=32, help="impulse-response lags")
    g_data.add_argument("--analytic-stride", type=int, default=4,
                        help="compute the exact analytical filter error every this many online steps")

    g_filt = p.add_argument_group("filter / TTT")
    g_filt.add_argument("--window", type=int, default=4, help="sliding-window length (>1 to train K)")
    g_filt.add_argument("--n-steps", type=int, default=1, help="online SGD steps per timestep")
    g_filt.add_argument("--step-size", type=float, default=None,
                        help="online learning rate (absolute); overrides --lr-scale if set")
    g_filt.add_argument("--lr-scale", type=float, default=0.3,
                        help="online learning rate as a multiple of eps (used if --step-size unset)")
    g_filt.add_argument("--active-threshold", type=float, default=0.5,
                        help="|lambda| above which an F_hat mode counts as active (near-circle)")
    g_filt.add_argument("--step-decay", type=float, default=0.0,
                        help="exponent a in the decaying online rate eta_t = step_size*(t+1)^(-a); "
                             "0 = constant step, a in (1/2,1) for a vanishing (Robbins-Monro) gain "
                             "(a method preset may override this)")
    g_filt.add_argument("--polyak-burnin", type=float, default=-1.0,
                        help="Polyak-Ruppert tail-averaging start as a fraction of L (report "
                             "theta_bar = mean_{t>=n0} theta_t); <0 disables averaging "
                             "(a method preset may override this)")
    g_filt.add_argument("--sd-horizon", type=int, default=1,
                        help="n-step latent self-distillation ladder depth (autonomous F "
                             "roll-out horizons k=1..n); 1 = the original single-step SD term. "
                             "Per-target horizons deeper than the window clamp to the detached "
                             "root (needs --window >= n+1 for full-depth interior terms; "
                             "a method preset may override this)")
    g_filt.add_argument("--keep-launch", action=argparse.BooleanOptionalAction, default=True,
                        help="whether the (non-root) launch state x_{t-k}^+ carries gradient "
                             "(--keep-launch, today's behavior) or is detached (--no-keep-launch); "
                             "the root launch s_start is always detached either way")

    g_loss = p.add_argument_group("loss weights")
    g_loss.add_argument("--anchor", type=float, default=0.05,
                        help="M3 a-priori grounding weight beta0 (the minimal observation anchor)")
    g_loss.add_argument("--post", type=float, default=0.05,
                        help="a-posteriori weight beta2 for the M3+post probe")
    g_loss.add_argument("--weight-decay", type=float, default=1e-3,
                        help="decoupled radial shrink paired with the a-posteriori term (rate analog)")

    g_run = p.add_argument_group("run")
    g_run.add_argument("--seed", type=int, default=0, help="random seed")
    g_run.add_argument("--methods", type=str, default="ladder",
                       choices=("ladder", "m3m4", "minimal", "mech", "nstep"),
                       help="'ladder' = the M1-M4 + M3+post catalogue; 'm3m4' = the clean "
                            "4-way (alpha,beta0,beta2) comparison (no wd, adapt F,H,K); "
                            "'minimal' = the vanishing-gain M4 (decay+avg) vs its constant-step control; "
                            "'mech' = the constant-gain-centric three-way (constant a-priori vs "
                            "constant SD+anchor vs decay+avg oracle), swept over --step-size; "
                            "'nstep' = constant a-priori vs SD+anchor at every ladder depth "
                            "n=1..window (the SD depth sweep), swept over --step-size")
    g_run.add_argument("--device", type=str, default=None, choices=("cpu", "cuda"),
                       help="compute device (default: cuda if available, else cpu)")
    g_run.add_argument("--out-name", type=str, default="sd_losses",
                       help="output subdirectory under OUTPUT_PATH")
    g_run.add_argument("--reload", action="store_true",
                       help="skip the TTT sweep and load the recorded results from "
                            "<out>/<tag>_save.pt, recomputing only the (cheap) FIR "
                            "benchmarks and re-rendering every plot")

    g_fir = p.add_argument_group("FIR benchmarks")
    g_fir.add_argument("--fir-lengths", type=str, default="1,2,4,8,16,32",
                       help="comma-separated IR lengths for the optimal analytical FIR "
                            "sweep (CnnAnalyticalLeastSquaresPredictor); empty to disable")
    g_fir.add_argument("--online-fir-lengths", type=str, default="4,8,16",
                       help="comma-separated IR lengths for the online least-squares FIR "
                            "error-vs-step curves; empty to disable")
    g_fir.add_argument("--fir-ridge", type=float, default=1.0,
                       help="ridge for the online least-squares FIR fit (early-context "
                            "regularization)")
    g_fir.add_argument("--no-fir", action="store_true",
                       help="skip the FIR benchmarks entirely (fast iteration on the TTT plots)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    torch.manual_seed(args.seed)
    out_dir = os.path.join(OUTPUT_PATH, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    step_size = args.step_size if args.step_size is not None else args.lr_scale * args.eps
    args.step_size = step_size

    S_D, O_D, eps = args.s_d, args.o_d, args.eps
    problem_shape = ProblemShape(environment=EnvironmentShape(observation=O_D), controller={})
    system_cfg = SystemConfig(S_D=S_D, problem_shape=problem_shape)
    distribution = ContinuousDistribution(args.f_mode, args.h_mode, eps, args.w_std, args.v_std)
    lsg = distribution.sample(system_cfg, ())

    N, L, R = args.traces, args.length, args.lags
    ds = lsg.generate_dataset(N, L)
    observations = ds["environment", "observation"]                       # [N x L x O_D]
    state0 = torch.zeros((N, S_D))

    F_t, H_t, K_t = lsg.environment.F, lsg.environment.H, lsg.environment.K
    true_eig = torch.linalg.eigvals(F_t).detach()
    truth_ir = kalman_ir(F_t, H_t, K_t, R).detach()                       # [R x O_D x O_D] (irreducible)

    # Exact analytical baselines (expectations over the noise, no sampling):
    # the irreducible floor (optimal Kalman) and the zero-predictor ceiling.
    lsg_td = lsg.td()
    floor = lsg.irreducible_loss.environment.observation.item()
    ceiling = lsg.zero_predictor_loss.environment.observation.item()
    # Online steps at which we snapshot the exact analytical filter error.
    sampled_steps = list(range(args.analytic_stride - 1, L, args.analytic_stride))
    if not sampled_steps or sampled_steps[-1] != L - 1:
        sampled_steps.append(L - 1)
    steps_axis = np.array(sampled_steps) + 1                              # 1-based step index for plotting

    print(f"System S_D={S_D}, O_D={O_D}, eps={eps}, step_size={step_size:.4g}, "
          f"N={N}, L={L}, window={args.window}, device={device}, methods={args.methods}")
    print(f"true |eig(F)| in [{true_eig.abs().min():.3f}, {true_eig.abs().max():.3f}], "
          f"irreducible floor = {floor:.4f}, zero-predictor ceiling = {ceiling:.4f}")

    tag = f"sd{S_D}_od{O_D}"
    save_path = os.path.join(out_dir, f"{tag}_save.pt")

    if args.reload:
        # Re-render from the recorded sweep. The system / data above are rebuilt
        # deterministically (same seed) only so the FIR benchmarks and the system
        # baselines stay consistent; the expensive per-method results are loaded.
        if not os.path.exists(save_path):
            raise SystemExit(f"--reload set but no recorded sweep at {save_path}")
        results, payload = load_results(save_path)
        floor, ceiling = payload["floor"], payload["ceiling"]
        true_eig, truth_ir = payload["true_eig"], payload["truth_ir"]
        steps_axis = payload["steps_axis"]
        sampled_steps = (steps_axis - 1).tolist()
        print(f"reloaded {len(results)} recorded method results from {save_path}")
    else:
        if args.methods == "m3m4":
            methods = m3m4_methods()
        elif args.methods == "minimal":
            # The decay+avg method takes its schedule from the global knobs (so the
            # knob-only sweep is one --step-decay/--polyak-burnin per run), falling
            # back to the original 0.67 / 0.1 defaults when they are unset.
            sd = args.step_decay if args.step_decay > 0.0 else 0.67
            pb = args.polyak_burnin if args.polyak_burnin >= 0.0 else 0.1
            methods = minimal_methods(sd, pb)
        elif args.methods == "mech":
            # The two constant arms are pinned (step_decay=0, polyak_burnin=-1);
            # only the decay+avg oracle reads the global schedule knobs, defaulting
            # to the brittle 0.51 / 0.5 recipe when unset.
            sd = args.step_decay if args.step_decay > 0.0 else 0.51
            pb = args.polyak_burnin if args.polyak_burnin >= 0.0 else 0.5
            methods = mech_methods(sd, pb)
        elif args.methods == "nstep":
            # One SD+anchor arm per ladder depth n = 1..window (a horizon deeper
            # than the window clamps to the root, so window is the natural
            # ceiling), plus the pure a-priori control. Depth 1 is the previous
            # single-step SD behavior, so no silent change to the default.
            methods = nstep_methods(args.window, args.anchor)
        else:
            methods = default_methods(args.anchor, args.post, args.weight_decay)
        # All methods share the system / observations / (re-seeded) init, so they
        # are adapted together in ONE batched online loop (M methods x N traces),
        # instead of one length-L loop per method. The per-step cost is flat in
        # the batch size (the loop is dispatch bound), so this is ~M times faster
        # while producing identical per-method results. Fills phase-err + prints.
        results = run_methods(methods, args, problem_shape, observations, state0, truth_ir,
                              lsg_td, floor, sampled_steps, true_eig)
        save_results(save_path, results, true_eig, truth_ir, floor, ceiling, steps_axis, args)
        print(f"\nrecorded sweep -> {save_path}")

    # Cheap, deterministic FIR baselines (recomputed every run, incl. --reload).
    fir = None if args.no_fir else compute_fir_benchmarks(
        args, problem_shape, lsg, lsg_td, observations, sampled_steps, floor)

    paths = [
        plot_complex_planes(results, true_eig, out_dir, tag),
        plot_impulse(results, truth_ir, out_dir, tag),
        plot_analytical_error_curve(results, steps_axis, floor, ceiling, out_dir, tag, fir),
        plot_excess_absolute(results, steps_axis, floor, out_dir, tag, fir),
        plot_summary(results, out_dir, tag),
    ]
    fir_path = plot_fir_vs_length(results, fir, floor, out_dir, tag)
    if fir_path is not None:
        paths.append(fir_path)
    print("\nsaved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
