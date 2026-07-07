"""Empirical test of the self-distillation eigen-phase hypothesis.

Hypothesis (see ``chat.txt``): for an LTI system with transition matrix ``F`` and a
randomly initialized filter

    x_hat_{t+1} = F_hat x_hat_t + K_hat y_t,

if ``F_hat`` is trained by self-distillation ``F_hat x_hat_t ~= x_hat_{t+1}`` with
*detached* targets, then the eigenvalues of ``F_hat`` are pushed to the unit circle
and their *phases* converge to the eigen-phases of the true ``F``.

Key correctness point: with naive end-to-end autodiff the residual
``x_hat_{t+1} - F_hat x_hat_t = K_hat y_t`` is independent of ``F_hat`` so the
gradient is identically zero. The interesting dynamic is the stop-gradient
(detached-target) form of the per-step self-distillation loss
``|| sg(x_hat_{t+1}) - F_hat x_hat_t ||^2`` whose gradient is
``-(K_hat y_t) x_hat_t^T``.

Both schedules are a single online (test-time-training) pass that streams
t = 0..L-1 over all traces in parallel, updating ``F_hat`` at each step and
immediately using it to advance the state, so parameters and state co-evolve.
Two update rules are available via ``--mode``:

- ``rls`` (DEFAULT): recursive least squares. The per-step update is the exact
  least-squares solution, i.e. the raw gradient ``-(K_hat y_t) x_t^T``
  preconditioned by the inverse state covariance ``P`` (a Newton step on the
  same loss). A forgetting factor tracks the slowly moving fixed point as
  ``F_hat`` co-evolves.
- ``lms``: normalized least-mean-squares. The raw gradient step, optionally
  divided by the running state energy (NLMS) to stay stable as the state
  variance inflates near the unit circle.

This script reuses the project only to sample an ``LTISystem`` (true ``F``/``H``/``K``)
and to generate observation traces ``y``; the filter recurrence above is implemented
directly (it is *not* the project's ``(I - K H) F`` Kalman form).

Examples
--------
Clean fully-observed demo (RLS default)::

    python scripts/self_distillation_eigenphase.py --s-d 2 --o-d 2 --seed 0

Paired fully-observed vs partial-observed on the *same* F::

    python scripts/self_distillation_eigenphase.py --compare-observation \
        --s-d 4 --o-d-full 4 --o-d-partial 2 --seed 0

Many-seed sweep of the partial regime (degeneracy is rare)::

    python scripts/self_distillation_eigenphase.py --o-d 2 --s-d 4 --n-seeds 50

(N)LMS instead of RLS::

    python scripts/self_distillation_eigenphase.py --s-d 2 --o-d 2 --mode lms
"""

import argparse
import os
import tempfile
from typing import NamedTuple

# Writable matplotlib cache (the user's ~/.config may be read-only on the cluster).
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "kf_rnn_mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import torch
import ecliseutils as eu
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tensordict import TensorDict

from kf_rnn.analysis import (
    active_phase_match_error,
    draw_complex_plane as _draw_complex_plane,
    spectral_radius as _spectral_radius,
)
from kf_rnn.infrastructure.config import EnvironmentShape, ProblemShape, SystemConfig
from kf_rnn.infrastructure.settings import OUTPUT_PATH
from kf_rnn.system.linear_time_invariant import LTISystem, MOPDistribution

# The project settings default to ``cuda:0``; force CPU so this analysis script
# runs anywhere (all heavy work here is tiny and done on CPU in float64 anyway).
torch.set_default_device("cpu")


# Everything in the iterative analysis is done on CPU in float64 for clean,
# overflow-free eigenvalue readouts, independent of the project's default device.
_ANALYSIS_DTYPE = torch.float64
_ANALYSIS_DEVICE = "cpu"


# SECTION: System / data construction (uses project code)

def build_system(S_D: int, O_D: int, w_std: float, v_std: float) -> LTISystem:
    """Sample a fresh ``LTISystem`` with the given dimensions."""
    cfg = SystemConfig(
        S_D=S_D,
        problem_shape=ProblemShape(environment=EnvironmentShape(observation=O_D), controller={}),
    )
    dist = MOPDistribution("gaussian", "gaussian", w_std, v_std)
    return dist.sample(cfg, ())


def build_shared_systems(
        S_D: int,
        O_D_full: int,
        O_D_partial: int,
        w_std: float,
        v_std: float,
) -> tuple[LTISystem, LTISystem]:
    """Build a fully-observed and a partially-observed system that *share* ``F``.

    The fully-observed system is sampled normally; the partial system reuses the
    exact same ``F`` (and process noise) and simply slices ``H`` to its first
    ``O_D_partial`` rows (and ``sqrt_S_V`` to the leading block), re-instantiating
    an ``LTISystem`` so the steady-state Kalman gain ``K`` is recomputed for the
    partial ``H``. The eigen-phase targets are therefore identical across runs.
    """
    cfg_full = SystemConfig(
        S_D=S_D,
        problem_shape=ProblemShape(environment=EnvironmentShape(observation=O_D_full), controller={}),
    )
    dist = MOPDistribution("gaussian", "gaussian", w_std, v_std)
    params_full = dist.sample_parameters(cfg_full, ())
    full_sys = LTISystem(cfg_full, params_full)

    # Reuse the same F (and sqrt_S_W); slice the observation model only.
    params_partial = params_full.clone()
    params_partial["environment", "H"] = params_full["environment", "H"][:O_D_partial].clone()
    params_partial["environment", "sqrt_S_V"] = (
        params_full["environment", "sqrt_S_V"][:O_D_partial, :O_D_partial].clone()
    )
    cfg_partial = SystemConfig(
        S_D=S_D,
        problem_shape=ProblemShape(environment=EnvironmentShape(observation=O_D_partial), controller={}),
    )
    partial_sys = LTISystem(cfg_partial, params_partial)
    return full_sys, partial_sys


def system_F(sys: LTISystem) -> torch.Tensor:
    return sys.environment.F.detach().to(_ANALYSIS_DEVICE, _ANALYSIS_DTYPE)


def system_H(sys: LTISystem) -> torch.Tensor:
    return sys.environment.H.detach().to(_ANALYSIS_DEVICE, _ANALYSIS_DTYPE)


def generate_observations(sys: LTISystem, n_traces: int, length: int) -> torch.Tensor:
    """Return observation trace ``y`` of shape ``[N, L, O_D]`` (cpu/float64).

    ``SystemGroup.generate_dataset`` calls ``ecliseutils.empty_cache`` (a full
    ``gc.collect()`` + ``torch.cuda.empty_cache()``) on *every* timestep. Since the
    rollout's ``history`` list grows each step, the per-step ``gc.collect`` scans an
    ever-larger object graph -> ~O(L^2) and ~40s for L=1000. That memory hygiene is
    meant for large GPU training runs; it is pointless for this tiny CPU-only
    analysis, so we neutralize it just for the rollout (>100x faster).
    """
    orig_empty_cache = eu.empty_cache
    eu.empty_cache = lambda: None
    try:
        ds = sys.generate_dataset(n_traces, length)
    finally:
        eu.empty_cache = orig_empty_cache
    y = ds["environment", "observation"].detach()
    return y.to(_ANALYSIS_DEVICE, _ANALYSIS_DTYPE)


# SECTION: Online self-distillation passes (single streaming pass through the trace)

_EPS = 1e-12


class _LmsUpdater:
    """(N)LMS ``F_hat`` update for :func:`_distill_pass`.

    The per-step self-distillation loss ``|| sg(x_{t+1}) - F_hat x_t ||^2`` has
    gradient ``-u_t x_t^T`` (``u_t`` the injection / residual), so the update is
    the rank-correlation step

        F_hat <- F_hat + lr * mean_n u_t^n (x_t^n)^T

    optionally normalized by the running state energy (NLMS) for stability.

    Behavioral flags consumed by the shared loop:

    - ``reg_scale = step_size``: the decoupled ``weight_decay``/``reg_identity``
      drifts are scaled by the learning rate (they are plain gradient steps).
    - ``norm_h_anchor = normalize``: the M3 H-anchor step shares the F update's
      energy normalization so the two steps stay on one scale.
    - ``advance_with_updated_F = True``: in M1 the state is advanced with the
      *updated* ``F_hat`` (the autonomous part is re-predicted post-update).
    - ``diag_before_update = False``: diagnostics are recorded after the update
      so M1's correction fraction reflects the updated prediction.
    """

    advance_with_updated_F = True
    diag_before_update = False

    def __init__(self, step_size: float, normalize: bool):
        self.step_size = step_size
        self.normalize = normalize
        self.reg_scale = step_size
        self.norm_h_anchor = normalize

    def init(self, S_Dh: int, N: int, dtype: torch.dtype, device) -> None:
        return None

    def update(self, F_hat: torch.Tensor, x: torch.Tensor, pred: torch.Tensor,
               u: torch.Tensor, state):
        N = x.shape[0]
        S_Dh = F_hat.shape[0]
        grad = (u.mT @ x) / N                            # mean_n u_n x_n^T  [S_Dh, S_Dh]
        if self.normalize:
            denom = (x.pow(2).sum(dim=-1).mean() / S_Dh) + _EPS
            grad = grad / denom
        # Diverged state (e.g. an unstable -alpha*I run) makes the gradient blow
        # up; signal the loop to stop updating so F_hat stays finite and the
        # (diverged but finite) spectrum can still be reported/plotted.
        if not torch.isfinite(grad).all():
            return F_hat, state, False
        return F_hat + self.step_size * grad, state, True


class _RlsUpdater:
    """Recursive Least Squares (Newton-style) ``F_hat`` update for :func:`_distill_pass`.

    Same streaming self-distillation regression as :class:`_LmsUpdater` (predict
    ``x_{t+1}`` from ``x_t``), but solved recursively with the inverse state
    covariance ``P`` as preconditioner, so each step is the exact least-squares
    update rather than a raw gradient. All ``F_hat`` rows share the regressor,
    hence a single ``P`` (d x d). The forgetting factor ``lam`` tracks the
    (slowly moving, as ``F_hat`` co-evolves) fixed point; ``P_0 = I / ridge``.
    The prediction error driving the update is ``E = x_next - F_hat x_t = u_t``.

    Behavioral flags consumed by the shared loop:

    - ``reg_scale = 1.0``: RLS has no learning rate, so the decoupled
      ``weight_decay``/``reg_identity`` drifts are applied directly (unscaled),
      persisting across the recursion.
    - ``norm_h_anchor = True``: the M3 H-anchor step is a plain gradient step, so
      it is energy-normalized to stay stable as the state inflates.
    - ``advance_with_updated_F = False``: the state is advanced with the same
      prediction ``F_hat x_t`` used to form the injection (current ``F_hat``).
    - ``diag_before_update = True``: diagnostics are recorded before the update.
    """

    advance_with_updated_F = False
    diag_before_update = True
    reg_scale = 1.0
    norm_h_anchor = True

    def __init__(self, forgetting: float, ridge: float):
        self.forgetting = forgetting
        self.ridge = ridge

    def init(self, S_Dh: int, N: int, dtype: torch.dtype, device):
        I = torch.eye(S_Dh, dtype=dtype, device=device)
        I_N = torch.eye(N, dtype=dtype, device=device)
        P = I / self.ridge
        return (P, I_N)

    def update(self, F_hat: torch.Tensor, x: torch.Tensor, pred: torch.Tensor,
               u: torch.Tensor, state):
        P, I_N = state
        M = x @ P                                        # Phi P            [N, S_Dh]
        S = self.forgetting * I_N + M @ x.mT             # lam I + Phi P Phi^T  [N, N]
        Kg = torch.linalg.solve(S, M).mT                 # P Phi^T S^{-1}   [S_Dh, N]
        E = (pred + u) - pred                            # prediction error = u  [N, S_Dh]
        F_new = F_hat + E.mT @ Kg.mT                     # RLS rank update
        P_new = (P - Kg @ M) / self.forgetting           # covariance recursion
        if not torch.isfinite(F_new).all() or not torch.isfinite(P_new).all():
            return F_hat, (P, I_N), False
        return F_new, (P_new, I_N), True


def _diag_row(pred: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """One ``[correction fraction, injection RMS]`` diagnostic row.

    ``correction fraction`` ||u||/(||F_hat x|| + ||u||) in [0, 1] is the bounded
    share of the next-state norm contributed by the injection: ~1 at spin-up
    (x~0), -> 0 as the state inflates and drowns the injection (the M1 collapse
    story). ``injection RMS`` is ``sqrt(mean_n ||u_n||^2)``.
    """
    frac = u.norm() / (pred.norm() + u.norm() + _EPS)
    innov_rms = u.pow(2).sum(dim=-1).mean().sqrt()
    return torch.stack([frac, innov_rms])


def _distill_pass(
        F_hat: torch.Tensor,
        K_hat: torch.Tensor,
        y: torch.Tensor,
        burn_in: int,
        updater: "_LmsUpdater | _RlsUpdater",
        weight_decay: float = 0.0,
        reg_identity: float = 0.0,
        H_hat: "torch.Tensor | None" = None,
        h_anchor_lr: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One online (test-time-training) self-distillation pass through the trace.

    Streams t = 0..L-1 over all traces in parallel, advancing the state with the
    filter recurrence and updating the shared ``F_hat`` at each step via
    ``updater`` (the only piece that differs between (N)LMS and RLS). State is
    reset to 0 at the start of the pass.

    The injection ``u_t`` selects the method (see module docstring):

    - ``H_hat is None`` (M1, raw injection): ``u_t = K_hat y_t`` -- the residual
      is the F-independent observation, which never vanishes.
    - ``H_hat`` given (M2, error correction): ``u_t = K_hat (y_t - H_hat F_hat x_t)``,
      i.e. the gain applied to the innovation ``e_t = y_t - H_hat F_hat x_t`` where
      ``H_hat F_hat x_t`` decodes the *predicted* next state (Kalman-style); the
      closed loop is then ``x_{t+1} = (I - K_hat H_hat) F_hat x_t + K_hat y_t``.
      With a fixed random ``H_hat`` the self-distillation fixed point decorrelates
      the residual from the state but does not null ``e_t``.

    When ``h_anchor_lr`` > 0 (M3 bridge, only with ``H_hat``) a light step on
    ``H_hat`` from the observation anchor ``0.5||y_t - H_hat F_hat x_t||^2``
    (``grad_H = -e_obs (F_hat x_t)^T``) is taken each step -- the minimal dose of
    observation-space signal that pins the latent->observation magnitude
    (anti-|lambda|-drift) while the self-distillation step still drives ``F_hat``.

    Two optional decoupled regularizers (scaled by ``updater.reg_scale``) shape
    the converged spectrum:

    - ``weight_decay`` (alpha): the multiplicative radial map
      ``lambda -> (1 - reg_scale*alpha) lambda``. Phase-preserving radial shrink.
    - ``reg_identity`` (alpha): the additive translation
      ``lambda -> lambda - reg_scale*alpha``. NOT weight decay: biases complex
      eigen-phases toward +-pi and can increase the magnitude of negative-real
      modes.

    Returns ``(F_hat, eig_history, diag_history)`` where ``eig_history`` is
    ``[L+1, S_Dh]`` (init spectrum followed by one spectrum per streamed timestep)
    and ``diag_history`` is ``[T, 2]`` with columns ``[correction fraction
    ||u||/(||F_hat x||+||u||) in [0,1], injection RMS sqrt(mean_n ||u_n||^2)]``.
    """
    N, L, _ = y.shape
    S_Dh = F_hat.shape[0]
    dtype, device = y.dtype, y.device
    x = torch.zeros((N, S_Dh), dtype=dtype, device=device)
    Kt = K_hat.mT
    I = torch.eye(S_Dh, dtype=dtype, device=device)
    eig_history = [torch.linalg.eigvals(F_hat)]
    diag_history = []
    state = updater.init(S_Dh, N, dtype, device)
    for t in range(L):
        pred = x @ F_hat.mT                              # autonomous part F_hat x_t  [N, S_Dh]
        if H_hat is None:
            u = y[:, t] @ Kt                            # raw injection K_hat y_t  [N, S_Dh]
            e_obs = None
        else:
            e_obs = y[:, t] - pred @ H_hat.mT           # obs innovation y_t - H_hat F_hat x_t  [N, O_D]
            u = e_obs @ Kt                              # K_hat (y_t - H_hat F_hat x_t)  [N, S_Dh]
        if updater.diag_before_update:
            diag_history.append(_diag_row(pred, u))
        if t >= burn_in:
            F_hat, state, ok = updater.update(F_hat, x, pred, u, state)
            if not ok:
                break
            if weight_decay != 0.0:
                F_hat = F_hat - updater.reg_scale * weight_decay * F_hat
            if reg_identity != 0.0:
                F_hat = F_hat - updater.reg_scale * reg_identity * I
            if h_anchor_lr != 0.0 and e_obs is not None:
                gH = (e_obs.mT @ pred) / N               # [O_D, S_Dh]
                if updater.norm_h_anchor:
                    gH = gH / ((pred.pow(2).sum(dim=-1).mean() / S_Dh) + _EPS)
                H_hat = H_hat + h_anchor_lr * gH
            # M1 advances with the *updated* F_hat (re-predict the autonomous
            # part); M2/RLS advance with the prediction used to form the injection.
            if H_hat is None and updater.advance_with_updated_F:
                pred = x @ F_hat.mT
        if not updater.diag_before_update:
            diag_history.append(_diag_row(pred, u))
        x = pred + u                                     # advance
        if not torch.isfinite(x).all():
            break
        eig_history.append(torch.linalg.eigvals(F_hat))
    diag = torch.stack(diag_history, dim=0) if diag_history else torch.zeros((0, 2), dtype=dtype, device=device)
    return F_hat, torch.stack(eig_history, dim=0), diag


class DistillResult(NamedTuple):
    F_hat: torch.Tensor                 # [S_Dh x S_Dh] converged filter transition
    eig_history: torch.Tensor           # [L+1 x S_Dh] complex eigenvalues per timestep
    diag_history: "torch.Tensor | None" = None  # [T x 2] [correction fraction, injection RMS]


def self_distill(
        y: torch.Tensor,
        S_Dh: int,
        O_D: int,
        args: argparse.Namespace,
        generator: torch.Generator,
        error_correction: "bool | None" = None,
        h_anchor_lr: "float | None" = None,
) -> DistillResult:
    """Single streaming self-distillation pass of F_hat over observations ``y``.

    When ``error_correction`` is True (M2) a fixed random decoder ``H_hat`` is
    built and the injection becomes the Kalman-style innovation
    ``K_hat (y_t - H_hat F_hat x_t)`` (the decoder reads the predicted next state);
    when False (M1) the raw injection ``K_hat y_t`` is used. Defaults to
    ``args.error_correction`` so the paired driver can override it per regime.

    When ``h_anchor_lr`` > 0 (only meaningful with ``error_correction``), ``H_hat``
    is *lightly trained* by the observation anchor ``0.5||y_t - H_hat F_hat x_t||^2``
    each step -- this is canvas method M3 (the bridge): the self-distillation step
    still drives ``F_hat`` while the minimal observation signal pins the
    latent->observation magnitude and curbs the ``|lambda|`` drift. Defaults to
    ``args.h_anchor_lr``.
    """
    dtype, device = y.dtype, y.device
    if error_correction is None:
        error_correction = args.error_correction
    if h_anchor_lr is None:
        h_anchor_lr = args.h_anchor_lr

    K_hat = torch.randn((S_Dh, O_D), generator=generator, dtype=dtype, device=device) / (O_D ** 0.5)
    F_hat = torch.randn((S_Dh, S_Dh), generator=generator, dtype=dtype, device=device)
    F_hat *= args.init_radius / _spectral_radius(F_hat)

    # M2: fixed random decoder mapping latent state -> observation space, used
    # only to form the innovation e_t = y_t - H_hat x_t. Never trained here.
    H_hat = None
    if error_correction:
        H_hat = torch.randn((O_D, S_Dh), generator=generator, dtype=dtype, device=device)
        H_hat *= args.h_init_scale / (S_Dh ** 0.5)

    if args.mode == "rls":
        updater = _RlsUpdater(args.forgetting, args.rls_ridge)
    else:
        updater = _LmsUpdater(args.step_size, args.normalize)
    F_hat, eig_history, diag_history = _distill_pass(
        F_hat, K_hat, y, args.burn_in, updater,
        weight_decay=args.weight_decay, reg_identity=args.reg_identity, H_hat=H_hat,
        h_anchor_lr=h_anchor_lr,
    )

    return DistillResult(F_hat=F_hat, eig_history=eig_history, diag_history=diag_history)


# SECTION: Metrics

def modal_observability(F_true: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Per-mode observability ||H v_i|| / ||v_i|| for eigenvectors v_i of F."""
    _, V = torch.linalg.eig(F_true)
    Hc = H.to(V.dtype)
    return (Hc @ V).norm(dim=0) / V.norm(dim=0)


class RegimeSummary(NamedTuple):
    label: str
    O_D: int
    F_true_eig: torch.Tensor
    result: DistillResult
    threshold: float
    mean_phase_err: float
    max_phase_err: float
    n_active: int
    min_observability: float
    final_radius_min: float
    final_radius_max: float
    final_rel_correction: float
    final_innovation_rms: float


def summarize_regime(label: str, F_true: torch.Tensor, H: torch.Tensor,
                     result: DistillResult, threshold: float) -> RegimeSummary:
    F_true_eig = torch.linalg.eigvals(F_true)
    final_eig = result.eig_history[-1]
    mean_err, max_err, n_active = active_phase_match_error(F_true_eig, final_eig, threshold)
    obs = modal_observability(F_true, H)
    final_abs = final_eig.abs()
    diag = result.diag_history
    if diag is not None and diag.shape[0] > 0:
        final_rel_correction = diag[-1, 0].item()
        final_innovation_rms = diag[-1, 1].item()
    else:
        final_rel_correction = float("nan")
        final_innovation_rms = float("nan")
    return RegimeSummary(
        label=label,
        O_D=H.shape[0],
        F_true_eig=F_true_eig,
        result=result,
        threshold=threshold,
        mean_phase_err=mean_err,
        max_phase_err=max_err,
        n_active=n_active,
        min_observability=obs.abs().min().item(),
        final_radius_min=final_abs.min().item(),
        final_radius_max=final_abs.max().item(),
        final_rel_correction=final_rel_correction,
        final_innovation_rms=final_innovation_rms,
    )


def print_summary(s: RegimeSummary) -> None:
    F_true_eig = s.F_true_eig
    final_eig = s.result.eig_history[-1]
    print(f"\n=== {s.label} (S_D={F_true_eig.numel()}, O_D={s.O_D}) ===")

    def fmt(eig: torch.Tensor) -> str:
        order = torch.argsort(torch.angle(eig))
        eig = eig[order]
        mags = ", ".join(f"{v:.4f}" for v in eig.abs().tolist())
        angs = ", ".join(f"{v:+.4f}" for v in torch.angle(eig).tolist())
        return f"|eig| = [{mags}]\n        angle = [{angs}]"

    print(f"  true F: {fmt(F_true_eig)}")
    print(f"  F_hat : {fmt(final_eig)}")
    print(f"  final |eig(F_hat)| in [{s.final_radius_min:.4f}, {s.final_radius_max:.4f}]"
          f"  (-> 1.0 means pushed to unit circle)")
    print(f"  active modes (|lambda|>={s.threshold}): {s.n_active}/{F_true_eig.numel()}")
    print(f"  active-mode phase error vs nearest true freq (rad): "
          f"mean={s.mean_phase_err:.4f}, max={s.max_phase_err:.4f}")
    print(f"  min modal observability ||H v_i||/||v_i||: {s.min_observability:.4f}"
          f"  (well above 0 => no obscured mode)")
    if not (s.final_rel_correction != s.final_rel_correction):  # not NaN
        print(f"  final correction fraction ||u||/(||F x||+||u||): {s.final_rel_correction:.4f}"
              f"  (-> 0 means the state inflated and drowned the injection)")
        print(f"  final injection RMS sqrt(mean ||u_n||^2): {s.final_innovation_rms:.4f}")


# SECTION: Plotting  (``_draw_complex_plane`` now imported from kf_rnn.analysis)

def _draw_traces(ax_ang, ax_mag, eig_history: torch.Tensor, F_true_eig: torch.Tensor,
                 title: str, phase_plot_threshold: float = 0.1) -> None:
    steps = np.arange(eig_history.shape[0])
    angles = torch.angle(eig_history).detach().numpy()    # [L+1, S_Dh]
    mags = eig_history.abs().detach().numpy()
    # The phase of a near-zero (collapsed) eigenvalue is numerically meaningless
    # and flips around; hide it so the active modes' phase locking is legible.
    angles_plot = np.where(mags >= phase_plot_threshold, angles, np.nan)
    for j in range(angles.shape[1]):
        ax_ang.plot(steps, angles_plot[:, j], lw=1.0)
        ax_mag.plot(steps, mags[:, j], lw=1.0)
    for a in torch.angle(F_true_eig).tolist():
        ax_ang.axhline(a, color="0.6", ls="--", lw=0.8)
    ax_mag.axhline(1.0, color="0.6", ls="--", lw=0.8)
    ax_ang.set_title(f"{title}: eigen-phase vs timestep")
    ax_ang.set_xlabel("timestep")
    ax_ang.set_ylabel("angle (rad)")
    ax_mag.set_title(f"{title}: eigen-magnitude vs timestep")
    ax_mag.set_xlabel("timestep")
    ax_mag.set_ylabel(r"$|\lambda|$")


def _draw_diag(ax_frac, ax_innov, diag_history: torch.Tensor, title: str) -> None:
    """Innovation/correction diagnostics vs timestep.

    ``correction fraction`` ||u||/(||F_hat x|| + ||u||) in [0, 1] is the share of
    the next-state norm contributed by the injection: ~1 at spin-up, then -> 0 as
    the state inflates and drowns the injection (the M1 collapse story). A nonzero
    steady floor means the correction keeps mattering relative to autonomous
    prediction. ``injection RMS`` (log scale) is the raw magnitude of the
    injection/innovation u_t; under M2 with a fixed random H_hat the innovation
    decorrelates from the state but is not nulled (it does not -> 0).
    """
    steps = np.arange(diag_history.shape[0])
    frac = diag_history[:, 0].detach().numpy()
    innov = diag_history[:, 1].detach().numpy()
    ax_frac.plot(steps, frac, lw=1.0, color="C3")
    ax_frac.set_ylim(-0.02, 1.02)
    ax_frac.set_title(f"{title}: correction fraction")
    ax_frac.set_xlabel("timestep")
    ax_frac.set_ylabel(r"$\|u\| / (\|\hat{F} x\| + \|u\|)$")
    ax_innov.plot(steps, innov, lw=1.0, color="C0")
    ax_innov.set_yscale("log")
    ax_innov.set_title(f"{title}: injection RMS")
    ax_innov.set_xlabel("timestep")
    ax_innov.set_ylabel(r"$\sqrt{\mathrm{mean}_n\,\|u_n\|^2}$")


def plot_regimes(summaries: list[RegimeSummary], out_dir: str, tag: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.0), squeeze=False)
    for ax, s in zip(axes[0], summaries):
        _draw_complex_plane(ax, s.F_true_eig, s.result.eig_history[-1], s.label)
    fig.suptitle("Eigenvalues: true F (x) vs converged F_hat (o)")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_eig_complex_plane.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 8.0), squeeze=False)
    for col, s in enumerate(summaries):
        _draw_traces(axes[0, col], axes[1, col], s.result.eig_history, s.F_true_eig, s.label)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_traces.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    if any(s.result.diag_history is not None and s.result.diag_history.shape[0] > 0
           for s in summaries):
        fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 8.0), squeeze=False)
        for col, s in enumerate(summaries):
            diag = s.result.diag_history
            if diag is None or diag.shape[0] == 0:
                continue
            _draw_diag(axes[0, col], axes[1, col], diag, s.label)
        fig.tight_layout()
        p = os.path.join(out_dir, f"{tag}_innovation.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    return paths


def plot_seed_sweep(mean_errs: list[float], min_obs: list[float], out_dir: str,
                    tag: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].hist(mean_errs, bins=min(20, max(5, len(mean_errs) // 2)), color="C3", alpha=0.8)
    axes[0].set_title("mean phase-match error over seeds")
    axes[0].set_xlabel("mean |angle error| (rad)")
    axes[0].set_ylabel("count")
    axes[1].hist(min_obs, bins=min(20, max(5, len(min_obs) // 2)), color="C0", alpha=0.8)
    axes[1].set_title("min modal observability over seeds")
    axes[1].set_xlabel(r"$\min_i \|H v_i\| / \|v_i\|$")
    axes[1].set_ylabel("count")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_seed_sweep.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# SECTION: Driver

def run_single_regime(sys: LTISystem, label: str, args: argparse.Namespace,
                      seed: int, error_correction: "bool | None" = None,
                      h_anchor_lr: "float | None" = None) -> RegimeSummary:
    F_true = system_F(sys)
    H = system_H(sys)
    O_D = H.shape[0]
    y = generate_observations(sys, args.n_traces, args.length)

    gen = torch.Generator(device=_ANALYSIS_DEVICE).manual_seed(seed)
    S_Dh = args.fhat_dim if args.fhat_dim is not None else F_true.shape[0]
    result = self_distill(y, S_Dh, O_D, args, gen, error_correction=error_correction,
                          h_anchor_lr=h_anchor_lr)
    return summarize_regime(label, F_true, H, result, args.active_threshold)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_sys = p.add_argument_group("system")
    g_sys.add_argument("--s-d", type=int, default=4, help="state dimension S_D")
    g_sys.add_argument("--o-d", type=int, default=2, help="observation dimension (single-regime mode)")
    g_sys.add_argument("--w-std", type=float, default=0.5, help="process-noise std")
    g_sys.add_argument("--v-std", type=float, default=0.5, help="observation-noise std")

    g_cmp = p.add_argument_group("paired comparison")
    g_cmp.add_argument("--compare-observation", action="store_true",
                       help="run fully-observed vs partial-observed on the SAME F")
    g_cmp.add_argument("--compare-feedback", action="store_true",
                       help="run raw injection (M1) vs innovation feedback (M2) on the SAME F")
    g_cmp.add_argument("--compare-bridge", action="store_true",
                       help="run raw injection (M1) vs error correction (M2) vs the M3 bridge "
                            "(M2 + a lightly-trained H_hat observation anchor) on the SAME F")
    g_cmp.add_argument("--o-d-full", type=int, default=None,
                       help="observation dim of the fully-observed run (default: S_D)")
    g_cmp.add_argument("--o-d-partial", type=int, default=2,
                       help="observation dim of the partial run")

    g_data = p.add_argument_group("data")
    g_data.add_argument("-N", "--n-traces", type=int, default=16, help="number of traces")
    g_data.add_argument("-L", "--length", type=int, default=10000, help="trace length")
    g_data.add_argument("--burn-in", type=int, default=50,
                        help="initial states discarded before forming OLS pairs")

    g_filt = p.add_argument_group("filter / training")
    g_filt.add_argument("--fhat-dim", type=int, default=None,
                        help="dimension of F_hat (default: S_D)")
    g_filt.add_argument("--init-radius", type=float, default=0.5,
                        help="spectral radius of the random F_hat initialization")
    g_filt.add_argument("--mode", type=str, default="rls", choices=("rls", "lms"),
                        help="online update rule: recursive least squares (default) or (N)LMS")
    g_filt.add_argument("--forgetting", type=float, default=0.99,
                        help="rls mode: forgetting factor lambda in (0, 1] (1.0 = no forgetting)")
    g_filt.add_argument("--rls-ridge", type=float, default=1e-2,
                        help="rls mode: ridge delta for the inverse-covariance init P_0 = I/delta")
    g_filt.add_argument("--step-size", type=float, default=0.5,
                        help="lms mode: per-step (N)LMS learning rate")
    g_filt.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="lms mode: normalize the per-step update by state energy (NLMS)")
    g_filt.add_argument("--weight-decay", type=float, default=0.0,
                        help="weight decay alpha (lambda->(1-alpha)lambda, phase-preserving radial "
                             "shrink). Applied per step; scaled by step-size in lms mode, directly in rls.")
    g_filt.add_argument("--reg-identity", type=float, default=0.0,
                        help="-alpha*I trace penalty (lambda->lambda-alpha, phase-biasing toward +-pi; "
                             "NOT weight decay). Applied per step; scaled by step-size in lms mode, directly in rls.")
    g_filt.add_argument("--active-threshold", type=float, default=0.5,
                        help="|lambda| above which an F_hat mode counts as active (near-circle)")
    g_filt.add_argument("--error-correction", action=argparse.BooleanOptionalAction, default=False,
                        help="M2: inject the Kalman-style innovation K_hat(y - H_hat F_hat x) using "
                             "a fixed random decoder H_hat instead of the raw injection K_hat y (M1, default).")
    g_filt.add_argument("--h-init-scale", type=float, default=1.0,
                        help="M2: scale of the fixed random decoder H_hat (rows normalized by "
                             "1/sqrt(S_Dh), mirroring K_hat).")
    g_filt.add_argument("--h-anchor-lr", type=float, default=0.0,
                        help="M3: learning rate for the light observation-anchor step on H_hat "
                             "(0.5||y - H_hat F_hat x||^2). >0 turns the M2 fixed-H run into the "
                             "M3 bridge (requires --error-correction or --compare-bridge).")

    g_run = p.add_argument_group("run")
    g_run.add_argument("--seed", type=int, default=0, help="base random seed")
    g_run.add_argument("--n-seeds", type=int, default=1,
                       help="repeat the (partial) regime over N seeds and report the distribution")
    g_run.add_argument("--out-name", type=str, default="self_distillation_eigenphase",
                       help="output subdirectory under OUTPUT_PATH")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = os.path.join(OUTPUT_PATH, args.out_name)

    # SECTION: Multi-seed sweep of the partial regime (degeneracy is rare).
    if args.n_seeds > 1:
        O_D = args.o_d_partial if args.compare_observation else args.o_d
        print(f"Seed sweep: partial regime S_D={args.s_d}, O_D={O_D}, "
              f"{args.n_seeds} seeds, mode={args.mode}")
        mean_errs, max_errs, min_obs = [], [], []
        for i in range(args.n_seeds):
            seed = args.seed + i
            torch.manual_seed(seed)
            sys = build_system(args.s_d, O_D, args.w_std, args.v_std)
            s = run_single_regime(sys, f"seed {seed}", args, seed)
            mean_errs.append(s.mean_phase_err)
            max_errs.append(s.max_phase_err)
            min_obs.append(s.min_observability)
        me = torch.tensor(mean_errs)
        me = me[~torch.isnan(me)]
        mo = torch.tensor(min_obs)
        print("\n=== seed-sweep summary (partial regime) ===")
        print(f"  mean phase error (rad): mean={me.mean():.4f}, median={me.median():.4f}, "
              f"q90={me.quantile(0.9):.4f}, max={me.max():.4f}")
        print(f"  min modal observability: min={mo.min():.4f}, median={mo.median():.4f}")
        path = plot_seed_sweep(mean_errs, min_obs, out_dir, f"seedsweep_sd{args.s_d}_od{O_D}")
        print(f"\nsaved: {path}")
        return

    # SECTION: Single run or paired (shared-F) comparison.
    if args.compare_observation:
        o_d_full = args.o_d_full if args.o_d_full is not None else args.s_d
        torch.manual_seed(args.seed)
        full_sys, partial_sys = build_shared_systems(
            args.s_d, o_d_full, args.o_d_partial, args.w_std, args.v_std)
        summaries = [
            run_single_regime(full_sys, f"fully observed (O_D={o_d_full})", args, args.seed),
            run_single_regime(partial_sys, f"partial (O_D={args.o_d_partial})", args, args.seed),
        ]
        tag = f"compare_sd{args.s_d}_full{o_d_full}_part{args.o_d_partial}"
    elif args.compare_feedback:
        # Same F (same seed -> identical K_hat/F_hat init): M1 raw injection vs
        # M2 innovation feedback, to isolate the effect of error correction.
        torch.manual_seed(args.seed)
        sys = build_system(args.s_d, args.o_d, args.w_std, args.v_std)
        summaries = [
            run_single_regime(sys, f"raw injection M1 (O_D={args.o_d})", args, args.seed,
                              error_correction=False),
            run_single_regime(sys, f"error correction M2 (O_D={args.o_d})", args, args.seed,
                              error_correction=True),
        ]
        tag = f"feedback_sd{args.s_d}_od{args.o_d}"
    elif args.compare_bridge:
        # Same F: M1 raw injection vs M2 error correction vs M3 bridge (M2 + a
        # lightly-trained H_hat observation anchor). Isolates what the minimal
        # observation signal buys on top of pure latent self-distillation.
        torch.manual_seed(args.seed)
        sys = build_system(args.s_d, args.o_d, args.w_std, args.v_std)
        anchor_lr = args.h_anchor_lr if args.h_anchor_lr > 0.0 else 0.05
        summaries = [
            run_single_regime(sys, f"raw injection M1 (O_D={args.o_d})", args, args.seed,
                              error_correction=False, h_anchor_lr=0.0),
            run_single_regime(sys, f"error correction M2 (O_D={args.o_d})", args, args.seed,
                              error_correction=True, h_anchor_lr=0.0),
            run_single_regime(sys, f"bridge M3 (anchor lr={anchor_lr:g})", args, args.seed,
                              error_correction=True, h_anchor_lr=anchor_lr),
        ]
        tag = f"bridge_sd{args.s_d}_od{args.o_d}"
    else:
        torch.manual_seed(args.seed)
        sys = build_system(args.s_d, args.o_d, args.w_std, args.v_std)
        summaries = [run_single_regime(sys, f"single (O_D={args.o_d})", args, args.seed)]
        tag = f"single_sd{args.s_d}_od{args.o_d}"

    for s in summaries:
        print_summary(s)
    paths = plot_regimes(summaries, out_dir, tag)
    print("\nsaved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
