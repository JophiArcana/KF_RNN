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

def _spectral_radius(M: torch.Tensor) -> float:
    return torch.linalg.eigvals(M).abs().max().item()


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

def lms_pass(
        F_hat: torch.Tensor,
        K_hat: torch.Tensor,
        y: torch.Tensor,
        step_size: float,
        burn_in: int,
        normalize: bool,
        weight_decay: float = 0.0,
        reg_identity: float = 0.0,
        H_hat: "torch.Tensor | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One online (test-time-training) (N)LMS pass through the trace.

    Streams t = 0..L-1 over all traces in parallel. At each step the per-step
    self-distillation loss ``|| sg(x_{t+1}) - F_hat x_t ||^2`` has gradient
    ``-u_t x_t^T`` where ``u_t`` is the injection / residual, so the update is the
    rank-correlation step

        F_hat <- F_hat + lr * mean_n u_t^n (x_t^n)^T

    optionally normalized by the running state energy (NLMS) for stability. State
    is reset to 0 at the start of the pass.

    The injection ``u_t`` selects the method (see module docstring):

    - ``H_hat is None`` (M1, raw injection): ``u_t = K_hat y_t`` -- the residual
      is the F-independent observation, which never vanishes. The *updated*
      ``F_hat`` is then used to advance the state.
    - ``H_hat`` given (M2, error correction): ``u_t = K_hat (y_t - H_hat F_hat x_t)``,
      i.e. the gain applied to the innovation ``e_t = y_t - H_hat F_hat x_t`` where
      ``H_hat F_hat x_t`` decodes the *predicted* next state (Kalman-style); the
      closed loop is then ``x_{t+1} = (I - K_hat H_hat) F_hat x_t + K_hat y_t``.
      The state is advanced with the same prediction ``F_hat x_t`` used to form the
      innovation (current ``F_hat``, mirroring RLS). With a fixed random ``H_hat``
      the self-distillation fixed point decorrelates the residual from the state
      but does not null ``e_t``.

    Two optional regularizers (decoupled, applied per update step scaled by the
    learning rate) shape the converged spectrum:

    - ``weight_decay`` (alpha): penalizes ``0.5 alpha ||F_hat||_F^2`` -> step
      ``-lr*alpha*F_hat``, the multiplicative radial map ``lambda -> (1-lr*alpha)lambda``.
      Phase-preserving radial shrink.
    - ``reg_identity`` (alpha): penalizes ``alpha*trace(F_hat)`` -> step
      ``-lr*alpha*I``, the additive translation ``lambda -> lambda - lr*alpha``.
      NOT weight decay: biases complex eigen-phases toward +-pi and can increase
      the magnitude of negative-real modes.

    Returns ``(F_hat, eig_history, diag_history)`` where ``eig_history`` is
    ``[L+1, S_Dh]`` (init spectrum followed by one spectrum per streamed timestep)
    and ``diag_history`` is ``[T, 2]`` with columns ``[correction fraction
    ||u||/(||F_hat x||+||u||) in [0,1], injection RMS sqrt(mean_n ||u_n||^2)]``
    per streamed step.
    """
    N, L, _ = y.shape
    S_Dh = F_hat.shape[0]
    x = torch.zeros((N, S_Dh), dtype=y.dtype, device=y.device)
    Kt = K_hat.mT
    I = torch.eye(S_Dh, dtype=y.dtype, device=y.device)
    eps = 1e-12
    eig_history = [torch.linalg.eigvals(F_hat)]
    diag_history = []
    for t in range(L):
        if H_hat is None:
            u = y[:, t] @ Kt                            # raw injection K_hat y_t  [N, S_Dh]
        else:
            pred = x @ F_hat.mT                          # predicted next state F_hat x_t  [N, S_Dh]
            u = (y[:, t] - pred @ H_hat.mT) @ Kt        # K_hat (y_t - H_hat F_hat x_t)  [N, S_Dh]
        if t >= burn_in:
            grad = (u.mT @ x) / N                        # mean_n u_n x_n^T  [S_Dh, S_Dh]
            if normalize:
                denom = (x.pow(2).sum(dim=-1).mean() / S_Dh) + eps
                grad = grad / denom
            # Diverged state (e.g. an unstable -alpha*I run) makes the gradient
            # blow up; stop updating so F_hat stays finite and the (diverged but
            # finite) spectrum can still be reported/plotted.
            if not torch.isfinite(grad).all():
                break
            F_hat = F_hat + step_size * grad
            if weight_decay != 0.0:
                F_hat = F_hat - step_size * weight_decay * F_hat
            if reg_identity != 0.0:
                F_hat = F_hat - step_size * reg_identity * I
        # M1 advances with the *updated* F_hat; M2 advances with the same
        # prediction F_hat x_t used to form the innovation (current F_hat).
        if H_hat is None:
            pred = x @ F_hat.mT                          # autonomous part F_hat x_t  [N, S_Dh]
        # Bounded correction fraction in [0, 1]: the share of the next-state norm
        # contributed by the injection. ~1 at spin-up (x~0), -> 0 as the state
        # inflates and drowns the injection (the M1 collapse story).
        frac = u.norm() / (pred.norm() + u.norm() + eps)
        innov_rms = u.pow(2).sum(dim=-1).mean().sqrt()  # sqrt(mean_n ||u_n||^2)
        diag_history.append(torch.stack([frac, innov_rms]))
        x = pred + u                                     # advance
        if not torch.isfinite(x).all():
            break
        eig_history.append(torch.linalg.eigvals(F_hat))
    diag = torch.stack(diag_history, dim=0) if diag_history else torch.zeros((0, 2), dtype=y.dtype, device=y.device)
    return F_hat, torch.stack(eig_history, dim=0), diag


def rls_pass(
        F_hat: torch.Tensor,
        K_hat: torch.Tensor,
        y: torch.Tensor,
        burn_in: int,
        forgetting: float,
        ridge: float,
        weight_decay: float = 0.0,
        reg_identity: float = 0.0,
        H_hat: "torch.Tensor | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One online Recursive Least Squares (RLS) pass through the trace.

    Same streaming self-distillation regression as :func:`lms_pass` (predict
    ``x_{t+1}`` from ``x_t``), but solved recursively with a Newton-style
    preconditioner: the gain is the inverse state covariance ``P`` so each step
    is the exact least-squares update rather than a raw gradient. All ``F_hat``
    rows share the regressor, hence a single ``P`` (d x d). With a forgetting
    factor ``lam`` it tracks the (slowly moving, as ``F_hat`` co-evolves) fixed
    point of the detached self-distillation regression; ``P_0 = I / ridge``.

    The injection ``u_t`` selects the method exactly as in :func:`lms_pass`:
    ``H_hat is None`` gives the raw injection ``K_hat y_t`` (M1); a given
    ``H_hat`` gives the Kalman-style error-correcting innovation
    ``K_hat (y_t - H_hat F_hat x_t)`` (M2), where ``H_hat F_hat x_t`` decodes the
    predicted next state. The prediction error driving the RLS update is
    ``E = x_next - F_hat x_t = u_t`` either way.

    The two optional regularizers are applied as decoupled per-step drifts so
    they *persist* across the recursion (RLS has no learning rate, so unlike
    :func:`lms_pass` they are applied directly, not scaled): ``weight_decay``
    gives ``lambda -> (1-weight_decay)lambda`` and ``reg_identity`` gives
    ``lambda -> lambda - reg_identity``.

    Returns ``(F_hat, eig_history, diag_history)`` with ``eig_history`` of shape
    ``[L+1, S_Dh]`` and ``diag_history`` of shape ``[T, 2]`` (columns ``[correction
    fraction ||u||/(||F_hat x||+||u||), injection RMS sqrt(mean_n ||u_n||^2)]``).
    """
    N, L, _ = y.shape
    S_Dh = F_hat.shape[0]
    x = torch.zeros((N, S_Dh), dtype=y.dtype, device=y.device)
    Kt = K_hat.mT
    I = torch.eye(S_Dh, dtype=y.dtype, device=y.device)
    I_N = torch.eye(N, dtype=y.dtype, device=y.device)
    eps = 1e-12
    P = I / ridge
    eig_history = [torch.linalg.eigvals(F_hat)]
    diag_history = []
    for t in range(L):
        pred = x @ F_hat.mT                              # predicted next state F_hat x_t  [N, S_Dh]
        if H_hat is None:
            u = y[:, t] @ Kt                            # raw injection K_hat y_t  [N, S_Dh]
        else:
            u = (y[:, t] - pred @ H_hat.mT) @ Kt        # K_hat (y_t - H_hat F_hat x_t)  [N, S_Dh]
        x_next = pred + u                                # advance with current F_hat
        # Bounded correction fraction in [0, 1] (see lms_pass for rationale).
        frac = u.norm() / (pred.norm() + u.norm() + eps)
        innov_rms = u.pow(2).sum(dim=-1).mean().sqrt()  # sqrt(mean_n ||u_n||^2)
        diag_history.append(torch.stack([frac, innov_rms]))
        if t >= burn_in:
            M = x @ P                                    # Phi P            [N, S_Dh]
            S = forgetting * I_N + M @ x.mT              # lam I + Phi P Phi^T  [N, N]
            Kg = torch.linalg.solve(S, M).mT            # P Phi^T S^{-1}   [S_Dh, N]
            E = x_next - pred                            # prediction error = u  [N, S_Dh]
            F_new = F_hat + E.mT @ Kg.mT                 # RLS rank update
            P = (P - Kg @ M) / forgetting               # covariance recursion
            if not torch.isfinite(F_new).all() or not torch.isfinite(P).all():
                break
            F_hat = F_new
            if weight_decay != 0.0:
                F_hat = F_hat - weight_decay * F_hat
            if reg_identity != 0.0:
                F_hat = F_hat - reg_identity * I
        x = x_next
        if not torch.isfinite(x).all():
            break
        eig_history.append(torch.linalg.eigvals(F_hat))
    diag = torch.stack(diag_history, dim=0) if diag_history else torch.zeros((0, 2), dtype=y.dtype, device=y.device)
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
) -> DistillResult:
    """Single streaming self-distillation pass of F_hat over observations ``y``.

    When ``error_correction`` is True (M2) a fixed random decoder ``H_hat`` is
    built and the injection becomes the Kalman-style innovation
    ``K_hat (y_t - H_hat F_hat x_t)`` (the decoder reads the predicted next state);
    when False (M1) the raw injection ``K_hat y_t`` is used. Defaults to
    ``args.error_correction`` so the paired driver can override it per regime.
    """
    dtype, device = y.dtype, y.device
    if error_correction is None:
        error_correction = args.error_correction

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
        F_hat, eig_history, diag_history = rls_pass(
            F_hat, K_hat, y, args.burn_in, args.forgetting, args.rls_ridge,
            weight_decay=args.weight_decay, reg_identity=args.reg_identity, H_hat=H_hat,
        )
    else:
        F_hat, eig_history, diag_history = lms_pass(
            F_hat, K_hat, y, args.step_size, args.burn_in, args.normalize,
            weight_decay=args.weight_decay, reg_identity=args.reg_identity, H_hat=H_hat,
        )

    return DistillResult(F_hat=F_hat, eig_history=eig_history, diag_history=diag_history)


# SECTION: Metrics

def _circular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d)).abs()


def active_phase_match_error(
        true_eig: torch.Tensor,
        est_eig: torch.Tensor,
        threshold: float,
) -> tuple[float, float, int]:
    """Phase error of the *active* filter modes (those near the unit circle).

    The hypothesis concerns eigenvalues that migrate to the unit circle; modes
    that collapse toward 0 have meaningless phase and are excluded. For each
    active ``F_hat`` eigenvalue (``|lambda| >= threshold``) we take the circular
    distance to the nearest *true* eigen-phase (precision: "is each filter
    resonance a real system frequency?"). Returns ``(mean, max, n_active)`` with
    angles in radians; ``(nan, nan, 0)`` if no mode is active.
    """
    active = est_eig[est_eig.abs() >= threshold]
    if active.numel() == 0:
        return float("nan"), float("nan"), 0
    true_ang = torch.angle(true_eig)
    errs = []
    for a in torch.angle(active):
        errs.append(_circular_distance(a, true_ang).min().item())
    errs_t = torch.tensor(errs)
    return errs_t.mean().item(), errs_t.max().item(), int(active.numel())


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


# SECTION: Plotting

def _draw_complex_plane(ax, F_true_eig: torch.Tensor, F_hat_eig: torch.Tensor, title: str) -> None:
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="0.7", lw=1.0, zorder=0)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    # Radial lines from the origin to each eigenvalue make the phase (angle)
    # difference between true F and converged F_hat visually obvious.
    for re, im in zip(F_true_eig.real.tolist(), F_true_eig.imag.tolist()):
        ax.plot([0.0, re], [0.0, im], color="C0", lw=0.8, alpha=0.5, zorder=1)
    F_hat_re = F_hat_eig.real.detach()
    F_hat_im = F_hat_eig.imag.detach()
    for re, im in zip(F_hat_re.tolist(), F_hat_im.tolist()):
        ax.plot([0.0, re], [0.0, im], color="C3", lw=0.8, alpha=0.5, zorder=1)
    ax.scatter(F_true_eig.real, F_true_eig.imag, marker="x", s=80, color="C0",
               label="true F", zorder=3)
    ax.scatter(F_hat_re, F_hat_im, marker="o", s=80,
               facecolors="none", edgecolors="C3", label=r"converged $\hat{F}$", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


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
                      seed: int, error_correction: "bool | None" = None) -> RegimeSummary:
    F_true = system_F(sys)
    H = system_H(sys)
    O_D = H.shape[0]
    y = generate_observations(sys, args.n_traces, args.length)

    gen = torch.Generator(device=_ANALYSIS_DEVICE).manual_seed(seed)
    S_Dh = args.fhat_dim if args.fhat_dim is not None else F_true.shape[0]
    result = self_distill(y, S_Dh, O_D, args, gen, error_correction=error_correction)
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
