"""Shared analysis helpers for the self-distillation / TTT studies.

Numeric utilities reused across the analysis scripts (impulse response, guarded
closed-form filter error, eigen-phase metrics) and by the ``sd_analytical``
training metric. Kept free of a top-level matplotlib import (the single plotting
helper imports it lazily) so this module is safe to import from the
training/metrics path.
"""
from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.model.sequential.base import SequentialPredictor


# Magnitude above which a matrix is treated as diverged WITHOUT calling an
# eigensolver: a diverging filter's entries grow geometrically and pass through
# huge-but-finite values before becoming inf, and LAPACK (MKL) can corrupt memory
# / hard-abort on such inputs -- which no Python ``try/except`` can catch. A stable
# filter has O(1) entries, so 1e6 is a safe, generous divergence cutoff.
_EIG_MAX_ABS = 1e6


def eig_unsafe(M: torch.Tensor) -> bool:
    """True if ``M`` must not be fed to an eigensolver (non-finite or huge)."""
    return (not torch.isfinite(M).all()) or (M.abs().max().item() > _EIG_MAX_ABS)


def spectral_radius(M: torch.Tensor) -> float:
    """Spectral radius, robust to a diverged (non-finite/huge) matrix (returns inf)."""
    if eig_unsafe(M):
        return float("inf")
    try:
        return torch.linalg.eigvals(M).abs().max().item()
    except RuntimeError:
        return float("inf")


def kalman_ir(F: torch.Tensor, H: torch.Tensor, K: torch.Tensor, R: int) -> torch.Tensor:
    """Observation->observation impulse response, ``[R x O_D x O_D]`` (lag, out, in).

    Same construction as ``CnnAnalyticalPredictor._analytical_initialization``:
    tap ``r`` is the linear map from ``y_{t-1-r}`` to ``y_hat_t`` for the filter
    ``(F, H, K)`` (true system -> ground-truth Kalman IIR; adapted -> learned IR).
    """
    S_D = F.shape[-1]
    powers = eu.pow_series(F @ (torch.eye(S_D, device=F.device) - K @ H), R)     # [R x S_D x S_D]
    return H @ powers @ (F @ K)                                                  # [R x O_D x O_D]


@torch.no_grad()
def filter_analytical_error(F_hat: torch.Tensor, H_hat: torch.Tensor,
                            K_hat: torch.Tensor, lsg_td: TensorDict) -> float:
    """Exact steady-state observation prediction error of the LTI filter
    ``(F_hat, H_hat, K_hat)`` against the true system (``lsg_td = lsg.td()``).

    This is the project's closed-form analytical error -- an expectation over the
    process/observation noise, not a finite-sample MSE. Returns ``+inf`` when the
    filter's closed loop ``F_hat (I - K_hat H_hat)`` is unstable (spectral radius
    ``>= 1``): the closed-form geometric series only converges for a contractive
    closed loop.
    """
    if any(eig_unsafe(M) for M in (F_hat, H_hat, K_hat)):
        return float("inf")
    S = F_hat.shape[-1]
    eye = torch.eye(S, dtype=F_hat.dtype, device=F_hat.device)
    closed = F_hat @ (eye - K_hat @ H_hat)                       # filter-side closed loop
    if eig_unsafe(closed):
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
    of filters; returns an ``[N]`` numpy array (``nan`` for each rejected filter).

    Any non-finite / huge / non-contractive trajectory is replaced by a trivially
    stable dummy (``F = H = K = 0`` -> closed loop ``0``) *before* any eigensolver
    runs, so LAPACK/cuSOLVER never sees a diverged matrix (the memory-corruption
    hazard the scalar guard exists to avoid); those entries are masked to ``nan``.
    """
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
def guarded_analytical_error(kfs_td: TensorDict, sg_td: TensorDict) -> torch.Tensor:
    """Guarded closed-form observation error over arbitrary leading batch dims.

    Generalization of :func:`batched_filter_analytical_error` that operates on a
    ``kfs_td`` (batched ``F``/``H``/``K`` and optional controller ``B``) and system
    td ``sg_td`` (already broadcast-aligned by the caller). Diverged / non-finite /
    non-contractive filters are gated to a trivially stable dummy before any
    eigensolver runs, then masked back to ``nan``. Returns the ``[B...]`` observation
    error tensor (``nan`` where the filter is unstable).
    """
    F, H, K = kfs_td["F"], kfs_td["H"], kfs_td["K"]
    S = F.shape[-1]
    eye = torch.eye(S, dtype=F.dtype, device=F.device)

    def _safe(M: torch.Tensor) -> torch.Tensor:                     # [B...] bool
        return torch.isfinite(M).all(dim=(-2, -1)) & (M.abs().amax(dim=(-2, -1)) <= _EIG_MAX_ABS)

    def _gate(M: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        return torch.where(keep[..., None, None], M, torch.zeros_like(M))

    ok = _safe(F) & _safe(H) & _safe(K)
    closed = _gate(F, ok) @ (eye - _gate(K, ok) @ _gate(H, ok))
    rho = torch.linalg.eigvals(closed).abs().amax(dim=-1)           # [B...]
    stable = ok & torch.isfinite(rho) & (rho < 1.0)

    gated = kfs_td.clone()
    gated["F"], gated["H"], gated["K"] = _gate(F, stable), _gate(H, stable), _gate(K, stable)
    err = SequentialPredictor.analytical_error(gated, sg_td)["environment", "observation"]
    stable = stable.expand_as(err)
    valid = stable & torch.isfinite(err) & (err >= 0.0)
    return torch.where(valid, err, torch.full_like(err, float("nan")))


def _circular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d)).abs()


def active_phase_match_error(
        true_eig: torch.Tensor,
        est_eig: torch.Tensor,
        threshold: float,
) -> tuple[float, float, int]:
    """Phase error of the *active* filter modes (those near the unit circle).

    For each active ``F_hat`` eigenvalue (``|lambda| >= threshold``) take the
    circular distance to the nearest *true* eigen-phase. Returns
    ``(mean, max, n_active)`` in radians; ``(nan, nan, 0)`` if no mode is active.
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


def draw_complex_plane(ax, F_true_eig: torch.Tensor, F_hat_eig: torch.Tensor, title: str) -> None:
    """Scatter true vs estimated eigenvalues on the complex plane with the unit
    circle and radial phase lines (matplotlib imported lazily)."""
    F_true_eig = F_true_eig.detach().cpu()
    F_hat_eig = F_hat_eig.detach().cpu()
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="0.7", lw=1.0, zorder=0)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    for re, im in zip(F_true_eig.real.tolist(), F_true_eig.imag.tolist()):
        ax.plot([0.0, re], [0.0, im], color="C0", lw=0.8, alpha=0.5, zorder=1)
    F_hat_re = F_hat_eig.real
    F_hat_im = F_hat_eig.imag
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
