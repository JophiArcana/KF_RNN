#!/usr/bin/env python
"""Convergence readout of the a-posteriori (beta2) SD sweep (``--methods post``).

Companion to ``plot_post_sweep.py`` (which shows only the *steady* excess). This
plots the full adaptation trajectory -- excess over the irreducible floor vs
online step -- so the convergence-vs-asymptote tradeoff of the a-posteriori term
is visible: does a larger ``beta2`` descend faster early before its bias raises
the floor?

One panel per step size ``eta`` (each ``ss*`` subdir), each panel overlaying every
``beta2`` arm (the ``beta2=0`` control drawn dashed/gray). Reads every
``output/<prefix>/**/<tag>_save.pt`` checkpoint (handles both the flat
``ss*/save.pt`` and the sharded ``ss*/b*/save.pt`` one-arm-per-job layouts).

Usage:
    python scripts/plot_post_convergence.py --prefix sd_post_L300k
"""
import argparse
import glob
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(os.path.dirname(HERE), "output")


def _step_from_name(name: str) -> float:
    m = re.search(r"ss([0-9p]+)", name)
    return float(m.group(1).replace("p", ".")) if m else float("nan")


def _post_of(label: str) -> float:
    """a-posteriori weight beta2 from the arm label ('b2=<value>')."""
    m = re.search(r"b2=([0-9.]+)", label)
    return float(m.group(1)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="sd_post_L300k")
    ap.add_argument("--tag", default="sd6_od2")
    ap.add_argument("--out", default=None,
                    help="output png path (default output/<prefix>_convergence.png)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(OUTPUT, args.prefix, "**", f"{args.tag}_save.pt"),
                             recursive=True))
    if not paths:
        raise SystemExit(f"no checkpoints matching {args.prefix}/**/{args.tag}_save.pt")

    # (step, beta2) -> (steps_axis, err_curve); floor is shared across the sweep.
    curves: dict[float, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
    floor = None
    for p in paths:
        step = _step_from_name(p)
        pl = torch.load(p, map_location="cpu", weights_only=False)
        floor = float(pl["floor"])
        steps_axis = np.asarray(pl["steps_axis"], dtype=float)
        for d in pl["results"]:
            b2 = _post_of(d["label"])
            if not np.isfinite(b2):
                continue
            curves.setdefault(step, {})[b2] = (steps_axis, np.asarray(d["err_curve"], dtype=float))

    steps = sorted(curves)
    betas = sorted({b for by_b in curves.values() for b in by_b})

    def arm_label(b2: float) -> str:
        return "beta2=0 (control)" if b2 == 0.0 else f"beta2={b2:g}"

    def arm_color(b2: float):
        if b2 == 0.0:
            return "0.3"
        nz = [b for b in betas if b > 0.0]
        return plt.cm.viridis(nz.index(b2) / max(1, len(nz) - 1))

    ncols = min(3, len(steps))
    nrows = int(np.ceil(len(steps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.4 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, step in zip(axes_flat, steps):
        for b2 in betas:
            if b2 not in curves[step]:
                continue
            xs, err = curves[step][b2]
            excess = (err - floor) / floor
            excess = np.where(excess > 0, excess, np.nan)          # log-safe
            ax.plot(xs, excess, lw=1.5, color=arm_color(b2), label=arm_label(b2),
                    ls="--" if b2 == 0.0 else "-")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("online step")
        ax.set_ylabel(r"excess over floor $(err-\mathrm{irr})/\mathrm{irr}$")
        ax.set_title(rf"$\eta$ = {step:g}")
        ax.grid(True, which="both", alpha=0.25)

    # Hide any unused panels; put a single shared legend on the first.
    for ax in axes_flat[len(steps):]:
        ax.set_visible(False)
    axes_flat[0].legend(fontsize=8, loc="best")

    fig.suptitle(f"a-posteriori (beta2) SD sweep -- convergence per step size -- {args.prefix}\n"
                 f"(floor = {floor:.4g}; one panel per eta, one line per beta2)")
    fig.tight_layout()
    out = args.out or os.path.join(OUTPUT, f"{args.prefix}_convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
