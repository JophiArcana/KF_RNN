#!/usr/bin/env python
"""Convergence readout of the A-init pathway SD sweep (``--methods init``).

Companion to ``analyze_mech_sweep.py`` (which reports only the *steady* excess).
This plots the full adaptation trajectory -- excess over the irreducible floor vs
online step -- so the crossover story of the study is visible: does an A-init arm
(``K = H^+`` at ``F = 0`` or the ``F = ZOH`` hybrid) leave the transient early,
like a positive ``beta2`` does in the post sweep, WITHOUT paying a converged-floor
penalty (the "free accelerator" claim)?

One panel per step size ``eta`` (each ``ss*`` subdir), each panel overlaying every
init arm (the ``fI_k0`` current-init control drawn dashed/gray). Reads every
``output/<prefix>/**/<tag>_save.pt`` checkpoint (handles both a flat
``ss*/save.pt`` and the sharded ``ss*/<arm>/save.pt`` one-arm-per-job layout).

Usage:
    python scripts/plot_init_convergence.py --prefix sd_init_L300k
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

# Canonical arm order (matches init_methods in self_distillation_losses.py); the
# control is first and drawn dashed/gray, the rest get the viridis cycle.
ARM_ORDER = ["fI_k0", "fI_kpinv", "f0_kpinv", "f0_k0", "fI_k0_b0p05", "f0_kpinv_b0p05"]
CONTROL_ARM = "fI_k0"


def _step_from_name(name: str) -> float:
    m = re.search(r"ss([0-9p]+)", name)
    return float(m.group(1).replace("p", ".")) if m else float("nan")


def _arm_of(label: str) -> str:
    """The init-arm token is the first line of the arm label (see init_methods)."""
    return label.splitlines()[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="sd_init_L300k")
    ap.add_argument("--tag", default="sd6_od2")
    ap.add_argument("--out", default=None,
                    help="output png path (default output/<prefix>_convergence.png)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(OUTPUT, args.prefix, "**", f"{args.tag}_save.pt"),
                             recursive=True))
    if not paths:
        raise SystemExit(f"no checkpoints matching {args.prefix}/**/{args.tag}_save.pt")

    # (step) -> {arm -> (steps_axis, err_curve)}; floor is shared across the sweep.
    curves: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    floor = None
    for p in paths:
        step = _step_from_name(p)
        pl = torch.load(p, map_location="cpu", weights_only=False)
        floor = float(pl["floor"])
        steps_axis = np.asarray(pl["steps_axis"], dtype=float)
        for d in pl["results"]:
            arm = _arm_of(d["label"])
            curves.setdefault(step, {})[arm] = (steps_axis, np.asarray(d["err_curve"], dtype=float))

    steps = sorted(curves)
    arms_present = [a for a in ARM_ORDER if any(a in by_a for by_a in curves.values())]
    # Any unexpected arm token still gets drawn (appended after the known order).
    arms_present += sorted({a for by_a in curves.values() for a in by_a} - set(arms_present))
    nonctrl = [a for a in arms_present if a != CONTROL_ARM]

    def arm_color(arm: str):
        if arm == CONTROL_ARM:
            return "0.3"
        return plt.cm.viridis(nonctrl.index(arm) / max(1, len(nonctrl) - 1))

    ncols = min(3, len(steps))
    nrows = int(np.ceil(len(steps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.4 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, step in zip(axes_flat, steps):
        for arm in arms_present:
            if arm not in curves[step]:
                continue
            xs, err = curves[step][arm]
            excess = (err - floor) / floor
            excess = np.where(excess > 0, excess, np.nan)          # log-safe
            ax.plot(xs, excess, lw=1.5, color=arm_color(arm),
                    label=(f"{arm} (control)" if arm == CONTROL_ARM else arm),
                    ls="--" if arm == CONTROL_ARM else "-")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("online step")
        ax.set_ylabel(r"excess over floor $(err-\mathrm{irr})/\mathrm{irr}$")
        ax.set_title(rf"$\eta$ = {step:g}")
        ax.grid(True, which="both", alpha=0.25)

    for ax in axes_flat[len(steps):]:
        ax.set_visible(False)
    axes_flat[0].legend(fontsize=8, loc="best")

    fig.suptitle(f"A-init pathway SD sweep -- convergence per step size -- {args.prefix}\n"
                 f"(floor = {floor:.4g}; one panel per eta, one line per init arm; "
                 f"crossover = A-init leaves the transient early)")
    fig.tight_layout()
    out = args.out or os.path.join(OUTPUT, f"{args.prefix}_convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
