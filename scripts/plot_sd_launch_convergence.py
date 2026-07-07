#!/usr/bin/env python
"""Convergence-rate comparison of keep_launch=True vs False (detached launch).

The per-arm *final* excess conflates the asymptotic floor with how fast each
variant gets there. This overlays the full excess-vs-context-step curve
(``(err_curve - floor)/floor`` vs ``steps_axis``) for the two runs at matched
step sizes, so a slower-descending-but-lower-floor tradeoff is visible directly.

For each requested step size (a panel) and depth, the keep run is drawn solid and
the detach run dashed in the same color, so vertical gaps read as "same context,
different excess" and horizontal gaps as "same excess, different context needed".

Usage:
    python scripts/plot_sd_launch_convergence.py \
        --keep-prefix sd_depth_L100k --detach-prefix sd_depth_detach_L100k \
        --steps 0.01,0.03 --depths 1,2
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


def _tag_for_step(step: float) -> str:
    # 0.01 -> 0p01, 0.03 -> 0p03, 0.1 -> 0p1, 1.0 -> 1p0
    return ("%g" % step).replace(".", "p")


def _depth_of(label: str) -> int:
    m = re.search(r"n=(\d+)", label)
    return int(m.group(1)) if m else -1


def _load_curve(prefix: str, tag: str, step: float, depth: int):
    """Return (steps_axis, excess_pct, floor) for one (prefix, step, depth) or None."""
    path = os.path.join(OUTPUT, prefix, f"ss{_tag_for_step(step)}", f"{tag}_save.pt")
    if not os.path.exists(path):
        return None
    pl = torch.load(path, map_location="cpu", weights_only=False)
    floor = float(pl["floor"])
    steps_axis = np.asarray(pl["steps_axis"], dtype=float)
    for d in pl["results"]:
        if _depth_of(d["label"]) == depth:
            err = np.asarray(d["err_curve"], dtype=float)
            exc = (err - floor) / floor * 100.0
            exc = np.where(exc > 0, exc, np.nan)
            return steps_axis, exc, floor
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-prefix", default="sd_depth_L100k")
    ap.add_argument("--detach-prefix", default="sd_depth_detach_L100k")
    ap.add_argument("--tag", default="sd6_od2")
    ap.add_argument("--steps", default="0.01,0.03",
                    help="comma-separated step sizes; one panel each")
    ap.add_argument("--depths", default="1,2",
                    help="comma-separated ladder depths to overlay")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    steps = [float(x) for x in args.steps.split(",") if x]
    depths = [int(x) for x in args.depths.split(",") if x]

    fig, axes = plt.subplots(1, len(steps), figsize=(6.5 * len(steps), 5.2), squeeze=False)
    colors = {d: f"C{i}" for i, d in enumerate(depths)}

    for ax, step in zip(axes[0], steps):
        for d in depths:
            keep = _load_curve(args.keep_prefix, args.tag, step, d)
            det = _load_curve(args.detach_prefix, args.tag, step, d)
            c = colors[d]
            if keep is not None:
                sx, exc, _ = keep
                ax.plot(sx, exc, color=c, lw=1.8, ls="-", marker=".", ms=3,
                        label=f"n={d} keep")
            if det is not None:
                sx, exc, _ = det
                ax.plot(sx, exc, color=c, lw=1.6, ls="--", marker="x", ms=3,
                        label=f"n={d} detach")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("online step (context length)")
        ax.set_ylabel(r"excess over floor $(err-\mathrm{irr})/\mathrm{irr}$ (%)")
        ax.set_title(rf"$\eta$ = {step:g}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Launch-gradient convergence: keep_launch=True (solid) vs False/detached (dashed)")
    fig.tight_layout()
    out = args.out or os.path.join(OUTPUT, "sd_launch_convergence_compare.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
