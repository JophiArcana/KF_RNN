#!/usr/bin/env python
"""Depth-focused readout of the n-step SD ladder sweep (``--methods nstep``).

Reads every ``output/<prefix>/ss*/<tag>_save.pt`` checkpoint produced by
``submit_sd_depth_sweep.sh`` and renders the two views the depth study calls for:

- left: steady excess over the irreducible floor vs step-size, one line per
  ladder depth n (plus the pure a-priori M4 control) -- the full tradeoff grid,
  with unstable/diverged points dropped;
- right: each arm's per-method-best excess (its own best step), i.e. the fair
  envelope comparison, as a bar per depth.

Usage:
    python scripts/plot_sd_depth_sweep.py --prefix sd_depth_L30k
    python scripts/plot_sd_depth_sweep.py --prefix sd_depth_L100k
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
    m = re.search(r"ss([0-9p]+)$", name)
    return float(m.group(1).replace("p", ".")) if m else float("nan")


def _depth_of(label: str) -> int:
    """-1 for the M4 control, else the ladder depth n from 'SD+anchor n=k'."""
    m = re.search(r"n=(\d+)", label)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="sd_depth_L30k")
    ap.add_argument("--tag", default="sd6_od2")
    ap.add_argument("--out", default=None,
                    help="output png path (default output/<prefix>_depth_sweep.png)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(OUTPUT, args.prefix, "ss*", f"{args.tag}_save.pt")))
    if not paths:
        raise SystemExit(f"no checkpoints matching {args.prefix}/ss*/{args.tag}_save.pt")

    # arm (depth) -> {step: (excess, n_diverged)}
    grid: dict[int, dict[float, tuple[float, int]]] = {}
    for p in paths:
        step = _step_from_name(os.path.basename(os.path.dirname(p)))
        pl = torch.load(p, map_location="cpu", weights_only=False)
        for d in pl["results"]:
            depth = _depth_of(d["label"])
            grid.setdefault(depth, {})[step] = (
                float(d.get("final_excess", float("nan"))),
                int(d.get("n_diverged", 0)),
            )

    depths = sorted(grid)                              # [-1 (M4), 1, 2, ...]
    steps = sorted({s for by_step in grid.values() for s in by_step})

    def arm_label(depth: int) -> str:
        return "M4 (a-priori)" if depth < 0 else f"SD+anchor n={depth}"

    def arm_color(depth: int) -> str:
        if depth < 0:
            return "0.3"
        sd_depths = [d for d in depths if d > 0]
        return plt.cm.viridis(sd_depths.index(depth) / max(1, len(sd_depths) - 1))

    fig, (ax_grid, ax_best) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # LEFT: excess vs step-size per arm (log-log), diverged points annotated.
    for depth in depths:
        xs = [s for s in steps if s in grid[depth]]
        ys = [grid[depth][s][0] * 100 for s in xs]
        ys = [y if np.isfinite(y) and y > 0 else np.nan for y in ys]
        ax_grid.plot(xs, ys, marker="o", ms=5, lw=1.6,
                     color=arm_color(depth), label=arm_label(depth),
                     ls="--" if depth < 0 else "-")
        for s in xs:
            exc, ndiv = grid[depth][s]
            if not np.isfinite(exc):
                ax_grid.annotate("inf", (s, ax_grid.get_ylim()[0]), fontsize=7,
                                 color=arm_color(depth), ha="center")
    ax_grid.set_xscale("log")
    ax_grid.set_yscale("log")
    ax_grid.set_xlabel(r"step size $\eta$")
    ax_grid.set_ylabel("steady excess over floor (%)")
    ax_grid.set_title("Steady excess vs step size, per ladder depth")
    ax_grid.legend(fontsize=8, loc="best")
    ax_grid.grid(True, which="both", alpha=0.25)

    # RIGHT: per-arm best (its own best step) -- the fair envelope comparison.
    best = {}
    for depth in depths:
        finite = [(s, e) for s, (e, _) in grid[depth].items() if np.isfinite(e)]
        if finite:
            best[depth] = min(finite, key=lambda t: t[1])
    xs = np.arange(len(best))
    labels, vals, bests = [], [], []
    for depth in sorted(best):
        s, e = best[depth]
        labels.append(arm_label(depth))
        vals.append(e * 100)
        bests.append(s)
    bars = ax_best.bar(xs, vals, color=[arm_color(d) for d in sorted(best)], alpha=0.9)
    for x, v, s in zip(xs, vals, bests):
        ax_best.text(x, v, f"{v:.3f}%\n@$\\eta$={s:g}", ha="center", va="bottom", fontsize=8)
    ax_best.set_xticks(xs)
    ax_best.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax_best.set_ylabel("best steady excess over floor (%)")
    ax_best.set_title("Per-arm best excess (each at its own best step)")
    ax_best.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"n-step SD ladder depth sweep -- {args.prefix}")
    fig.tight_layout()
    out = args.out or os.path.join(OUTPUT, f"{args.prefix}_depth_sweep.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
