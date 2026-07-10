#!/usr/bin/env python
"""a-posteriori-focused readout of the beta2 SD sweep (``--methods post``).

Reads every ``output/<prefix>/ss*/<tag>_save.pt`` checkpoint produced by
``submit_post_sweep.sh`` and renders the two views the a-posteriori study calls
for (mirror of ``plot_sd_depth_sweep.py``, with the arm axis being the
a-posteriori weight ``beta2`` instead of the ladder depth ``n``):

- left: steady excess over the irreducible floor vs step-size, one line per
  ``beta2`` weight (the ``beta2=0`` control drawn dashed/gray) -- the full
  tradeoff grid, with unstable/diverged points dropped;
- right: each arm's per-method-best excess (its own best step), i.e. the fair
  envelope comparison, as a bar per ``beta2``.

Usage:
    python scripts/plot_post_sweep.py --prefix sd_post_L300k
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
    # Parse the ``ssXXX`` step token anywhere in the path, so both the flat
    # ``<prefix>/ss<step>/`` layout and the sharded ``<prefix>/ss<step>/b<beta>/``
    # layout (one arm per job) resolve to the same step.
    m = re.search(r"ss([0-9p]+)", name)
    return float(m.group(1).replace("p", ".")) if m else float("nan")


def _post_of(label: str) -> float:
    """The a-posteriori weight beta2 parsed from the arm label ('b2=...').

    Every ``post`` arm label carries a 'b2=<value>' tag (the control is 'b2=0'),
    so this is unambiguous; falls back to nan for any non-post label."""
    m = re.search(r"b2=([0-9.]+)", label)
    return float(m.group(1)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="sd_post_L300k")
    ap.add_argument("--tag", default="sd6_od2")
    ap.add_argument("--out", default=None,
                    help="output png path (default output/<prefix>_post_sweep.png)")
    args = ap.parse_args()

    # Recursive glob matches both the flat ``ss*/save.pt`` layout and the sharded
    # ``ss*/b*/save.pt`` layout (one arm per job); each checkpoint's arm(s) merge
    # into the per-(beta2, step) grid, with the step parsed from the path.
    paths = sorted(glob.glob(os.path.join(OUTPUT, args.prefix, "**", f"{args.tag}_save.pt"),
                             recursive=True))
    if not paths:
        raise SystemExit(f"no checkpoints matching {args.prefix}/**/{args.tag}_save.pt")

    # arm (beta2) -> {step: (excess, n_diverged)}
    grid: dict[float, dict[float, tuple[float, int]]] = {}
    for p in paths:
        step = _step_from_name(p)
        pl = torch.load(p, map_location="cpu", weights_only=False)
        for d in pl["results"]:
            b2 = _post_of(d["label"])
            if not np.isfinite(b2):
                continue
            grid.setdefault(b2, {})[step] = (
                float(d.get("final_excess", float("nan"))),
                int(d.get("n_diverged", 0)),
            )

    betas = sorted(grid)                               # [0.0 (control), 0.01, ...]
    steps = sorted({s for by_step in grid.values() for s in by_step})

    def arm_label(b2: float) -> str:
        return "beta2=0 (control)" if b2 == 0.0 else f"beta2={b2:g}"

    def arm_color(b2: float):
        if b2 == 0.0:
            return "0.3"
        nz = [b for b in betas if b > 0.0]
        return plt.cm.viridis(nz.index(b2) / max(1, len(nz) - 1))

    fig, (ax_grid, ax_best) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # LEFT: excess vs step-size per arm (log-log), diverged points annotated.
    for b2 in betas:
        xs = [s for s in steps if s in grid[b2]]
        ys = [grid[b2][s][0] * 100 for s in xs]
        ys = [y if np.isfinite(y) and y > 0 else np.nan for y in ys]
        ax_grid.plot(xs, ys, marker="o", ms=5, lw=1.6,
                     color=arm_color(b2), label=arm_label(b2),
                     ls="--" if b2 == 0.0 else "-")
        for s in xs:
            exc, ndiv = grid[b2][s]
            if not np.isfinite(exc):
                ax_grid.annotate("inf", (s, ax_grid.get_ylim()[0]), fontsize=7,
                                 color=arm_color(b2), ha="center")
    ax_grid.set_xscale("log")
    ax_grid.set_yscale("log")
    ax_grid.set_xlabel(r"step size $\eta$")
    ax_grid.set_ylabel("steady excess over floor (%)")
    ax_grid.set_title(r"Steady excess vs step size, per a-posteriori weight $\beta_2$")
    ax_grid.legend(fontsize=8, loc="best")
    ax_grid.grid(True, which="both", alpha=0.25)

    # RIGHT: per-arm best (its own best step) -- the fair envelope comparison.
    best = {}
    for b2 in betas:
        finite = [(s, e) for s, (e, _) in grid[b2].items() if np.isfinite(e)]
        if finite:
            best[b2] = min(finite, key=lambda t: t[1])
    xs = np.arange(len(best))
    labels, vals, bests = [], [], []
    for b2 in sorted(best):
        s, e = best[b2]
        labels.append(arm_label(b2))
        vals.append(e * 100)
        bests.append(s)
    ax_best.bar(xs, vals, color=[arm_color(b) for b in sorted(best)], alpha=0.9)
    for x, v, s in zip(xs, vals, bests):
        ax_best.text(x, v, f"{v:.3f}%\n@$\\eta$={s:g}", ha="center", va="bottom", fontsize=8)
    ax_best.set_xticks(xs)
    ax_best.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax_best.set_ylabel("best steady excess over floor (%)")
    ax_best.set_title("Per-arm best excess (each at its own best step)")
    ax_best.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"a-posteriori (beta2) SD sweep -- {args.prefix}")
    fig.tight_layout()
    out = args.out or os.path.join(OUTPUT, f"{args.prefix}_post_sweep.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
