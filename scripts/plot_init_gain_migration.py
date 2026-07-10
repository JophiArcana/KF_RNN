#!/usr/bin/env python
"""Gain-migration diagnostic for the A-init pathway SD sweep (``--methods init``).

The toy realizes "K = A" as the single gain ``K = H^+`` (the REPLACE / high-gain
limit: the observable subspace is replaced by ``H^+ y``, ``null(H)`` coasts). This
diagnostic measures how far the *learned* gain has walked from that replace limit
toward the Kalman gain, using two endpoint norms of the trajectory-averaged final
filter ``(K, H)``:

- ``||K||``          -- the gain magnitude. The ``K = 0`` arms start at 0 and grow;
  the ``K = H^+`` arms start at ``||H^+||`` and adapt from there.
- ``||KH - H^+H||``  -- distance from the REPLACE limit in the observable subspace.
  It is exactly 0 at ``K = H^+`` (since then ``KH = H^+H``) and grows as the gain
  walks toward the Kalman gain. This is the toy shadow of real-case regions
  funnelling A -> K: 0 = sitting at the A / replace default, large = a genuine
  Kalman corrector.

``H^+`` is recomputed per cell from that cell's own final ``H`` (each arm re-seeds
the same ``H``, but training moves it), so the metric tracks the actual decoder.

Two panels (``||K||`` and ``||KH - H^+H||``) vs step size, one line per init arm,
plus a printed per-(step, arm) table. Reads every
``output/<prefix>/**/<tag>_save.pt`` checkpoint (flat or sharded layout).

Usage:
    python scripts/plot_init_gain_migration.py --prefix sd_init_L300k
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

ARM_ORDER = ["fI_k0", "fI_kpinv", "f0_kpinv", "f0_k0", "fI_k0_b0p05", "f0_kpinv_b0p05"]


def _step_from_name(name: str) -> float:
    m = re.search(r"ss([0-9p]+)", name)
    return float(m.group(1).replace("p", ".")) if m else float("nan")


def _arm_of(label: str) -> str:
    return label.splitlines()[0].strip()


def _migration(K, H):
    """(||K||, ||KH - H^+H||) for one final filter, or (nan, nan) if unavailable."""
    if K is None or H is None:
        return float("nan"), float("nan")
    K = torch.as_tensor(K, dtype=torch.float64)
    H = torch.as_tensor(H, dtype=torch.float64)
    if not (torch.isfinite(K).all() and torch.isfinite(H).all()):
        return float("nan"), float("nan")
    Hpinv = torch.linalg.pinv(H)
    k_norm = float(K.norm())
    replace_gap = float((K @ H - Hpinv @ H).norm())
    return k_norm, replace_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="sd_init_L300k")
    ap.add_argument("--tag", default="sd6_od2")
    ap.add_argument("--out", default=None,
                    help="output png path (default output/<prefix>_gain_migration.png)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(OUTPUT, args.prefix, "**", f"{args.tag}_save.pt"),
                             recursive=True))
    if not paths:
        raise SystemExit(f"no checkpoints matching {args.prefix}/**/{args.tag}_save.pt")

    # (step) -> {arm -> (||K||, ||KH - H^+H||)}
    data: dict[float, dict[str, tuple[float, float]]] = {}
    for p in paths:
        step = _step_from_name(p)
        pl = torch.load(p, map_location="cpu", weights_only=False)
        for d in pl["results"]:
            arm = _arm_of(d["label"])
            data.setdefault(step, {})[arm] = _migration(d.get("K_hat_mean"), d.get("H_hat_mean"))

    steps = sorted(data)
    arms_present = [a for a in ARM_ORDER if any(a in by_a for by_a in data.values())]
    arms_present += sorted({a for by_a in data.values() for a in by_a} - set(arms_present))

    # Printed table.
    print(f"\nA-init gain migration ({args.prefix}): ||K|| and ||KH - H^+H|| "
          f"(0 == the K=H^+ replace limit)\n")
    hdr = f"{'arm':18s} {'step':>6s} {'||K||':>10s} {'||KH-H+H||':>12s}"
    print(hdr)
    print("-" * len(hdr))
    for arm in arms_present:
        for s in steps:
            if arm not in data[s]:
                continue
            kn, gap = data[s][arm]
            print(f"{arm:18s} {s:6g} {kn:10.4f} {gap:12.4f}")
        print()

    # Two-panel plot: ||K|| and the replace-gap vs step, one line per arm.
    def arm_color(arm):
        return plt.cm.tab10(arms_present.index(arm) % 10)

    fig, (ax_k, ax_gap) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for arm in arms_present:
        xs = [s for s in steps if arm in data[s]]
        kn = [data[s][arm][0] for s in xs]
        gap = [data[s][arm][1] for s in xs]
        ax_k.plot(xs, kn, marker="o", lw=1.6, color=arm_color(arm), label=arm)
        ax_gap.plot(xs, gap, marker="o", lw=1.6, color=arm_color(arm), label=arm)

    for ax, title, ylab in (
        (ax_k, r"gain magnitude $\|K\|$", r"$\|K\|$"),
        (ax_gap, r"distance from the replace limit $\|KH - H^+H\|$",
         r"$\|KH - H^+H\|$  (0 = replace limit)"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("step size $\\eta$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    ax_gap.axhline(0.0, color="k", ls="--", lw=0.8)

    fig.suptitle(f"A-init gain migration -- {args.prefix}\n"
                 f"(final trajectory-averaged gain; the filter walking from K=H^+ "
                 f"toward the Kalman gain)")
    fig.tight_layout()
    out = args.out or os.path.join(OUTPUT, f"{args.prefix}_gain_migration.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
