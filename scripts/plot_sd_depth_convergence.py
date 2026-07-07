#!/usr/bin/env python
"""Per-depth convergence overlay: does a deeper ladder converge faster but to a
higher floor (the same speed/floor tradeoff as the launch gradient)?

For one sweep prefix and step size, overlays the excess-vs-context-step curve
(``(err_curve - floor)/floor`` vs ``steps_axis``) for every ladder depth n. If
the hypothesis holds, higher-n curves lead early (steeper initial descent) and are
overtaken by lower-n curves at the tail (the crossover).

Usage:
    python scripts/plot_sd_depth_convergence.py \
        --prefixes sd_depth_L300k,sd_depth_detach_L300k --step 0.03 --depths 1,2,3,4
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
    return ("%g" % step).replace(".", "p")


def _depth_of(label: str) -> int:
    m = re.search(r"n=(\d+)", label)
    return int(m.group(1)) if m else -1


def _load(prefix, tag, step, depth):
    path = os.path.join(OUTPUT, prefix, f"ss{_tag_for_step(step)}", f"{tag}_save.pt")
    if not os.path.exists(path):
        return None
    pl = torch.load(path, map_location="cpu", weights_only=False)
    floor = float(pl["floor"])
    sx = np.asarray(pl["steps_axis"], dtype=float)
    for d in pl["results"]:
        if _depth_of(d["label"]) == depth:
            err = np.asarray(d["err_curve"], dtype=float)
            exc = (err - floor) / floor * 100.0
            return sx, np.where(exc > 0, exc, np.nan)
    return None


def _title(prefix):
    return "keep_launch=True" if "detach" not in prefix else "keep_launch=False (detach)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", default="sd_depth_L300k,sd_depth_detach_L300k",
                    help="comma-separated sweep prefixes; one panel each")
    ap.add_argument("--tag", default="sd6_od2")
    ap.add_argument("--step", type=float, default=0.03)
    ap.add_argument("--depths", default="1,2,3,4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prefixes = [p for p in args.prefixes.split(",") if p]
    depths = [int(x) for x in args.depths.split(",") if x]
    colors = {d: plt.cm.viridis(i / max(1, len(depths) - 1)) for i, d in enumerate(depths)}

    fig, axes = plt.subplots(1, len(prefixes), figsize=(6.6 * len(prefixes), 5.2), squeeze=False)
    for ax, prefix in zip(axes[0], prefixes):
        for d in depths:
            got = _load(prefix, args.tag, args.step, d)
            if got is None:
                continue
            sx, exc = got
            ax.plot(sx, exc, color=colors[d], lw=1.7, marker=".", ms=3, label=f"n={d}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("online step (context length)")
        ax.set_ylabel(r"excess over floor $(err-\mathrm{irr})/\mathrm{irr}$ (%)")
        ax.set_title(f"{_title(prefix)}  ($\\eta$={args.step:g})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("SD ladder depth convergence: deeper n converges faster but settles higher")
    fig.tight_layout()
    out = args.out or os.path.join(OUTPUT, f"sd_depth_convergence_eta{_tag_for_step(args.step)}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
