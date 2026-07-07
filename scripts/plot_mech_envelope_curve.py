#!/usr/bin/env python
"""Standard two-panel analytical-error curve for the ``mech`` study, but with each
of the three methods drawn at its OWN best step (the per-method-best envelope).

Mirrors ``plot_analytical_error_curve`` in ``self_distillation_losses.py`` (left:
analytical observation error vs online step with the irreducible floor and
zero-predictor ceiling; right: excess over the floor on a log axis), except the
three curves are pulled from different ``--step-size`` runs -- each method from
the step that minimises its steady excess. The x-axis (online step = context
length) is common, so the overlay is meaningful.

Usage:
    python scripts/plot_mech_envelope_curve.py --prefix sd_mech_L100000
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


def _tail_stable(d, frac=0.2):
    a = np.asarray(d.get("frac_stable", []), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    k = max(1, int(round(a.size * frac)))
    return float(np.nanmean(a[-k:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="sd_mech_L100000")
    ap.add_argument("--tag", default="sd6_od2")
    ap.add_argument("--out", default=None, help="output png path (default output/<prefix>_envelope_curve.png)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(OUTPUT, args.prefix, "ss*", f"{args.tag}_save.pt")))
    if not paths:
        raise SystemExit(f"no checkpoints matching {args.prefix}/ss*/{args.tag}_save.pt")

    # Gather, per (method, step): the result dict + that run's steps_axis/floor/ceiling.
    by_method = {}   # label -> list of (step, result, steps_axis, floor, ceiling)
    floor = ceiling = None
    for p in paths:
        step = _step_from_name(os.path.basename(os.path.dirname(p)))
        pl = torch.load(p, map_location="cpu", weights_only=False)
        floor = float(pl["floor"]); ceiling = float(pl["ceiling"])
        steps_axis = np.asarray(pl["steps_axis"])
        for d in pl["results"]:
            lab = d["label"].splitlines()[0]
            by_method.setdefault(lab, []).append((step, d, steps_axis))

    # For each method pick the step with the smallest finite steady excess.
    chosen = {}
    for lab, entries in by_method.items():
        finite = [(s, d, sx) for (s, d, sx) in entries
                  if np.isfinite(d.get("final_excess", np.inf))]
        if not finite:
            continue
        chosen[lab] = min(finite, key=lambda e: e[1]["final_excess"])

    # Stable method order: a-priori, SD+anchor, oracle (so colors are consistent).
    def order_key(lab):
        return (0 if lab.startswith("M4 constant") else
                1 if lab.startswith("SD+anchor") else 2)
    labels = sorted(chosen.keys(), key=order_key)

    fig, (ax_err, ax_exc) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    colors = {lab: f"C{order_key(lab)}" for lab in labels}

    for lab in labels:
        step, d, steps = chosen[lab]
        c = colors[lab]
        exc = d["final_excess"] * 100.0
        stbl = _tail_stable(d) * 100.0
        leg = f"{lab}  (eta={step:g}: {exc:.3f}% exc, {stbl:.0f}% stable)"
        err = np.asarray(d["err_curve"], dtype=float)
        ax_err.plot(steps, err, color=c, lw=1.6, marker=".", ms=3, label=leg)
        e = (err - floor) / floor
        e = np.where(e > 0, e, np.nan)
        ax_exc.plot(steps, e, color=c, lw=1.6, marker=".", ms=3, label=leg)
        # Raw (no-averaging) overlay where it differs -- the oracle's Polyak case.
        raw = np.asarray(d.get("raw_err_curve", np.empty(0)), dtype=float)
        if raw.size == err.size and not np.allclose(np.nan_to_num(raw), np.nan_to_num(err)):
            ax_err.plot(steps, raw, color=c, lw=1.0, ls="--", alpha=0.5, label=f"{lab} raw")
            re_ = np.where((raw - floor) / floor > 0, (raw - floor) / floor, np.nan)
            ax_exc.plot(steps, re_, color=c, lw=1.0, ls="--", alpha=0.5, label=f"{lab} raw")

    ax_err.axhline(floor, color="k", ls="--", lw=1.2, label="irreducible floor")
    ax_err.axhline(ceiling, color="0.5", ls=":", lw=1.2, label="zero-predictor ceiling")
    ax_err.set_yscale("log")
    ax_err.set_xlabel("online step (context length)")
    ax_err.set_ylabel("analytical observation error")
    ax_err.set_title("Filter error vs adaptation step (median over trajectories)")
    ax_err.legend(fontsize=8, loc="best")

    ax_exc.axhline(0.0, color="k", ls="--", lw=0.8)
    ax_exc.set_yscale("log")
    ax_exc.set_xlabel("online step (context length)")
    ax_exc.set_ylabel(r"excess over floor $(err-\mathrm{irr})/\mathrm{irr}$")
    ax_exc.set_title("Excess over irreducible (-> 0 means reaching the optimum)")
    ax_exc.legend(fontsize=8, loc="best")

    fig.suptitle("Constant-gain mechanism study: three methods, each at its own best step "
                 "(exact closed form; gaps = every trajectory unstable)")
    fig.tight_layout()
    out = args.out or os.path.join(OUTPUT, f"{args.prefix}_envelope_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    print("chosen steps: " + ", ".join(f"{lab.split()[0]}@{chosen[lab][0]:g}" for lab in labels))


if __name__ == "__main__":
    main()
