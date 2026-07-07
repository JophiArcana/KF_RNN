#!/usr/bin/env python
"""Loss-over-online-steps readout for the n-step self-distillation depth sweep.

The depth-sweep analog of ``plot_mech_envelope_curve.py``: the same two panels
(left: analytical observation error vs online step with the irreducible floor and
zero-predictor ceiling; right: excess over the floor on a log axis), but the
overlaid curves are the pure a-priori ``M4 constant`` control and one SD+anchor
arm per ladder depth ``n = 1..window`` (``n=1`` is the single-step / previous
default). Distinct colors per depth so the marginal effect of deepening the
ladder is legible as context grows.

By default all curves are read from a single fixed ``--step-size`` run (the fair
"same gain, vary only the depth" comparison; the SD depths' individual best steps
all sit at the marginal-stability small-eta end, which is not a deployable
operating point). Pass ``--envelope`` to instead draw each arm at its own
lowest-steady-excess step, matching the mech envelope's framing.

Usage:
    python scripts/plot_sd_depth_curve.py [--prefix sd_depth] [--step-size 0.3]
    python scripts/plot_sd_depth_curve.py --envelope
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


def _depth_from_label(label: str):
    m = re.search(r"n=(\d+)", label)
    return int(m.group(1)) if m else None


def _tail_stable(d, frac=0.2):
    a = np.asarray(d.get("frac_stable", []), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    k = max(1, int(round(a.size * frac)))
    return float(np.nanmean(a[-k:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="sd_depth")
    ap.add_argument("--tag", default="sd6_od2")
    ap.add_argument("--step-size", type=float, default=0.3,
                    help="fixed step-size run to read all curves from (ignored under --envelope)")
    ap.add_argument("--envelope", action="store_true",
                    help="draw each arm at its own lowest-steady-excess step instead of a fixed step")
    ap.add_argument("--out", default=None, help="output png (default output/<prefix>_depth_curve.png)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(OUTPUT, args.prefix, "ss*", f"{args.tag}_save.pt")))
    if not paths:
        raise SystemExit(f"no checkpoints matching {args.prefix}/ss*/{args.tag}_save.pt")

    # label -> list of (step, result_dict, steps_axis); plus shared floor/ceiling.
    by_method = {}
    floor = ceiling = None
    for p in paths:
        step = _step_from_name(os.path.basename(os.path.dirname(p)))
        pl = torch.load(p, map_location="cpu", weights_only=False)
        floor = float(pl["floor"]); ceiling = float(pl["ceiling"])
        steps_axis = np.asarray(pl["steps_axis"])
        for d in pl["results"]:
            by_method.setdefault(d["label"].replace("\n", " "), []).append((step, d, steps_axis))

    # Choose the (step, result, steps_axis) to draw for each method.
    chosen = {}
    for lab, entries in by_method.items():
        if args.envelope:
            finite = [e for e in entries if np.isfinite(e[1].get("final_excess", np.inf))]
            if finite:
                chosen[lab] = min(finite, key=lambda e: e[1]["final_excess"])
        else:
            match = [e for e in entries if abs(e[0] - args.step_size) < 1e-9]
            if match:
                chosen[lab] = match[0]
    if not chosen:
        raise SystemExit(f"no run found at --step-size {args.step_size} "
                         f"(available: {sorted({s for es in by_method.values() for s, _, _ in es})})")

    # Order: M4 baseline first, then SD arms by ascending depth.
    def order_key(lab):
        if lab.startswith("M4 constant"):
            return (0, 0)
        return (1, _depth_from_label(lab) or 0)
    labels = sorted(chosen.keys(), key=order_key)

    # Colors: M4 in black, SD depths along viridis.
    sd_labels = [l for l in labels if not l.startswith("M4 constant")]
    depth_colors = {l: c for l, c in
                    zip(sd_labels, plt.cm.viridis(np.linspace(0.15, 0.9, max(1, len(sd_labels)))))}

    def color_of(lab):
        return "k" if lab.startswith("M4 constant") else depth_colors[lab]

    def short(lab):
        if lab.startswith("M4 constant"):
            return "M4 constant (a-priori)"
        n = _depth_from_label(lab)
        return f"SD+anchor n={n}"

    fig, (ax_err, ax_exc) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for lab in labels:
        step, d, steps = chosen[lab]
        c = color_of(lab)
        exc = d.get("final_excess", float("nan")) * 100.0
        stbl = _tail_stable(d) * 100.0
        leg = f"{short(lab)}  (eta={step:g}: {exc:.2f}% exc, {stbl:.0f}% stable)"
        err = np.asarray(d["err_curve"], dtype=float)
        ax_err.plot(steps, err, color=c, lw=1.6, marker=".", ms=3, label=leg)
        e = (err - floor) / floor
        e = np.where(e > 0, e, np.nan)
        ax_exc.plot(steps, e, color=c, lw=1.6, marker=".", ms=3, label=leg)

    ax_err.axhline(floor, color="0.4", ls="--", lw=1.2, label="irreducible floor")
    ax_err.axhline(ceiling, color="0.6", ls=":", lw=1.2, label="zero-predictor ceiling")
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

    mode = ("each arm at its own best step" if args.envelope
            else f"all arms at eta={args.step_size:g}")
    fig.suptitle(f"n-step self-distillation depth sweep: M4 vs SD depths n=1..{len(sd_labels)} "
                 f"({mode}; exact closed form)")
    fig.tight_layout()
    out = args.out or os.path.join(OUTPUT, f"{args.prefix}_depth_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    print("drawn: " + ", ".join(f"{short(l)}@eta={chosen[l][0]:g}" for l in labels))


if __name__ == "__main__":
    main()
