#!/usr/bin/env python
"""Summarize the constant-gain-centric ``mech`` step-size sweep.

Reads every ``output/<prefix>/ss*/sd6_od2_save.pt`` checkpoint produced by
``self_distillation_losses.py --methods mech`` and builds the per-method-best
("Pareto envelope") view the study calls for:

- a per-(step-size, method) table of steady excess over the irreducible floor
  (both the reported/averaged readout and the raw iterate), tail closed-loop
  stability, divergence count, and median |lambda(F_hat)|;
- for each method, its BEST step (minimum steady excess) with the stability at
  that step -- the fair "each method at its own best step" comparison.

Usage:
    python scripts/analyze_mech_sweep.py [--prefix sd_mech] [--tail-frac 0.2]
"""
import argparse
import glob
import os
import re

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(os.path.dirname(HERE), "output")


def _tail(arr, frac):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)] if a.size else a
    if a.size == 0:
        return float("nan")
    k = max(1, int(round(a.size * frac)))
    return float(np.nanmean(a[-k:]))


def _step_from_name(name: str) -> float:
    m = re.search(r"ss([0-9p]+)$", name)
    if not m:
        return float("nan")
    return float(m.group(1).replace("p", "."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="sd_mech")
    ap.add_argument("--tag", default="sd6_od2")
    ap.add_argument("--tail-frac", type=float, default=0.2,
                    help="fraction of the tail sampled-steps used for the stability average")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(OUTPUT, args.prefix, "ss*", f"{args.tag}_save.pt")))
    if not paths:
        print(f"no checkpoints matching {args.prefix}/ss*/{args.tag}_save.pt yet")
        return

    # rows[(method_label)] -> list of dicts across steps
    rows = []
    floors = {}
    for p in paths:
        run = os.path.basename(os.path.dirname(p))
        step = _step_from_name(run)
        pl = torch.load(p, map_location="cpu", weights_only=False)
        floors[step] = float(pl["floor"])
        for d in pl["results"]:
            lab = d["label"].replace("\n", " ")
            rows.append(dict(
                step=step, method=lab,
                exc_avg=float(d.get("final_excess", float("nan"))),
                exc_raw=float(d.get("raw_final_excess", float("nan"))),
                stbl_avg=_tail(d.get("frac_stable", []), args.tail_frac),
                stbl_raw=_tail(d.get("raw_frac_stable", []), args.tail_frac),
                ndiv=int(d.get("n_diverged", -1)),
                radius=float(d.get("radius_med", float("nan"))),
            ))

    steps = sorted(set(r["step"] for r in rows))
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    print(f"\nfloors by step: " + ", ".join(f"{s:g}:{floors[s]:.4f}" for s in steps))
    print(f"(tail stability = mean over last {args.tail_frac:.0%} of sampled steps)\n")

    hdr = f"{'method':28s} {'step':>6s} {'exc_avg':>9s} {'exc_raw':>9s} {'stbl_av':>7s} {'stbl_rw':>7s} {'ndiv':>4s} {'rad':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for meth in methods:
        for s in steps:
            r = next((x for x in rows if x["method"] == meth and x["step"] == s), None)
            if r is None:
                continue
            print(f"{meth:28s} {s:6g} {r['exc_avg']*100:8.3f}% {r['exc_raw']*100:8.3f}% "
                  f"{r['stbl_avg']:7.2f} {r['stbl_raw']:7.2f} {r['ndiv']:4d} {r['radius']:5.2f}")
        print()

    # Per-method-best envelope: minimum steady excess (averaged readout) over steps.
    print("=" * 72)
    print("PER-METHOD BEST (each method at its own best step; fair comparison)")
    print("=" * 72)
    print(f"{'method':28s} {'best step':>9s} {'best exc':>9s} {'stbl@best':>9s} {'ndiv':>4s}")
    for meth in methods:
        cand = [x for x in rows if x["method"] == meth and np.isfinite(x["exc_avg"])]
        if not cand:
            print(f"{meth:28s} {'--':>9s}  (no finite results yet)")
            continue
        best = min(cand, key=lambda x: x["exc_avg"])
        print(f"{meth:28s} {best['step']:9g} {best['exc_avg']*100:8.3f}% {best['stbl_avg']:9.2f} {best['ndiv']:4d}")


if __name__ == "__main__":
    main()
