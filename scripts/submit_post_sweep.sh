#!/usr/bin/env bash
# Submit the a-posteriori (beta2) SD sweep as a SHARDED SLURM sweep: ONE job per
# (--step-size, beta2) cell -- i.e. STEPS x BETAS jobs, each running
# scripts/self_distillation_losses.py --methods post with a SINGLE-value
# --post-grid (one SD+anchor arm) on the FIXED detached-launch, n=window base.
#
# Why sharded (one arm per job)? Every post arm runs the full n=window ladder --
# the heaviest arm in the depth study -- so packing all 8 beta2 arms into one job
# (the earlier layout) blew past the wall-clock limit and saved nothing (the
# checkpoint is only written after all arms in a run finish). Sharding makes each
# job a single n=window arm (~the cost of one depth-sweep arm, well under any
# limit) and 40 jobs run in parallel. Each shard writes its own checkpoint at
# output/<prefix>/ss<step>/b<beta>/<tag>_save.pt; analyze_mech_sweep.py and
# plot_post_sweep.py glob these recursively and merge them into the usual
# per-(step, beta2) grid.
#
# The base config is settled by the earlier depth/launch studies: latent SD + light
# a-priori anchor (alpha=1, beta0=0.05), the DETACHED launch (--no-keep-launch), the
# n = window ladder (--window sets sd_horizon), constant gain. The only independent
# variable is beta2 (the a-posteriori observation loss), which is known to speed
# convergence but bias the asymptote -- this sweep characterizes that tradeoff.
#
# This is a thin loop over scripts/submit_sd_losses.sh -- it forwards the same
# submission env knobs (GPU_TYPE, TIME, CPUS, PARTITION, MEM, GPUS, DRY_RUN) to
# every per-cell submission, and gives each its own output dir / job name.
#
# Usage (run on the login node, e.g. kopsgridlogin1, NOT inside a grid-tunnel):
#   GPU_TYPE=h100 TIME=04:00:00 bash scripts/submit_post_sweep.sh
#
#   # see every sbatch command without submitting:
#   DRY_RUN=1 bash scripts/submit_post_sweep.sh
#
# Override the sweep shape via env vars (these are exclusive to this wrapper):
#   STEPS        space-separated --step-size grid   (default "0.01 0.03 0.1 0.3 1.0")
#   BETAS        space-separated beta2 grid (one job each)
#                                                   (default "0.0 0.01 0.02 0.05 0.1 0.2 0.5 1.0")
#   OUT_PREFIX   per-run --out-name prefix          (default sd_post_L300k)
# All submission knobs (GPU_TYPE, TIME, ...) are read by submit_sd_losses.sh.
#
# Every study flag is a baked, overridable default (see DEFAULT_ARGS below). Any
# trailing CLI flags are forwarded verbatim to self_distillation_losses.py,
# appended AFTER the baked defaults, so they OVERRIDE them (argparse takes the
# last value). So the trace length, FIR lengths, anchor, etc. are just passed as
# trailing args, e.g. a shorter horizon:
#   bash scripts/submit_post_sweep.sh -L 100000
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMIT="$PROJECT_ROOT/scripts/submit_sd_losses.sh"

# Trailing CLI flags (e.g. -L 100000) are forwarded to each per-cell submission,
# AFTER the baked defaults below, so they override them.
EXTRA_ARGS=("$@")

# Exclusive-to-this-wrapper knobs: the two sweep axes (each (step, beta2) cell is
# its own job) and the prefix used to derive a distinct --out-name per job.
STEPS="${STEPS:-0.01 0.03 0.1 0.3 1.0}"
BETAS="${BETAS:-0.0 0.01 0.02 0.05 0.1 0.2 0.5 1.0}"
OUT_PREFIX="${OUT_PREFIX:-sd_post_L300k}"

# Baked study config (all overridable by trailing args). --methods post is the
# study identity; --no-keep-launch + --window 4 (=> sd_horizon=4=n) fix the
# detached-launch, n=window base that the preset also bakes per arm. --post-grid
# is set per job (a single beta2) in the loop below, so it is NOT baked here.
DEFAULT_ARGS=(
  --methods post --s-d 6 --o-d 2 --window 4 --eps 0.1
  -N 16 -L 300000 --weight-decay 0 --anchor 0.05 --no-keep-launch --device cuda
  --fir-lengths 1,2,3,4,5,6,7,8 --online-fir-lengths 2,4,8,16
  --analytic-stride 100 --seed 0
)

if [ ! -f "$SUBMIT" ]; then
  echo "ERROR: $SUBMIT not found" >&2
  exit 1
fi

echo "============================================================"
echo "SD a-posteriori (beta2) sweep (sharded, 1 arm/job)"
echo "  STEPS = [$STEPS]"
echo "  BETAS = [$BETAS]"
echo "  out-prefix=$OUT_PREFIX  defaults: ${DEFAULT_ARGS[*]}"
[ "${#EXTRA_ARGS[@]}" -gt 0 ] && echo "  extra (override) args: ${EXTRA_ARGS[*]}"
echo "============================================================"

njobs=0
for ss in $STEPS; do
  # filesystem-safe tag: 0.01 -> 0p01, 0.1 -> 0p1, 1.0 -> 1p0
  stag="${ss//./p}"
  for b in $BETAS; do
    btag="${b//./p}"
    out_name="${OUT_PREFIX}/ss${stag}/b${btag}"
    echo
    echo ">>> submitting step-size=$ss  beta2=$b  ->  out-name=$out_name"
    bash "$SUBMIT" "${DEFAULT_ARGS[@]}" \
      --step-size "$ss" --post-grid "$b" --out-name "$out_name" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    njobs=$((njobs + 1))
  done
done

echo
echo "all $njobs per-cell jobs submitted (or printed under DRY_RUN)."
echo "outputs land in $PROJECT_ROOT/output/${OUT_PREFIX}/ss<step>/b<beta>/"
echo "read out with:  python scripts/analyze_mech_sweep.py --prefix ${OUT_PREFIX}"
echo "                python scripts/plot_post_sweep.py    --prefix ${OUT_PREFIX}"
