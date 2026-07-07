#!/usr/bin/env bash
# Submit the n-step self-distillation DEPTH study as a PER-STEP-SIZE SLURM sweep
# (one job per --step-size, so they run in parallel). Each job runs
# scripts/self_distillation_losses.py --methods nstep, whose arms are the pure
# a-priori control plus one SD+anchor arm per ladder depth n = 1..WINDOW -- so a
# single job already sweeps the SD depth, and the step-size loop below adds the
# second axis (each depth is fairest at its own best step, as in the mech study).
#
# A ladder horizon deeper than the window clamps to the detached root, so WINDOW
# is the natural depth ceiling (default 4 == the study's default --window).
#
# This is a thin loop over scripts/submit_sd_losses.sh -- it forwards the same
# submission env knobs (GPU_TYPE, TIME, CPUS, PARTITION, MEM, GPUS, DRY_RUN) to
# every per-step submission, and gives each its own output dir / job name.
#
# Usage (run on the login node, e.g. kopsgridlogin1, NOT inside a grid-tunnel):
#   GPU_TYPE=h100 TIME=04:00:00 bash scripts/submit_sd_depth_sweep.sh
#
#   # see every sbatch command without submitting:
#   DRY_RUN=1 bash scripts/submit_sd_depth_sweep.sh
#
# Override the sweep shape via env vars (these are exclusive to this wrapper):
#   STEPS        space-separated --step-size grid   (default "0.01 0.03 0.1 0.3 1.0")
#   OUT_PREFIX   per-run --out-name prefix          (default sd_depth)
# All submission knobs (GPU_TYPE, TIME, ...) are read by submit_sd_losses.sh.
#
# Every study flag is a baked, overridable default (see DEFAULT_ARGS below). Any
# trailing CLI flags are forwarded verbatim to self_distillation_losses.py,
# appended AFTER the baked defaults, so they OVERRIDE them (argparse takes the
# last value). So the SD depth ceiling, FIR lengths, anchor, trace length, etc.
# are just passed as trailing args, e.g. a depth-6 run out to 100k steps:
#   GPU_TYPE=h100 bash scripts/submit_sd_depth_sweep.sh --window 6 -L 100000
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMIT="$PROJECT_ROOT/scripts/submit_sd_losses.sh"

# Trailing CLI flags (e.g. --window 6 -L 100000) are forwarded to each per-step
# submission, AFTER the baked defaults below, so they override them.
EXTRA_ARGS=("$@")

# Exclusive-to-this-wrapper knobs: the per-job --step-size axis and the prefix
# used to derive a distinct --out-name for each parallel job.
STEPS="${STEPS:-0.01 0.03 0.1 0.3 1.0}"
OUT_PREFIX="${OUT_PREFIX:-sd_depth}"

# Baked study config (all overridable by trailing args). --methods nstep is the
# study identity; --window is the SD depth ceiling (ladder depths n=1..window).
DEFAULT_ARGS=(
  --methods nstep --s-d 6 --o-d 2 --window 4 --eps 0.1
  -N 16 --weight-decay 0 --anchor 0.05 --device cuda
  --fir-lengths 1,2,3,4,5,6,7,8 --online-fir-lengths 2,4,8,16
  --analytic-stride 100 --seed 0
)

if [ ! -f "$SUBMIT" ]; then
  echo "ERROR: $SUBMIT not found" >&2
  exit 1
fi

echo "============================================================"
echo "SD depth sweep: STEPS = [$STEPS]"
echo "  out-prefix=$OUT_PREFIX  defaults: ${DEFAULT_ARGS[*]}"
[ "${#EXTRA_ARGS[@]}" -gt 0 ] && echo "  extra (override) args: ${EXTRA_ARGS[*]}"
echo "============================================================"

for ss in $STEPS; do
  # filesystem-safe tag: 0.01 -> 0p01, 0.1 -> 0p1, 1.0 -> 1p0
  tag="${ss//./p}"
  out_name="${OUT_PREFIX}/ss${tag}"
  echo
  echo ">>> submitting step-size=$ss  ->  out-name=$out_name"
  bash "$SUBMIT" "${DEFAULT_ARGS[@]}" \
    --step-size "$ss" --out-name "$out_name" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
done

echo
echo "all per-step jobs submitted (or printed under DRY_RUN)."
echo "outputs land in $PROJECT_ROOT/output/${OUT_PREFIX}/ss<step>/"
