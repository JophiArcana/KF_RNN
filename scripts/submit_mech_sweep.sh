#!/usr/bin/env bash
# Submit the constant-gain-centric "mech" mechanism study as a PER-STEP-SIZE
# SLURM sweep (one job per --step-size, so they run in parallel). Each job runs
# scripts/self_distillation_losses.py --methods mech, which contains the three
# arms: constant a-priori, constant SD+anchor, and the decay+avg oracle.
#
# This is a thin loop over scripts/submit_sd_losses.sh -- it forwards the same
# submission env knobs (GPU_TYPE, TIME, CPUS, PARTITION, MEM, GPUS, DRY_RUN) to
# every per-step submission, and gives each its own output dir / job name.
#
# Usage (run on the login node, e.g. kopsgridlogin1, NOT inside a grid-tunnel):
#   GPU_TYPE=h100 TIME=04:00:00 bash scripts/submit_mech_sweep.sh
#
#   # see every sbatch command without submitting:
#   DRY_RUN=1 bash scripts/submit_mech_sweep.sh
#
# Override the sweep shape via env vars (these are exclusive to this wrapper):
#   STEPS        space-separated --step-size grid   (default "0.01 0.03 0.1 0.3 1.0")
#   OUT_PREFIX   per-run --out-name prefix          (default sd_mech)
# All submission knobs (GPU_TYPE, TIME, ...) are read by submit_sd_losses.sh.
#
# Every study flag is a baked, overridable default (see DEFAULT_ARGS below). Any
# trailing CLI flags are forwarded verbatim to self_distillation_losses.py,
# appended AFTER the baked defaults, so they OVERRIDE them (argparse takes the
# last value). So the trace length, decay-oracle schedule, ladder depth, FIR
# lengths, etc. are just passed as trailing args -- e.g. the nstep ladder
# head-to-head, deepened and shortened:
#   bash scripts/submit_mech_sweep.sh --methods nstep --sd-horizon 4 -L 10000
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMIT="$PROJECT_ROOT/scripts/submit_sd_losses.sh"

# Trailing CLI flags (e.g. --methods nstep -L 10000) are forwarded to each
# per-step submission, AFTER the baked defaults below, so they override them.
EXTRA_ARGS=("$@")

# Exclusive-to-this-wrapper knobs: the per-job --step-size axis and the prefix
# used to derive a distinct --out-name for each parallel job.
STEPS="${STEPS:-0.01 0.03 0.1 0.3 1.0}"
OUT_PREFIX="${OUT_PREFIX:-sd_mech}"

# Baked study config (all overridable by trailing args). --methods mech is the
# study identity; --step-decay/--polyak-burnin drive the decay+avg oracle arm.
DEFAULT_ARGS=(
  --methods mech --s-d 6 --o-d 2 --window 4 --eps 0.1
  -N 16 -L 30000 --weight-decay 0 --device cuda
  --step-decay 0.51 --polyak-burnin 0.5 --sd-horizon 1
  --fir-lengths 1,2,3,4,5,6,7,8 --online-fir-lengths 2,4,8,16
  --analytic-stride 100 --seed 0
)

if [ ! -f "$SUBMIT" ]; then
  echo "ERROR: $SUBMIT not found" >&2
  exit 1
fi

echo "============================================================"
echo "mech sweep: STEPS = [$STEPS]"
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
