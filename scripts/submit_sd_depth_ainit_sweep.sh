#!/usr/bin/env bash
# Submit the A-INIT version of the n-step self-distillation DEPTH x LAUNCH study
# as a PER-CELL SLURM sweep: one job per (launch-variant, --step-size), so they
# run in parallel. Each job runs scripts/self_distillation_losses.py --methods
# nstep on the A-init pathway's f0_kpinv (F=0, K=H^+, via --f-init zero --k-init
# pinv), whose arms are the pure a-priori control plus one SD+anchor arm per
# ladder depth n = 1..WINDOW -- so a single job already sweeps the SD depth, and
# the loops below add the step-size and launch-gradient axes.
#
# This is the A-init counterpart of submit_sd_depth_sweep.sh (which runs the same
# depth x launch grid on the current default init). The prior studies settled:
#   - the SD base: latent SD + light a-priori anchor (alpha=1, beta0=0.05),
#     constant gain, adapting (F, H, K)  [mechanism_constant_gain_sd_findings.md]
#   - the depth x launch verdict at the DEFAULT init: detach wins the converged
#     floor + stability, n=1 wins the floor  [nstep_sd_ladder_launch_findings.md]
#   - the A-init f0_kpinv (F=0, K=H^+) converges faster, is 100% stable, and
#     lands in a contractive basin at the SAME floor  [init_pathway_findings.md]
# This sweep asks whether the depth x launch verdict survives on the A-init base.
# Every arm (incl. the M4 control) is put on the A-init, for an apples-to-apples
# grid at the new init; compare to the default-init numbers in the prior doc.
#
# Two launch variants (both run by default):
#   keep    -> --keep-launch     (default; non-root launch carries gradient)
#             out prefix  ${OUT_PREFIX}
#   detach  -> --no-keep-launch  (non-root launch detached)
#             out prefix  ${OUT_PREFIX}_detach
# A single L=300000 run per cell yields the full convergence curve (err_curve vs
# steps_axis), so the convergence-rate plots need no separate short-horizon sweep.
#
# This is a thin loop over scripts/submit_sd_losses.sh -- it forwards the same
# submission env knobs (GPU_TYPE, TIME, CPUS, PARTITION, MEM, GPUS, DRY_RUN) to
# every per-cell submission, and gives each its own output dir / job name.
#
# Usage (run on the login node, e.g. kopsgridlogin1, NOT inside a grid-tunnel):
#   GPU_TYPE=h100 TIME=06:00:00 bash scripts/submit_sd_depth_ainit_sweep.sh
#
#   # see every sbatch command without submitting:
#   DRY_RUN=1 bash scripts/submit_sd_depth_ainit_sweep.sh
#
# Override the sweep shape via env vars (these are exclusive to this wrapper):
#   STEPS        space-separated --step-size grid   (default "0.01 0.03 0.1 0.3 1.0")
#   LAUNCHES     space-separated launch variants     (default "keep detach")
#   OUT_PREFIX   per-run --out-name prefix           (default sd_depth_ainit_L300k)
# All submission knobs (GPU_TYPE, TIME, ...) are read by submit_sd_losses.sh.
#
# Every study flag is a baked, overridable default (see DEFAULT_ARGS below). Any
# trailing CLI flags are forwarded verbatim to self_distillation_losses.py,
# appended AFTER the baked defaults, so they OVERRIDE them (argparse takes the
# last value). So a longer horizon / different depth ceiling is just a trailing
# arg, e.g. escalate eta=0.01 to 1M:  bash scripts/submit_sd_depth_ainit_sweep.sh -L 1000000
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMIT="$PROJECT_ROOT/scripts/submit_sd_losses.sh"

# Trailing CLI flags (e.g. -L 1000000) are forwarded to each per-cell submission,
# AFTER the baked defaults below, so they override them.
EXTRA_ARGS=("$@")

# Exclusive-to-this-wrapper knobs: the two sweep axes (each (launch, step) cell
# is its own job) and the prefix used to derive a distinct --out-name per job.
STEPS="${STEPS:-0.01 0.03 0.1 0.3 1.0}"
LAUNCHES="${LAUNCHES:-keep detach}"
OUT_PREFIX="${OUT_PREFIX:-sd_depth_ainit_L300k}"

# Baked study config (all overridable by trailing args). --methods nstep is the
# study identity; --window is the SD depth ceiling (ladder depths n=1..window);
# --f-init zero --k-init pinv puts the whole grid on the A-init f0_kpinv. The
# launch flag is added per variant in the loop below (NOT baked here).
DEFAULT_ARGS=(
  --methods nstep --s-d 6 --o-d 2 --window 4 --eps 0.1
  -N 16 -L 300000 --weight-decay 0 --anchor 0.05
  --f-init zero --k-init pinv --device cuda
  --fir-lengths 1,2,3,4,5,6,7,8 --online-fir-lengths 2,4,8,16
  --analytic-stride 100 --seed 0
)

if [ ! -f "$SUBMIT" ]; then
  echo "ERROR: $SUBMIT not found" >&2
  exit 1
fi

echo "============================================================"
echo "SD A-init depth x launch sweep (sharded, 1 (launch,step)/job)"
echo "  STEPS    = [$STEPS]"
echo "  LAUNCHES = [$LAUNCHES]"
echo "  out-prefix=$OUT_PREFIX  defaults: ${DEFAULT_ARGS[*]}"
[ "${#EXTRA_ARGS[@]}" -gt 0 ] && echo "  extra (override) args: ${EXTRA_ARGS[*]}"
echo "============================================================"

njobs=0
for launch in $LAUNCHES; do
  case "$launch" in
    keep)   launch_flag="--keep-launch";    prefix="$OUT_PREFIX" ;;
    detach) launch_flag="--no-keep-launch"; prefix="${OUT_PREFIX/_L/_detach_L}" ;;
    *) echo "ERROR: unknown launch variant '$launch' (want keep|detach)" >&2; exit 1 ;;
  esac
  # If OUT_PREFIX has no _L token, fall back to a plain _detach suffix.
  [ "$launch" = "detach" ] && [ "$prefix" = "$OUT_PREFIX" ] && prefix="${OUT_PREFIX}_detach"
  for ss in $STEPS; do
    # filesystem-safe tag: 0.01 -> 0p01, 0.1 -> 0p1, 1.0 -> 1p0
    stag="${ss//./p}"
    out_name="${prefix}/ss${stag}"
    echo
    echo ">>> submitting launch=$launch  step-size=$ss  ->  out-name=$out_name"
    bash "$SUBMIT" "${DEFAULT_ARGS[@]}" "$launch_flag" \
      --step-size "$ss" --out-name "$out_name" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    njobs=$((njobs + 1))
  done
done

echo
echo "all $njobs per-cell jobs submitted (or printed under DRY_RUN)."
echo "outputs land in $PROJECT_ROOT/output/${OUT_PREFIX}{,_detach}/ss<step>/"
echo "read out with:  python scripts/analyze_mech_sweep.py    --prefix ${OUT_PREFIX}"
echo "                python scripts/analyze_mech_sweep.py    --prefix ${OUT_PREFIX/_L/_detach_L}"
echo "                python scripts/plot_sd_depth_sweep.py   --prefix ${OUT_PREFIX}"
echo "                python scripts/plot_sd_launch_convergence.py --keep-prefix ${OUT_PREFIX} --detach-prefix ${OUT_PREFIX/_L/_detach_L} --depths 1,2,3,4"
echo "                python scripts/plot_sd_depth_convergence.py  --prefixes ${OUT_PREFIX},${OUT_PREFIX/_L/_detach_L} --step 0.03 --depths 1,2,3,4"
