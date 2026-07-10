#!/usr/bin/env bash
# Submit the A-init pathway SD sweep as a SHARDED SLURM sweep: ONE job per
# (--step-size, init-arm) cell -- i.e. STEPS x ARMS jobs, each running
# scripts/self_distillation_losses.py --methods init with a SINGLE-value
# --init-grid (one named init arm) on the settled SD+anchor base.
#
# Why sharded (one arm per job)? Same reason as submit_post_sweep.sh: one arm per
# job keeps each cell cheap and lets the whole grid run in parallel, and the
# checkpoint is only written after a run's arms all finish, so packing arms into
# one job only delays the readout. Each shard writes its own checkpoint at
# output/<prefix>/ss<step>/<arm>/<tag>_save.pt; analyze_mech_sweep.py and
# plot_init_convergence.py glob these recursively and merge them into the usual
# per-(step, arm) grid.
#
# The study is the A-init pathway toy: K = H^+ is the exact toy realization of
# "K = A" (at F=0 the innovation is the raw observation, so the corrector is the
# REPLACE / high-gain limit). Each arm initializes at (or near) that high-gain end
# vs the current low-gain (K=0) init growing up. Every arm shares the settled base
# -- latent SD + light a-priori anchor (alpha=1, beta0=0.05), the DETACHED launch
# (--no-keep-launch), the single-step ladder (n=1, --sd-horizon default 1),
# constant gain, adapting (F, H, K). The only independent variables are the init
# pair (f_init, k_init) and (for two arms) the a-posteriori weight beta2. The
# 2x2 init grid plus two beta2-substitution arms are the six named tokens below.
#
# This is a thin loop over scripts/submit_sd_losses.sh -- it forwards the same
# submission env knobs (GPU_TYPE, TIME, CPUS, PARTITION, MEM, GPUS, DRY_RUN) to
# every per-cell submission, and gives each its own output dir / job name.
#
# Usage (run on the login node, e.g. kopsgridlogin1, NOT inside a grid-tunnel):
#   GPU_TYPE=h100 TIME=04:00:00 bash scripts/submit_init_sweep.sh
#
#   # see every sbatch command without submitting:
#   DRY_RUN=1 bash scripts/submit_init_sweep.sh
#
# Override the sweep shape via env vars (these are exclusive to this wrapper):
#   STEPS        space-separated --step-size grid   (default "0.01 0.03 0.1 0.3 1.0")
#   ARMS         space-separated init-arm grid (one job each)
#                       (default "fI_k0 fI_kpinv f0_kpinv f0_k0 fI_k0_b0p05 f0_kpinv_b0p05")
#   OUT_PREFIX   per-run --out-name prefix          (default sd_init_L300k)
# All submission knobs (GPU_TYPE, TIME, ...) are read by submit_sd_losses.sh.
#
# Every study flag is a baked, overridable default (see DEFAULT_ARGS below). Any
# trailing CLI flags are forwarded verbatim to self_distillation_losses.py,
# appended AFTER the baked defaults, so they OVERRIDE them (argparse takes the
# last value). So a shorter horizon is just a trailing arg:
#   bash scripts/submit_init_sweep.sh -L 100000
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMIT="$PROJECT_ROOT/scripts/submit_sd_losses.sh"

# Trailing CLI flags (e.g. -L 100000) are forwarded to each per-cell submission,
# AFTER the baked defaults below, so they override them.
EXTRA_ARGS=("$@")

# Exclusive-to-this-wrapper knobs: the two sweep axes (each (step, arm) cell is
# its own job) and the prefix used to derive a distinct --out-name per job.
STEPS="${STEPS:-0.01 0.03 0.1 0.3 1.0}"
ARMS="${ARMS:-fI_k0 fI_kpinv f0_kpinv f0_k0 fI_k0_b0p05 f0_kpinv_b0p05}"
OUT_PREFIX="${OUT_PREFIX:-sd_init_L300k}"

# Baked study config (all overridable by trailing args). --methods init is the
# study identity; --no-keep-launch fixes the detached launch and the single-step
# ladder (n=1) is the argparse default that the preset also bakes per arm.
# --init-grid is set per job (a single arm) in the loop below, so it is NOT baked
# here.
DEFAULT_ARGS=(
  --methods init --s-d 6 --o-d 2 --window 4 --eps 0.1
  -N 16 -L 300000 --weight-decay 0 --anchor 0.05 --no-keep-launch --device cuda
  --fir-lengths 1,2,3,4,5,6,7,8 --online-fir-lengths 2,4,8,16
  --analytic-stride 100 --seed 0
)

if [ ! -f "$SUBMIT" ]; then
  echo "ERROR: $SUBMIT not found" >&2
  exit 1
fi

echo "============================================================"
echo "SD A-init pathway sweep (sharded, 1 arm/job)"
echo "  STEPS = [$STEPS]"
echo "  ARMS  = [$ARMS]"
echo "  out-prefix=$OUT_PREFIX  defaults: ${DEFAULT_ARGS[*]}"
[ "${#EXTRA_ARGS[@]}" -gt 0 ] && echo "  extra (override) args: ${EXTRA_ARGS[*]}"
echo "============================================================"

njobs=0
for ss in $STEPS; do
  # filesystem-safe tag: 0.01 -> 0p01, 0.1 -> 0p1, 1.0 -> 1p0
  stag="${ss//./p}"
  for arm in $ARMS; do
    out_name="${OUT_PREFIX}/ss${stag}/${arm}"
    echo
    echo ">>> submitting step-size=$ss  arm=$arm  ->  out-name=$out_name"
    bash "$SUBMIT" "${DEFAULT_ARGS[@]}" \
      --step-size "$ss" --init-grid "$arm" --out-name "$out_name" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    njobs=$((njobs + 1))
  done
done

echo
echo "all $njobs per-cell jobs submitted (or printed under DRY_RUN)."
echo "outputs land in $PROJECT_ROOT/output/${OUT_PREFIX}/ss<step>/<arm>/"
echo "read out with:  python scripts/analyze_mech_sweep.py     --prefix ${OUT_PREFIX}"
echo "                python scripts/plot_init_convergence.py  --prefix ${OUT_PREFIX}"
echo "                python scripts/plot_init_gain_migration.py --prefix ${OUT_PREFIX}"
