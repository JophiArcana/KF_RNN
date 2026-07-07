#!/usr/bin/env bash
# Submit the *best minimal-method configuration* (the knob-only vanishing-gain
# filter from docs/minimal_convergent_filter_findings.md) at a long context so we
# can verify the excess-over-floor (the "regret") keeps decreasing past L=10000.
#
# This is a thin, opinionated preset over scripts/self_distillation_losses.py: the
# study args below are the headline config (M4 decay+avg, a=0.51, Polyak n0=0.5,
# base step eta_0=1.0, window=4, no projection / warm start / RTRL). The horizon
# -L and the --out-name are NOT baked -- pass them yourself. The meta-configuration
# (GPU / TIME / CPUs) is stated explicitly at the top; every extra CLI arg you pass
# is forwarded verbatim and, since argparse takes the last value, OVERRIDES the
# baked defaults. E.g.:
#
#   # the L=30000 verification run, on the login node (kopsgridlogin1):
#   bash scripts/submit_minimal.sh -L 30000 --out-name sd_minimal_L30000
#
#   # tweak a study knob too (forwarded verbatim; overrides the baked default):
#   bash scripts/submit_minimal.sh -L 30000 --out-name sd_min --polyak-burnin 0.8
#
#   # pin a GPU model and a tighter wall clock:
#   GPU_TYPE=l40s TIME=02:00:00 bash scripts/submit_minimal.sh -L 30000 --out-name sd_min
#
#   # see the exact sbatch command without submitting:
#   DRY_RUN=1 bash scripts/submit_minimal.sh -L 30000 --out-name sd_min
#
# The SLURM job name and log location are derived from --out-name, so a run's
# stdout sits next to its plots under <project>/output/<out-name>/.
#
# ===== META-CONFIGURATION (env-overridable) ================================
#   GPUS       GPUs for the job          (default 1)
#   GPU_TYPE   GPU model to require      (default: any; e.g. l40s, h100)
#   CPUS       --cpus-per-task           (default 8)
#   PARTITION  SLURM partition           (default kop-gpus)
#   TIME       --time wall clock limit   (default 04:00:00)
#   MEM        --mem                     (default: partition default, unset)
#   JOB_NAME   SLURM job name            (default: sd-<out-name>)
#   DRY_RUN    1 => print sbatch cmd only, do not submit
#
# The online horizon (-L) and --out-name are passed through to the study, not set
# here. Set --out-name so the job name / log dir are meaningful (it defaults to the
# study's own "sd_losses" otherwise).
#
# Note on the device: this study is a per-step Python recursion over a tiny
# (S_D=6) system, which runs *faster on CPU* than GPU (no per-step kernel-launch
# overhead) -- so the baked default is --device cpu, using the allocated CPUS. We
# still request a GPU because kop-gpus is a GPU partition; pass --device cuda
# (forwarded, overrides the default) to use it instead.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="$PROJECT_ROOT/.conda"
STUDY_PY="$PROJECT_ROOT/scripts/self_distillation_losses.py"

# --- meta-configuration -----------------------------------------------------
GPUS="${GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-}"
CPUS="${CPUS:-8}"
PARTITION="${PARTITION:-kop-gpus}"
TIME="${TIME:-04:00:00}"
MEM="${MEM:-}"
DRY_RUN="${DRY_RUN:-0}"

# --- the best minimal config (baked defaults; forwarded args override) ------
# Headline config from docs/minimal_convergent_filter_findings.md. The horizon -L
# and --out-name are intentionally NOT baked -- pass them on the command line.
# --no-fir skips the (pre-existing, numerically brittle) optimal-FIR Newton init;
# the irreducible floor line -- the comparator for "is the regret still
# decreasing" -- is drawn regardless. A coarser --analytic-stride keeps the
# long-horizon analytical-error snapshots cheap.
DEFAULT_ARGS=(
  --methods minimal
  --step-decay 0.51
  --polyak-burnin 0.5
  --step-size 1.0
  --s-d 6 --o-d 2 --eps 0.1
  --window 4
  -N 16
  --analytic-stride 100
  --device cpu
  --no-fir
)
FORWARD=("${DEFAULT_ARGS[@]}" "$@")

# --- peek at the effective --out-name (last occurrence wins, like argparse) -
OUT_NAME="sd_losses"
n="${#FORWARD[@]}"
i=0
while [ "$i" -lt "$n" ]; do
  arg="${FORWARD[$i]}"
  case "$arg" in
    --out-name) OUT_NAME="${FORWARD[$((i + 1))]}" ;;
    --out-name=*) OUT_NAME="${arg#*=}" ;;
  esac
  i=$((i + 1))
done
JOB_NAME="${JOB_NAME:-sd-${OUT_NAME//\//_}}"

# --- sanity checks ----------------------------------------------------------
# (skipped for DRY_RUN so the assembled command can be previewed off the login node)
if [ "$DRY_RUN" != "1" ] && ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on the login node (kopsgridlogin1)," >&2
  echo "       not inside a grid-tunnel compute shell." >&2
  exit 1
fi
if [ ! -d "$ENV_PREFIX" ]; then
  echo "ERROR: conda env not found at $ENV_PREFIX (create it per the repo README)." >&2
  exit 1
fi

# --- per-run dir (must exist before sbatch resolves --output) --------------
LOG_DIR="$PROJECT_ROOT/output/$OUT_NAME"
mkdir -p "$LOG_DIR"

# --- build the in-job command ----------------------------------------------
# `conda run -p <env>` activates the env without needing `conda init` in the
# batch shell. The study is single-process, so no torchrun.
JOB_CMD=(conda run --no-capture-output -p "$ENV_PREFIX" python "$STUDY_PY" "${FORWARD[@]}")
WRAP_CMD="$(printf '%q ' "${JOB_CMD[@]}")"

# --- assemble sbatch invocation --------------------------------------------
# A typed request (--gpus-per-task=nvidia_l40s:1) pins the GPU model; bare "1"
# lets the scheduler pick any GPU on the partition. SLURM matches the GRES *type*
# exactly (the full driver string, e.g. nvidia_l40s), so accept the friendly
# short names and translate them here. List a partition's real types with:
#   sinfo -p "$PARTITION" -N -o '%n %G'
case "$GPU_TYPE" in
  l40s) GPU_TYPE="nvidia_l40s" ;;
  h100) GPU_TYPE="nvidia_h100_80gb_hbm3" ;;
esac
GPU_SPEC="$GPUS"
[ -n "$GPU_TYPE" ] && GPU_SPEC="$GPU_TYPE:$GPUS"
SBATCH_ARGS=(
  --parsable
  --job-name="$JOB_NAME"
  --partition="$PARTITION"
  --nodes=1
  --ntasks-per-node=1
  --gpus-per-task="$GPU_SPEC"
  --cpus-per-task="$CPUS"
  --chdir="$PROJECT_ROOT"
  --output="$LOG_DIR/slurm-%j.out"
)
[ -n "$TIME" ] && SBATCH_ARGS+=(--time="$TIME")
[ -n "$MEM" ] && SBATCH_ARGS+=(--mem="$MEM")

echo "============================================================"
echo "submit: $JOB_NAME"
echo "  partition : $PARTITION"
echo "  gpus      : $GPU_SPEC   cpus/task: $CPUS   time: ${TIME:-<partition default>}"
echo "  out-name  : $OUT_NAME"
echo "  log       : $LOG_DIR/slurm-<jobid>.out"
echo "  command   : $WRAP_CMD"
echo "============================================================"

if [ "$DRY_RUN" = "1" ]; then
  printf 'DRY_RUN: sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}" --wrap="$WRAP_CMD"
  printf '\n'
  exit 0
fi

JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" --wrap="$WRAP_CMD")"
echo "submitted job $JOB_ID"
echo "  tail logs : tail -F $LOG_DIR/slurm-$JOB_ID.out"
echo "  status    : squeue --me"
echo "  cancel    : scancel $JOB_ID"
