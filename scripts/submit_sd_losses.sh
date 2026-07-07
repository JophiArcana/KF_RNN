#!/usr/bin/env bash
# Submit scripts/self_distillation_losses.py as a SLURM batch job from the login
# node (kopsgridlogin1). This is the non-interactive counterpart to running the
# study inside a `grid-tunnel` shell: `grid-tunnel` itself just wraps
# `sbatch ... --wrap=<sshd tunnel>`, so here we `sbatch ... --wrap=<study>` with
# the same resource flags (--partition=kop-gpus, --gpus-per-task, --cpus...).
#
# The study is a single-process analysis (no DDP / torchrun), so this wrapper is
# simpler than submit_train_autoencoder.sh: it asks for one GPU and runs the
# script once. Every CLI arg is forwarded verbatim to self_distillation_losses.py
# (which selects the device from --device, default cuda-if-available).
#
# Usage (run on kopsgridlogin1, NOT inside a grid-tunnel):
#   # the m3-vs-m4 stability study at long context:
#   bash scripts/submit_sd_losses.sh \
#       --methods m3m4 --s-d 6 --o-d 2 --window 4 --eps 0.1 --step-size 0.03 \
#       -N 16 -L 2000 --weight-decay 0 --device cuda --out-name sd_m3m4_long
#
#   # pin a GPU model / ask for a wall-clock limit:
#   GPU_TYPE=l40s TIME=02:00:00 bash scripts/submit_sd_losses.sh \
#       --methods m3m4 -L 2000 --out-name sd_m3m4_long
#
#   # see the exact sbatch command without submitting:
#   DRY_RUN=1 bash scripts/submit_sd_losses.sh --methods m3m4 --out-name sd_m3m4_long
#
# Submission knobs are env vars (override inline as shown above); the SLURM job
# name and log location are both derived from the study's --out-name so a run's
# stdout sits next to its plots under <project>/output/<out-name>/.
#
#   GPUS       GPUs for the job            (default 1)
#   GPU_TYPE   GPU model to require        (default: any; e.g. l40s, h100)
#   CPUS       --cpus-per-task             (default 4)
#   PARTITION  SLURM partition             (default kop-gpus)
#   TIME       --time wall clock limit     (default: partition default, unset)
#   MEM        --mem                       (default: partition default, unset)
#   JOB_NAME   SLURM job name              (default: sd-<out-name>)
#   DRY_RUN    1 => print sbatch cmd only, do not submit
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="$PROJECT_ROOT/.conda"
STUDY_PY="$PROJECT_ROOT/scripts/self_distillation_losses.py"

# --- submission knobs (env-overridable) ------------------------------------
GPUS="${GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-}"
CPUS="${CPUS:-4}"
PARTITION="${PARTITION:-kop-gpus}"
TIME="${TIME:-}"
MEM="${MEM:-}"
DRY_RUN="${DRY_RUN:-0}"

# --- forward every CLI arg to the study, but peek at --out-name ------------
# --out-name decides the job name and the log directory; default matches the
# study's own argparse default ("sd_losses").
FORWARD=("$@")
OUT_NAME="sd_losses"
i=0
while [ "$i" -lt "$#" ]; do
  arg="${FORWARD[$i]}"
  case "$arg" in
    --out-name) OUT_NAME="${FORWARD[$((i + 1))]}" ;;
    --out-name=*) OUT_NAME="${arg#*=}" ;;
  esac
  i=$((i + 1))
done
JOB_NAME="${JOB_NAME:-sd-${OUT_NAME//\//_}}"

# --- sanity checks ----------------------------------------------------------
if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on the login node (kopsgridlogin1)," >&2
  echo "       not inside a grid-tunnel compute shell." >&2
  exit 1
fi
if [ ! -d "$ENV_PREFIX" ]; then
  echo "ERROR: conda env not found at $ENV_PREFIX (create it per the repo README)." >&2
  exit 1
fi

# --- per-run dir (must exist before sbatch resolves --output) --------------
# The study writes its plots to output/<out-name>/; put the SLURM stdout there too.
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
echo "  gpus      : $GPU_SPEC   cpus/task: $CPUS"
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
