#!/bin/bash
# SLURM template: one nefi job on one GPU.
#
#   mkdir -p runs/slurm
#   sbatch --export=ALL,TARGET=configs/nv_relaxometry_paper.yaml scripts/slurm_template.sbatch
#   # 3 seeds as an array job (seed = array index):
#   sbatch --array=0-2 --export=ALL,TARGET=thermal_tomography,NEFI_ARGS="--scene layered" \
#          scripts/slurm_template.sbatch
#   # values containing commas (e.g. --seeds 0,1,2) cannot go inside --export=...: export them
#   # in the submitting shell and pass --export=ALL. A benchmark instead of a single run:
#   export NEFI_CMD=bench TARGET=configs/toy1d.yaml NEFI_ARGS="--n 16 --seeds 0,1,2"
#   sbatch --export=ALL scripts/slurm_template.sbatch
#   # a sharded benchmark: 8 array tasks, task i runs shard i/8 (batched solving where possible)
#   # into ONE directory; a dependent job then merges the shards into the final table:
#   export NEFI_CMD=bench NEFI_SHARDS=8 TARGET=configs/deconvolution_full.yaml
#   export NEFI_ARGS="--n 32 --seeds 0,1,2 --batched"
#   JOB=$(sbatch --parsable --array=0-7 --export=ALL scripts/slurm_template.sbatch)
#   DIR=runs/slurm/deconvolution_full-bench-${JOB}      # <target basename>-bench-<array job id>
#   sbatch --dependency=afterok:${JOB} --export=ALL,NEFI_CMD=bench-merge,TARGET=${DIR} \
#          scripts/slurm_template.sbatch
#
# Adjust partition / account / time to your cluster. Paper-scale expectations (single GPU):
# NeTMY 64² × 10k steps ≈ 5-13 min per measurement (A6000); NeFTY 64×64×16, 100 frames,
# 10k steps ≈ 10 min and ≈ 4 GB peak memory with the discrete adjoint (RTX PRO 6000).
# See docs/performance.md for the throughput switches (--batched, compile, autocast).
#SBATCH --job-name=nefi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x-%A_%a.out
##SBATCH --partition=gpu
##SBATCH --account=cab

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- environment (site-specific: edit the module line if your cluster needs one) -------------
module load anaconda3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME:-nefi}"
fi
export NEFI_DEVICE=cuda
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

# --- job ----------------------------------------------------------------------------------
TARGET="${TARGET:-configs/toy1d.yaml}"
CMD="${NEFI_CMD:-run}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
python -c "import torch, nefi; print('torch', torch.__version__, 'nefi', nefi.__version__)"
if [[ "$CMD" == "bench-merge" ]]; then
  # TARGET = the directory holding shard-*-of-*.json (writes summary.md / csv / json next to them)
  nefi bench-merge "$TARGET"
  exit 0
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
EXTRA=()
if [[ "$CMD" == "bench" && -n "${NEFI_SHARDS:-}" ]]; then
  # every array task writes its shard into the same directory (keyed by the array job id)
  OUT="runs/slurm/$(basename "${TARGET%.*}")-bench-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
  EXTRA=(--shard "${SLURM_ARRAY_TASK_ID:-0}/${NEFI_SHARDS}")
else
  OUT="runs/slurm/$(basename "${TARGET%.*}")-${CMD}-${SLURM_JOB_ID:-local}-s${SEED}"
  if [[ "$CMD" == "run" || "$CMD" == "diagnose" ]]; then
    EXTRA=(--seed "$SEED")
  fi
fi
# shellcheck disable=SC2086
nefi "$CMD" "$TARGET" --device cuda ${EXTRA[@]+"${EXTRA[@]}"} --out "$OUT" ${NEFI_ARGS:-}
