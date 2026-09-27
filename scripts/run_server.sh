#!/usr/bin/env bash
# Run a nefi job on a CUDA server (e.g. a CUDA server): environment setup + `nefi run`.
#
# Usage (from anywhere inside the repository clone on the server):
#   scripts/run_server.sh configs/nv_relaxometry_paper.yaml
#   scripts/run_server.sh thermal_tomography --scene layered --seed 1
#   NEFI_CMD=bench scripts/run_server.sh configs/toy1d.yaml --methods neural,grid --n 8 --seeds 0,1,2
#   CUDA_VISIBLE_DEVICES=1 SKIP_INSTALL=1 scripts/run_server.sh deconvolution --plot
#
# Environment variables:
#   ENV_NAME       conda environment to activate (default: nefi; created from environment.yml if
#                  missing). Set VENV=/path/to/venv to use a virtualenv instead of conda.
#   NEFI_CMD       nefi sub-command: run (default) | bench | diagnose | ablate | sweep
#   DEVICE         device passed to nefi (default: cuda)
#   OUT_ROOT       output root (default: runs); the job writes to $OUT_ROOT/<target>-<timestamp>
#   SKIP_INSTALL=1 skip `pip install -e ".[dev]"` (fast re-runs)
# Everything after the target is forwarded to nefi (e.g. --set stage.lr=5e-4 --plot).
set -euo pipefail

TARGET="${1:?usage: scripts/run_server.sh <instance|config.yaml> [nefi options...]}"
shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${VENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  ENV_NAME="${ENV_NAME:-nefi}"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[run_server] creating conda env $ENV_NAME from environment.yml"
    conda env create -n "$ENV_NAME" -f environment.yml
  fi
  conda activate "$ENV_NAME"
else
  echo "[run_server] no conda and no VENV set: using $(command -v python)" >&2
fi

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install --quiet -e ".[dev]"
fi

python - <<'PY'
import torch
print(f"[run_server] torch {torch.__version__}, cuda available: {torch.cuda.is_available()}",
      f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")
PY

CMD="${NEFI_CMD:-run}"
NAME="$(basename "${TARGET%.*}")"
OUT="${OUT_ROOT:-runs}/${NAME}-${CMD}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$(dirname "$OUT")"
export PYTHONUNBUFFERED=1
echo "[run_server] nefi $CMD $TARGET --device ${DEVICE:-cuda} --out $OUT $*"
nefi "$CMD" "$TARGET" --device "${DEVICE:-cuda}" --out "$OUT" "$@" 2>&1 | tee "$OUT.log"
echo "[run_server] done: $OUT (log: $OUT.log)"
