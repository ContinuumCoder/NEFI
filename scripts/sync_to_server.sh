#!/usr/bin/env bash
# Sync the code to a GPU server, or pull small result files back (keeps large tensors on the server).
#
#   HOST=gpu-server scripts/sync_to_server.sh                 # push code to $HOST:$REMOTE_DIR
#   HOST=gpu-server scripts/sync_to_server.sh --dry-run       # show what would be sent
#   HOST=gpu-server DELETE=1 scripts/sync_to_server.sh        # also delete remote files removed locally
#   HOST=gpu-server PULL=1 scripts/sync_to_server.sh          # fetch runs/ reports (md/json/csv/yaml/png/log,
#                                                        # no tensors) into runs/remote-$HOST/
#
# Variables: HOST (required, an ssh host alias), REMOTE_DIR (default: ~/nefi).
# Never transferred when pushing: runs/, outputs/, .git/, caches, virtualenvs, build artefacts,
# tensors (*.pt, *.pth, *.npz, *.npy, *.h5).
set -euo pipefail

HOST="${HOST:?set HOST, e.g. HOST=gpu-server scripts/sync_to_server.sh}"
REMOTE_DIR="${REMOTE_DIR:-~/nefi}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${PULL:-0}" == "1" ]]; then
  DEST="runs/remote-${HOST}/"
  mkdir -p "$DEST"
  rsync -az --prune-empty-dirs \
    --include='*/' \
    --include='*.md' --include='*.json' --include='*.csv' --include='*.yaml' \
    --include='*.png' --include='*.log' --include='*.out' \
    --exclude='*' \
    "$@" "${HOST}:${REMOTE_DIR}/runs/" "$DEST"
  echo "[sync] pulled reports from ${HOST}:${REMOTE_DIR}/runs/ into $DEST"
  exit 0
fi

EXCLUDES=(
  --exclude='runs/' --exclude='outputs/' --exclude='.git/'
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/'
  --exclude='.ruff_cache/' --exclude='.mypy_cache/' --exclude='*.egg-info/'
  --exclude='.venv/' --exclude='venv/' --exclude='env/' --exclude='site/'
  --exclude='docs/_build/' --exclude='build/' --exclude='dist/' --exclude='.DS_Store'
  --exclude='*.pt' --exclude='*.pth' --exclude='*.npz' --exclude='*.npy' --exclude='*.h5'
  --exclude='.ipynb_checkpoints/'
)
DEL=()
if [[ "${DELETE:-0}" == "1" ]]; then
  DEL=(--delete)
fi
ssh "$HOST" "mkdir -p ${REMOTE_DIR}"
rsync -az ${DEL[@]+"${DEL[@]}"} "${EXCLUDES[@]}" "$@" ./ "${HOST}:${REMOTE_DIR}/"
echo "[sync] pushed $(pwd) -> ${HOST}:${REMOTE_DIR}"
echo "[sync] next: ssh ${HOST} 'cd ${REMOTE_DIR} && scripts/run_server.sh configs/toy1d.yaml'"
