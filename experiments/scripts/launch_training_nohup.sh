#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
DEFAULT_SCRIPT="experiments/scripts/train_shared_tokenizer.py"
LOG_DIR="${ROOT_DIR}/experiments/nohup_logs"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    exit 1
fi

TRAIN_SCRIPT="${DEFAULT_SCRIPT}"
if [[ $# -gt 0 && "$1" != --* ]]; then
    TRAIN_SCRIPT="$1"
    shift
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT_NAME="$(basename "${TRAIN_SCRIPT}" .py)"
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}_${STAMP}.log"

cd "${ROOT_DIR}"
CMD=("${PYTHON_BIN}" "${TRAIN_SCRIPT}" "$@")

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!

echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
