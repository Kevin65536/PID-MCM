#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
DEFAULT_SCRIPT="experiments/scripts/train_shared_tokenizer.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    exit 1
fi

TRAIN_SCRIPT="${DEFAULT_SCRIPT}"
if [[ $# -gt 0 && "$1" != --* ]]; then
    TRAIN_SCRIPT="$1"
    shift
fi

cd "${ROOT_DIR}"
CMD=("${PYTHON_BIN}" "${TRAIN_SCRIPT}" "$@")

nohup "${CMD[@]}" > /dev/null 2>&1 < /dev/null &
PID=$!

echo "PID: ${PID}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
