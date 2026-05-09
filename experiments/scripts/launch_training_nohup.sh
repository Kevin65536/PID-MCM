#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

print_help() {
    cat <<'EOF'
Usage:
  bash experiments/scripts/launch_training_nohup.sh --task TASK [--foreground] [task args]

Supported tasks:
  source-observation-tokenizer
    Script: experiments/scripts/train_source_observation_tokenizer.py
    Args: --config PATH [--resume PATH] [--run-name NAME] [--skip-post-analysis]

  tokenizer
    Script: experiments/scripts/train_tokenizer.py
    Args: --config PATH [--resume PATH]

  downstream
    Script: experiments/scripts/train_downstream.py
    Args: --config PATH

  foundation-interface
    Script: experiments/scripts/train_foundation_interface.py
    Args: [--config PATH]

Examples:
  bash experiments/scripts/launch_training_nohup.sh --task source-observation-tokenizer --config debug/simultaneous_nback_short_train.yaml --run-name smoke_run
  bash experiments/scripts/launch_training_nohup.sh --task tokenizer --config phase0plus/eeg_labram_vqnsp.yaml
  bash experiments/scripts/launch_training_nohup.sh --task downstream --config downstream/P1A_eeg_classification.yaml

Notes:
  - This is the only repository-supported training launcher.
  - Task-specific arguments are passed through directly to the selected training script.
  - Use --foreground only for short interactive debugging runs.
EOF
}

resolve_script() {
    case "$1" in
        source-observation-tokenizer)
            echo "experiments/scripts/train_source_observation_tokenizer.py"
            ;;
        tokenizer)
            echo "experiments/scripts/train_tokenizer.py"
            ;;
        downstream)
            echo "experiments/scripts/train_downstream.py"
            ;;
        foundation-interface)
            echo "experiments/scripts/train_foundation_interface.py"
            ;;
        *)
            echo "Unknown task: $1" >&2
            print_help >&2
            exit 1
            ;;
    esac
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    exit 1
fi

TASK=""
RUN_MODE="nohup"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --task" >&2
                exit 1
            fi
            TASK="$2"
            shift 2
            ;;
        --foreground)
            RUN_MODE="foreground"
            shift
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

if [[ -z "${TASK}" ]]; then
    echo "--task is required" >&2
    print_help >&2
    exit 1
fi

TRAIN_SCRIPT="$(resolve_script "${TASK}")"

cd "${ROOT_DIR}"
CMD=("${PYTHON_BIN}" "${TRAIN_SCRIPT}" "$@")

if [[ "${RUN_MODE}" == "foreground" ]]; then
    export NEURAL_TOKEN_TRAIN_LAUNCHER=1
    exec "${CMD[@]}"
fi

NEURAL_TOKEN_TRAIN_LAUNCHER=1 nohup "${CMD[@]}" > /dev/null 2>&1 < /dev/null &
PID=$!

echo "PID: ${PID}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
