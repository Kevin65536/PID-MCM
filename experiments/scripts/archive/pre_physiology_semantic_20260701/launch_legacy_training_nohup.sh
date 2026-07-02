#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

print_help() {
    cat <<'EOF'
Usage:
  bash experiments/scripts/archive/pre_physiology_semantic_20260701/launch_legacy_training_nohup.sh --task TASK [--foreground] [task args]

Supported tasks:
  source-observation-tokenizer
    Script: experiments/scripts/archive/pre_physiology_semantic_20260701/train_source_observation_tokenizer.py
    Args: --config PATH [--resume PATH] [--run-name NAME] [--skip-post-analysis]

  source-observation-coupling-calibration
    Script: experiments/scripts/archive/pre_physiology_semantic_20260701/calibrate_source_observation_coupling.py
    Args: --config PATH [--checkpoint PATH] [--run-name NAME] [--skip-post-analysis]

  tokenizer
    Script: experiments/scripts/archive/pre_physiology_semantic_20260701/train_tokenizer.py
    Args: --config PATH [--resume PATH]

  source-observation-token-export
    Script: experiments/scripts/archive/pre_physiology_semantic_20260701/export_source_observation_tokens.py
    Args: [--config PATH] [--tokenizer-run-dir PATH] [--checkpoint PATH] [--run-name NAME]

  foundation-interface
    Script: experiments/scripts/archive/pre_physiology_semantic_20260701/train_foundation_interface.py
    Args: [--config PATH]

  wholebrain-foundation
    Script: experiments/scripts/archive/pre_physiology_semantic_20260701/train_wholebrain_foundation.py
    Args: [--config PATH]

  wholebrain-pretrain
    Script: experiments/scripts/archive/pre_physiology_semantic_20260701/train_wholebrain_pretrain.py
    Args: [--config PATH] [--dry-run]

Examples:
  bash experiments/scripts/archive/pre_physiology_semantic_20260701/launch_legacy_training_nohup.sh --task source-observation-tokenizer --config experiments/configs/archive/pre_physiology_semantic_20260701/source_observation/phase1/default.yaml --run-name historical_rerun
  bash experiments/scripts/archive/pre_physiology_semantic_20260701/launch_legacy_training_nohup.sh --task tokenizer --config experiments/configs/archive/pre_physiology_semantic_20260701/phase0plus/eeg_neurorvq.yaml
  bash experiments/scripts/archive/pre_physiology_semantic_20260701/launch_legacy_training_nohup.sh --task source-observation-token-export --foreground --max-batches 1

Notes:
  - This launcher is compatibility-only and is not an active architecture entrypoint.
  - Task-specific arguments are passed through directly to the selected training script.
  - Use --foreground only for short interactive debugging runs.
EOF
}

resolve_script() {
    case "$1" in
        source-observation-tokenizer)
            echo "experiments/scripts/archive/pre_physiology_semantic_20260701/train_source_observation_tokenizer.py"
            ;;
        source-observation-coupling-calibration)
            echo "experiments/scripts/archive/pre_physiology_semantic_20260701/calibrate_source_observation_coupling.py"
            ;;
        tokenizer)
            echo "experiments/scripts/archive/pre_physiology_semantic_20260701/train_tokenizer.py"
            ;;
        source-observation-token-export)
            echo "experiments/scripts/archive/pre_physiology_semantic_20260701/export_source_observation_tokens.py"
            ;;
        foundation-interface)
            echo "experiments/scripts/archive/pre_physiology_semantic_20260701/train_foundation_interface.py"
            ;;
        wholebrain-foundation)
            echo "experiments/scripts/archive/pre_physiology_semantic_20260701/train_wholebrain_foundation.py"
            ;;
        wholebrain-pretrain)
            echo "experiments/scripts/archive/pre_physiology_semantic_20260701/train_wholebrain_pretrain.py"
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
