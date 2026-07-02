#!/usr/bin/env bash

set -euo pipefail

print_help() {
    cat <<'EOF'
Usage:
  bash experiments/scripts/launch_training_nohup.sh --task physiology-semantic-tokenizer [task args]

Status:
  Launches the active P2-P5 training entrypoint. The entrypoint enforces the
  E0 optimizer gate; software smoke runs cannot bypass it.

Historical workflows:
  See experiments/scripts/archive/pre_physiology_semantic_20260701/README.md.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    print_help
    exit 0
fi

if [[ "${1:-}" != "--task" || -z "${2:-}" ]]; then
    echo "--task is required" >&2
    print_help >&2
    exit 2
fi

if [[ "$2" != "physiology-semantic-tokenizer" ]]; then
    echo "Task '$2' is not registered in the active architecture." >&2
    echo "Pre-redesign launchers are isolated in the dated script archive." >&2
    exit 2
fi

shift 2

if [[ "$#" -eq 0 ]]; then
    echo "Training arguments are required; pass --config and --dry-run or --smoke." >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
entrypoint="$repo_root/experiments/train_physiology_semantic_tokenizer.py"
log_root="$repo_root/experiments/runs/physiology_semantic_tokenizer/launcher_logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="$log_root/${timestamp}_physiology_semantic_tokenizer.log"

if [[ ! -x "$python_bin" ]]; then
    echo "Missing project Python: $python_bin" >&2
    exit 2
fi

mkdir -p "$log_root"
cd "$repo_root"
setsid -f nohup "$python_bin" "$entrypoint" "$@" >"$log_path" 2>&1 < /dev/null
echo "Launched physiology-semantic-tokenizer. Log: $log_path"
