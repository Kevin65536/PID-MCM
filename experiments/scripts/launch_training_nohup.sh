#!/usr/bin/env bash

set -euo pipefail

print_help() {
    cat <<'EOF'
Usage:
  bash experiments/scripts/launch_training_nohup.sh --task physiology-semantic-tokenizer [task args]

Status:
  The active physiology-semantic training entrypoint is reserved but has not
  passed its implementation gates yet. This launcher intentionally starts no
  pre-redesign task.

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

echo "Task 'physiology-semantic-tokenizer' is reserved but not implemented." >&2
echo "Implement P1/P2 and active run-root assertions before enabling launch." >&2
exit 2
