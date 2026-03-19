#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${1:-$PWD}"
LOCAL_REPO_ROOT="${LOCAL_REPO_ROOT:-/home/uais5/hkw/neural_token}"

link_dir() {
  local source_dir="$1"
  local target_dir="$2"

  if [[ ! -e "${source_dir}" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "${target_dir}")"

  if [[ -L "${target_dir}" ]]; then
    return 0
  fi

  if [[ -d "${target_dir}" ]]; then
    if [[ -z "$(find "${target_dir}" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
      rmdir "${target_dir}"
    else
      return 0
    fi
  elif [[ -e "${target_dir}" ]]; then
    return 0
  fi

  ln -s "${source_dir}" "${target_dir}"
}

echo "Preparing local agent environment from ${LOCAL_REPO_ROOT} into ${WORKSPACE_ROOT}"

link_dir "${LOCAL_REPO_ROOT}/data" "${WORKSPACE_ROOT}/data"
link_dir "${LOCAL_REPO_ROOT}/experiments/configs" "${WORKSPACE_ROOT}/experiments/configs"
link_dir "${LOCAL_REPO_ROOT}/experiments/runs" "${WORKSPACE_ROOT}/experiments/runs"
link_dir "${LOCAL_REPO_ROOT}/experiments/probe_results" "${WORKSPACE_ROOT}/experiments/probe_results"
link_dir "${LOCAL_REPO_ROOT}/logs" "${WORKSPACE_ROOT}/logs"

if [[ -x "${LOCAL_REPO_ROOT}/.venv/bin/python" ]]; then
  echo "LOCAL_VENV_PYTHON=${LOCAL_REPO_ROOT}/.venv/bin/python" >> "${GITHUB_ENV:-/dev/null}" 2>/dev/null || true
  echo "Detected local virtual environment: ${LOCAL_REPO_ROOT}/.venv/bin/python"
fi