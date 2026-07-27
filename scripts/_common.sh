#!/usr/bin/env bash

set -euo pipefail

vlm4vad_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

vlm4vad_info() {
  echo "> $*"
}

vlm4vad_error() {
  echo "> ERROR: $*" >&2
}

vlm4vad_require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    vlm4vad_error "Missing command: ${cmd}"
    exit 1
  fi
}

vlm4vad_python_bin() {
  local root="${1:-$(vlm4vad_repo_root)}"

  if [[ -n "${VLM4VAD_PYTHON:-}" ]]; then
    printf '%s\n' "${VLM4VAD_PYTHON}"
    return 0
  fi

  if [[ -x "${root}/.venv/bin/python" ]]; then
    printf '%s\n' "${root}/.venv/bin/python"
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  return 1
}

vlm4vad_require_python() {
  if ! vlm4vad_python_bin "$1" >/dev/null; then
    vlm4vad_error "Missing command: python"
    exit 1
  fi
}

vlm4vad_ensure_dirs() {
  local root="$1"
  mkdir -p \
    "${root}/.cache/clip" \
    "${root}/data/videos" \
    "${root}/data/datasets" \
    "${root}/checkpoints" \
    "${root}/outputs" \
    "${root}/experiments" \
    "${root}/notebooks" \
    "${root}/tests" \
    "${root}/docs/guides" \
    "${root}/docs/references"
}
