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
