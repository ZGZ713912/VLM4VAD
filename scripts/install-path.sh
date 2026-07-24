#!/usr/bin/env bash
# Install global command shims so detect-vlm4vad works from any cwd.
# Usage:
#   ./scripts/install-path.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/_common.sh"

ROOT="$(vlm4vad_repo_root)"
BIN_DIR="${HOME}/.local/bin"
ENV_SETUP="${ROOT}/scripts/env_setup.sh"
MARKER="VLM4VAD_ENV_SETUP"

mkdir -p "${BIN_DIR}"
chmod +x \
  "${ROOT}/scripts/setup-vlm4vad" \
  "${ROOT}/scripts/detect-vlm4vad" \
  "${ROOT}/scripts/check-vlm4vad" \
  "${ROOT}/scripts/env_setup.sh" \
  "${ROOT}/scripts/install-path.sh" \
  "${ROOT}/scripts/_common.sh" 2>/dev/null || true

for cmd in setup-vlm4vad detect-vlm4vad check-vlm4vad; do
  ln -sfn "${ROOT}/scripts/${cmd}" "${BIN_DIR}/${cmd}"
done

append_source_line() {
  local rc_file="$1"
  local line="$2"
  touch "${rc_file}"
  if ! grep -q "${MARKER}" "${rc_file}" 2>/dev/null; then
    printf '\n# %s\n%s\n' "${MARKER}" "${line}" >> "${rc_file}"
  fi
}

source_line="[ -f \"${ENV_SETUP}\" ] && source \"${ENV_SETUP}\""
append_source_line "${HOME}/.zshrc" "${source_line}"
append_source_line "${HOME}/.bashrc" "${source_line}"

# Also keep a portable env file for non-login shells.
if [[ ! -f "${HOME}/env_setup.sh" ]] || [[ "$(readlink -f "${HOME}/env_setup.sh" 2>/dev/null || true)" != "${ENV_SETUP}" ]]; then
  ln -sfn "${ENV_SETUP}" "${HOME}/env_setup.sh"
fi

vlm4vad_info "Installed shims into ${BIN_DIR}"
vlm4vad_info "Shell rc sources: ${ENV_SETUP}"
vlm4vad_info "Open a new terminal, or run: source ${ENV_SETUP}"
vlm4vad_info "Then: detect-vlm4vad / setup-vlm4vad / check-vlm4vad"
