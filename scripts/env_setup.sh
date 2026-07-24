#!/usr/bin/env bash
# Source this from shell rc so VLM4VAD scripts are available anywhere.
#   source /workspaces/VLM4VAD/scripts/env_setup.sh

: "${VLM4VAD_PATH:=/workspaces/VLM4VAD}"

# Prefer the real repo path of this file when sourced from the workspace.
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _vlm4vad_self="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  VLM4VAD_PATH="${_vlm4vad_self}"
  unset _vlm4vad_self
elif [[ -n "${ZSH_VERSION:-}" && -n "${(%):-%x}" ]]; then
  _vlm4vad_self="$(cd "$(dirname "${(%):-%x}")/.." && pwd)"
  VLM4VAD_PATH="${_vlm4vad_self}"
  unset _vlm4vad_self
fi

export VLM4VAD_PATH
export PATH="${VLM4VAD_PATH}/scripts:${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${VLM4VAD_PATH}/src${PYTHONPATH:+:${PYTHONPATH}}"
