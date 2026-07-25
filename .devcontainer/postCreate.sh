#!/usr/bin/env bash
set -euo pipefail

# Torch variant/version come from devcontainer.json containerEnv, or defaults in setup-vlm4vad.
bash scripts/setup-vlm4vad --install
python -m pip install ipykernel
bash scripts/install-path.sh
python -c "import cv2, torch, torchvision, yaml, scipy" >/dev/null
