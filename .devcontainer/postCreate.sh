#!/usr/bin/env bash
set -euo pipefail

bash scripts/install-path.sh
mkdir -p .cache/clip outputs data/videos data/datasets checkpoints experiments
