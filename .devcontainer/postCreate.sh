#!/usr/bin/env bash
set -euo pipefail

torch_variant="${VLM4VAD_TORCH_VARIANT:-cpu}"
torch_version="${VLM4VAD_TORCH_VERSION:-2.4.1}"
torchvision_version="${VLM4VAD_TORCHVISION_VERSION:-0.19.1}"

case "${torch_variant}" in
  cpu)
    torch_index_url="https://download.pytorch.org/whl/cpu"
    ;;
  cu121)
    torch_index_url="https://download.pytorch.org/whl/cu121"
    ;;
  *)
    echo "Unsupported VLM4VAD_TORCH_VARIANT: ${torch_variant}" >&2
    echo "Use one of: cpu, cu121" >&2
    exit 1
    ;;
esac

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "torch==${torch_version}" \
  "torchvision==${torchvision_version}" \
  --index-url "${torch_index_url}"
python -m pip install -r requirements.txt
python -m pip install ipykernel

mkdir -p outputs data/videos data/datasets checkpoints experiments scripts

bash scripts/install-path.sh

python -c "import cv2, torch, torchvision, yaml" >/dev/null
