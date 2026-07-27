FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VARIANT=cpu
ARG TORCH_VERSION=2.4.1
ARG TORCHVISION_VERSION=0.19.1

COPY requirements.txt /tmp/vlm4vad-requirements.txt

RUN rm -f /etc/apt/sources.list.d/yarn.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libxext6 \
      libxrender1 \
      pkg-config \
      zsh \
 && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    case "${TORCH_VARIANT}" in \
      cpu) TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu" ;; \
      cu121) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" ;; \
      *) echo "Unsupported TORCH_VARIANT: ${TORCH_VARIANT}" >&2; exit 1 ;; \
    esac \
 && PIP_FLAGS="--retries 20 --timeout 120" \
 && python -m pip install ${PIP_FLAGS} --upgrade pip setuptools wheel \
 && python -m pip install \
      ${PIP_FLAGS} \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      --index-url "${TORCH_INDEX_URL}" \
 && python -m pip install \
      ${PIP_FLAGS} \
      -r /tmp/vlm4vad-requirements.txt \
      ipykernel \
 && rm -f /tmp/vlm4vad-requirements.txt

USER vscode

RUN if [ ! -d /home/vscode/.oh-my-zsh/.git ]; then rm -rf /home/vscode/.oh-my-zsh; git clone --depth=1 https://github.com/ohmyzsh/ohmyzsh.git /home/vscode/.oh-my-zsh; fi \
 && if [ ! -f /home/vscode/.zshrc ]; then cp /home/vscode/.oh-my-zsh/templates/zshrc.zsh-template /home/vscode/.zshrc; fi \
 && sed -i 's/^ZSH_THEME=.*/ZSH_THEME="robbyrussell"/' /home/vscode/.zshrc \
 && sed -i 's/^plugins=.*/plugins=(git)/' /home/vscode/.zshrc

USER root

RUN usermod --shell /bin/zsh vscode
