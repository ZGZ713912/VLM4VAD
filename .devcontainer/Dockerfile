FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

ARG DEBIAN_FRONTEND=noninteractive

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

USER vscode

RUN if [ ! -d /home/vscode/.oh-my-zsh/.git ]; then rm -rf /home/vscode/.oh-my-zsh; git clone --depth=1 https://github.com/ohmyzsh/ohmyzsh.git /home/vscode/.oh-my-zsh; fi \
 && if [ ! -f /home/vscode/.zshrc ]; then cp /home/vscode/.oh-my-zsh/templates/zshrc.zsh-template /home/vscode/.zshrc; fi \
 && sed -i 's/^ZSH_THEME=.*/ZSH_THEME="robbyrussell"/' /home/vscode/.zshrc \
 && sed -i 's/^plugins=.*/plugins=(git)/' /home/vscode/.zshrc

USER root

RUN usermod --shell /bin/zsh vscode
