# VLM4VAD
基于视觉语言模型的视频异常检测。

## 用法

```bash
python src/detect.py --video-dir /path/to/videos --anomaly-text "打架" --checkpoint ../VadCLIP/model/model_xd.pth
```

也可以使用配置文件：

```bash
python src/detect.py --config configs/inference.yaml
```

输出为每个视频的异常分数和判断结果。

## 开发容器

仓库已经提供 `.devcontainer/`，直接用 VS Code Dev Containers / Codespaces 打开即可。

- 默认使用 `Python 3.11`
- 默认安装 `torch==2.4.1`、`torchvision==0.19.1`
- 默认是 CPU 版本，如需 CUDA 12.1，把 `VLM4VAD_TORCH_VARIANT` 改成 `cu121`
- 如果切到 CUDA，还需要在 devcontainer 运行参数里额外开启 GPU，例如 `--gpus=all`
- 容器内会自动安装 `requirements.txt` 并创建 `outputs/`

如果只想手动构建容器，也可以把 `.devcontainer/Dockerfile` 当作基础镜像使用。
