# VLM4VAD
基于视觉语言模型的视频异常检测。

## 目录结构

```text
VLM4VAD/
├── src/                 # 源代码
├── configs/             # 配置文件
├── data/
│   ├── videos/          # 本地测试视频（不提交）
│   └── datasets/        # 数据集（不提交）
├── checkpoints/         # 模型权重（不提交）
├── scripts/             # 工具脚本
├── docs/
│   ├── references/      # 参考材料 / 申报书等
│   └── guides/          # 开发与使用说明
├── experiments/         # 实验记录
├── outputs/             # 推理结果
├── tests/               # 单元测试
├── notebooks/           # Jupyter 实验笔记
└── .devcontainer/       # 开发容器
```

## 用法

把测试视频放到 `data/videos/`，把权重放到 `checkpoints/`，然后运行：

```bash
python src/detect.py \
  --video-dir data/videos \
  --anomaly-text "打架" \
  --checkpoint checkpoints/model_xd.pth
```

也可以使用配置文件：

```bash
python src/detect.py --config configs/inference.yaml
```

输出为每个视频的异常分数和判断结果，默认写入 `outputs/results.json`。

## 开发容器

仓库已经提供 `.devcontainer/`，直接用 VS Code Dev Containers / Codespaces 打开即可。

- 默认使用 `Python 3.11`
- 默认安装 `torch==2.4.1`、`torchvision==0.19.1`
- 默认是 CPU 版本，如需 CUDA 12.1，把 `VLM4VAD_TORCH_VARIANT` 改成 `cu121`
- 如果切到 CUDA，还需要在 devcontainer 运行参数里额外开启 GPU，例如 `--gpus=all`
- 容器内会自动安装 `requirements.txt` 并创建 `outputs/`、`data/videos/`、`checkpoints/`

如果只想手动构建容器，也可以把 `.devcontainer/Dockerfile` 当作基础镜像使用。
