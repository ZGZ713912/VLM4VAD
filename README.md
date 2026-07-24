# VLM4VAD

基于视觉语言模型（VLM）的视频异常检测。

输入一段监控/场景视频和异常文本描述（如「打架」「爆炸」），输出每个视频的异常分数、是否异常，以及结果 JSON。

## 方法概述

整体流程：

```text
输入视频
  → 抽帧采样
  → CLIP 视觉特征提取
  → 异常文本 Prompt 编码
  → 视觉-文本相似度 / 异常打分
  → 时序窗口聚合
  → 阈值判定 + JSON 输出
```

核心思路：

1. **视觉编码**：用 CLIP（ViT-B/16）对采样帧提取视觉特征。
2. **文本引导**：把异常类别写成自然语言 prompt（中文/英文均可，内部会规范化）。
3. **跨模态匹配**：在视觉特征与异常文本特征之间计算异常相关分数。
4. **时序聚合**：按窗口切分长视频特征，对窗口内高分做 top-k 平均，得到视频级分数。
5. **阈值判定**：分数 ≥ `threshold` 判为 abnormal，否则 normal。

默认配置见 `configs/inference.yaml`。

## 快速开始

### 1. 准备数据

```bash
# 测试视频
cp /path/to/your.mp4 data/videos/

# 模型权重（示例名）
cp /path/to/model_xd.pth checkpoints/model_xd.pth
```

### 2. 环境准备 / 检查

```bash
# 首次：创建目录 + 安装依赖 + 检查
./scripts/setup-vlm4vad --install

# 日常：检查视频/权重/依赖是否就绪
./scripts/check-vlm4vad
```

### 3. 一键检测（视频 → 结果）

```bash
./scripts/detect-vlm4vad
```

等价于：读 `data/videos/` + `checkpoints/model_xd.pth`，写出 `outputs/results.json`。

常用变体：

```bash
# 指定异常文本
./scripts/detect-vlm4vad --anomaly-text "爆炸"

# 单个视频
./scripts/detect-vlm4vad --video data/videos/demo.mp4 --anomaly-text "打架"

# 自定义配置
./scripts/detect-vlm4vad --config configs/inference.yaml

# 自定义输出
./scripts/detect-vlm4vad --output outputs/run_001.json
```

终端会打印：

```text
demo.mp4    abnormal    0.8123    fighting
```

JSON 中包含分数、阈值、窗口分数等字段。

## 脚本说明（类似 RMCS 的一键入口）

| 脚本 | 作用 |
|------|------|
| `scripts/setup-vlm4vad` | 创建目录、检查 Python 依赖 |
| `scripts/check-vlm4vad` | 检查配置、视频、权重是否就绪 |
| `scripts/detect-vlm4vad` | 一键完成「视频 → 异常分数 → JSON」 |

### 全局命令（像 RMCS 一样任意目录可调用）

RMCS 通过 `env_setup.zsh` 把 `.script` 加进 `PATH`。VLM4VAD 同样处理：

```bash
# 容器创建时会自动执行；本地也可手动：
./scripts/install-path.sh
source scripts/env_setup.sh   # 或开新终端
```

之后在任意目录：

```bash
detect-vlm4vad
setup-vlm4vad
check-vlm4vad
```

Dev Container 里还通过 `remoteEnv.PATH` 注入 `${containerWorkspaceFolder}/scripts`，新终端开箱即用。

## 目录结构

```text
VLM4VAD/
├── src/                 # 源代码
│   ├── detect.py        # CLI 入口
│   ├── detection/       # 异常打分
│   ├── features/        # CLIP 特征
│   ├── preprocessing/   # 读视频 / 采样
│   ├── prompts/         # 异常文本规范化
│   └── clip/            # CLIP 实现
├── configs/             # YAML 配置
├── data/
│   ├── videos/          # 本地测试视频（不提交）
│   └── datasets/        # 数据集（不提交）
├── checkpoints/         # 模型权重（不提交）
├── scripts/             # 一键脚本
├── docs/
│   ├── references/      # 参考材料 / 申报书
│   └── guides/          # 开发说明
├── experiments/         # 实验记录
├── outputs/             # 推理结果
├── tests/
├── notebooks/
└── .devcontainer/
```

## 配置说明

`configs/inference.yaml` 关键字段：

| 字段 | 含义 | 默认 |
|------|------|------|
| `video_dir` | 视频目录 | `data/videos` |
| `anomaly_text` | 异常描述 | `打架` |
| `model_checkpoint` | 权重路径 | `checkpoints/model_xd.pth` |
| `device` | `auto` / `cpu` / `cuda` | `auto` |
| `threshold` | 异常阈值 | `0.5` |
| `frame_stride` | 抽帧步长 | `16` |
| `output` | 结果 JSON | `outputs/results.json` |

也可以不走脚本，直接调用 Python：

```bash
PYTHONPATH=src python src/detect.py --config configs/inference.yaml
```

## 开发容器

仓库已提供 `.devcontainer/`，用 VS Code Dev Containers / Codespaces 打开即可。

- 默认 Python 3.11
- 默认 `torch==2.4.1`、`torchvision==0.19.1`（CPU）
- CUDA 12.1：把 `VLM4VAD_TORCH_VARIANT` 设为 `cu121`，并加 `--gpus=all`
- 容器创建后会安装依赖，并准备 `outputs/`、`data/videos/`、`checkpoints/`、`scripts/` PATH

也可手动用 `.devcontainer/Dockerfile` 作为基础镜像。

## 输出示例

`outputs/results.json`：

```json
[
  {
    "video_path": ".../data/videos/demo.mp4",
    "anomaly_text": "打架",
    "canonical_anomaly_text": "fighting",
    "abnormal": true,
    "score": 0.8123,
    "threshold": 0.5,
    "sampled_frames": 42,
    "window_scores": [0.71, 0.81],
    "frame_scores": [0.12, 0.55, 0.88]
  }
]
```

## 注意

- `data/`、`checkpoints/`、`outputs/` 中的大文件默认不进 Git
- 权重需自行放到 `checkpoints/`，或用 `--checkpoint` 指定绝对路径
- 首次加载 CLIP 权重时会下载/缓存，需保证网络可用
