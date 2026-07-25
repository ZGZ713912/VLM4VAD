# AGENTS.md — VLM4VAD 项目 AI 助手使用规范

## 项目概述

- **项目**: 基于视觉语言模型的视频异常检测
- **语言**: Python 3.10+
- **框架**: PyTorch
- **核心依赖**: torch, torchvision, opencv-python, numpy, scipy, Pillow, PyYAML, ftfy, regex, tqdm
- **VLM 模型**: CLIP（仓库内 `src/clip`）；后续可扩展 BLIP 等
- **依赖清单**: 根目录 `requirements.txt`（不含 torch；CPU/CUDA 由 `setup-vlm4vad --install` 单独安装）

## 目录结构约定

```
VLM4VAD/
├── src/                 # 源代码
│   ├── preprocessing/   # 数据预处理（抽帧、切片）
│   ├── features/        # 特征提取（VLM 调用）
│   ├── detection/       # 异常检测算法
│   ├── prompts/         # 文本 prompt
│   ├── clip/            # CLIP 实现
│   └── utils/           # 通用工具
├── configs/             # 配置文件
├── data/
│   ├── videos/          # 本地测试视频（不提交）
│   └── datasets/        # 数据集（不提交）
├── checkpoints/         # 模型权重（不提交）
├── scripts/             # 工具脚本
├── docs/
│   ├── references/      # 参考材料 / 申报书
│   └── guides/          # 开发与使用说明
├── experiments/         # 实验记录
├── outputs/             # 推理结果
├── tests/               # 单元测试
├── notebooks/           # Jupyter 实验笔记
├── .devcontainer/       # 开发容器
├── PLAN.md              # 项目计划
└── AGENTS.md            # AI 助手规范
```

## 代码规范

- 遵循 PEP 8 风格
- 类型注解：函数参数和返回值必须标注类型
- import 顺序：标准库 → 第三方库 → 本地模块
- 日志使用 `logging` 模块，不随意 `print`
- 配置使用 yaml 文件，不硬编码路径和超参数

## 常用命令

```bash
# 安装依赖（先 torch/torchvision，再 requirements.txt）
./scripts/setup-vlm4vad --install
# CUDA 12.1 示例：
# ./scripts/setup-vlm4vad --install --torch-variant cu121

# 代码检查（如使用）
ruff check src/
mypy src/

# 运行测试
pytest tests/

# 推理
python src/detect.py --config configs/inference.yaml
python src/detect.py --video-dir data/videos --anomaly-text "打架" --checkpoint checkpoints/model_xd.pth
```

## 文档获取指引

当需要查询框架/库的文档时，使用 Context7 MCP 工具：

1. 使用 `resolve-library-id` 搜索库名（如 PyTorch, transformers, CLIP）
2. 使用 `query-docs` 获取最新文档和代码示例

优先查询的内容：
- PyTorch 模型定义与训练
- HuggingFace transformers 的 VLM 使用
- OpenCV 视频处理 API
- 视觉语言模型的最新进展

## 注意

- 不将数据文件提交到仓库（`data/videos/`、`data/datasets/` 已在 `.gitignore` 中）
- 模型权重放在 `checkpoints/`，不提交；记录下载链接或本地路径
- 实验配置与结果记录在 `experiments/`，推理结果写入 `outputs/`
- 参考材料放在 `docs/references/`，开发说明放在 `docs/guides/`
