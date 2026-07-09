# AGENTS.md — VLM4VAD 项目 AI 助手使用规范

## 项目概述

- **项目**: 基于视觉语言模型的视频异常检测
- **语言**: Python 3.10+
- **框架**: PyTorch
- **核心依赖**: transformers, opencv-python, torchvision, numpy, scikit-learn
- **VLM 模型**: CLIP (OpenAI), BLIP (Salesforce)

## 目录结构约定

```
VLM4VAD/
├── data/              # 数据集存放
├── src/               # 源代码
│   ├── preprocessing/ # 数据预处理（抽帧、切片）
│   ├── features/      # 特征提取（VLM 调用）
│   ├── detection/     # 异常检测算法
│   ├── temporal/      # 时序优化模块
│   └── visualization/ # 可视化展示
├── configs/           # 配置文件
├── notebooks/         # Jupyter 实验笔记
├── tests/             # 单元测试
├── scripts/           # 工具脚本
├── docs/              # 文档
├── PLAN.md            # 项目计划
└── AGENTS.md          # AI 助手规范
```

## 代码规范

- 遵循 PEP 8 风格
- 类型注解：函数参数和返回值必须标注类型
- import 顺序：标准库 → 第三方库 → 本地模块
- 日志使用 `logging` 模块，不随意 `print`
- 配置使用 yaml 文件，不硬编码路径和超参数

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 代码检查（如使用）
ruff check src/
mypy src/

# 运行测试
pytest tests/

# 训练/推理
python src/train.py --config configs/default.yaml
python src/detect.py --video path/to/video.mp4
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

- 不将数据文件提交到仓库（data/ 在 .gitignore 中）
- 模型权重文件不提交，记录下载链接或路径
- 实验配置与结果记录在 experiments/ 下，方便复现
