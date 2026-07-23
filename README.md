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
