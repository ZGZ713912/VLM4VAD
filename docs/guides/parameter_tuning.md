# VLM4VAD 调参说明

这份说明只覆盖推理阶段最值得调的参数。目标不是把参数调得更多，而是先把最影响效果的几个点调对。

## 建议先调哪些

建议顺序：

1. `threshold`
2. `frame_stride`
3. `window_stride`
4. `smoothing_kernel`
5. `window_topk_ratio`
6. `video_topk`

如果视频很长，再考虑 `max_frames` 控制开销。

## 关键参数

| 参数 | 作用 | 默认值 | 建议范围 | 调大后的影响 | 调小后的影响 |
|------|------|--------|----------|--------------|--------------|
| `threshold` | 最终异常判定阈值 | `0.45` | `0.40` - `0.60` | 误报更少，但更容易漏检 | 更敏感，但误报更高 |
| `frame_stride` | 抽帧步长 | `8` | `4` - `16` | 更快，但短时异常更容易被跳过 | 更细致，但计算更慢 |
| `window_stride` | 滑动窗口步长 | `128` | `64` - `256` | 窗口更稀疏，速度更快 | 窗口重叠更多，边界处更稳 |
| `smoothing_kernel` | 帧分数平滑核大小，必须是奇数 | `5` | `1` / `3` / `5` / `7` | 更平滑，尖峰更少，但短异常可能被抹平 | 更敏感，保留突发异常 |
| `window_topk_ratio` | 每个窗口里取最高分帧的比例做平均 | `0.125` | `0.08` - `0.20` | 更关注持续性异常，抗噪更强 | 更关注少量高分帧，适合瞬时异常 |
| `video_topk` | 最终视频分数取 top-N 窗口平均 | `2` | `1` - `3` | 更稳，但单个异常片段不够突出时分数会被拉低 | 更敏感，但更容易被偶发高分误触发 |
| `max_frames` | 单视频最多采样帧数，`null` 表示不限 | `null` | `256` - `1024` 或 `null` | 更省时，但长视频后段信息可能丢失 | 保留更多信息，但更慢 |

## 现在这版默认参数为什么这样设

- `frame_stride: 8`
  之前 `16` 对短时异常偏粗，容易漏掉打架、摔倒、爆炸这类持续时间不长的事件。
- `window_stride: 128`
  让 `visual_length: 256` 的窗口有 50% 重叠，减少异常刚好落在窗口边界时被切碎的问题。
- `smoothing_kernel: 5`
  用轻量平滑压掉单帧尖峰，不会像大核那样明显抹平短异常。
- `video_topk: 2`
  比直接取单个最大窗口更稳，能降低偶发误报。
- `threshold: 0.45`
  配合更稳的窗口聚合后，默认阈值略低一些更容易保住召回率。

## 典型场景建议

### 1. 短时暴力/冲突事件，优先召回

适合打架、推搡、突然爆炸这类持续时间短的异常：

```yaml
threshold: 0.40
frame_stride: 4
window_stride: 64
smoothing_kernel: 3
window_topk_ratio: 0.08
video_topk: 1
```

### 2. 常规监控，优先稳健

适合日常监控巡检，希望少一些误报：

```yaml
threshold: 0.48
frame_stride: 8
window_stride: 128
smoothing_kernel: 5
window_topk_ratio: 0.125
video_topk: 2
```

### 3. 长视频批量扫描，优先速度

适合先粗筛一遍，再回头复查高分视频：

```yaml
threshold: 0.50
frame_stride: 12
max_frames: 512
window_stride: 256
smoothing_kernel: 3
window_topk_ratio: 0.15
video_topk: 2
```

## 不建议随便改的参数

下面这些参数和 checkpoint 的结构强相关，推理时通常不要改：

- `visual_length`
- `visual_width`
- `embed_dim`
- `visual_head`
- `visual_layers`
- `attn_window`
- `prompt_prefix`
- `prompt_postfix`

这几项如果和训练 checkpoint 不匹配，轻则效果变差，重则直接加载失败。

## 实际调参建议

1. 先固定 checkpoint 和 `anomaly_text`，只调 `threshold`。
2. 如果漏检多，先减小 `frame_stride`，再减小 `threshold`。
3. 如果误报多，先增大 `threshold`，再增大 `video_topk` 或 `window_topk_ratio`。
4. 如果异常经常出现在窗口边界附近，优先减小 `window_stride`。
5. 用验证集调，不要拿测试集反复试。

## CLI 示例

```bash
./scripts/detect-vlm4vad \
  --anomaly-text "打架" \
  --threshold 0.42 \
  --frame-stride 4 \
  --window-stride 64 \
  --smoothing-kernel 3 \
  --video-topk 1
```
