from __future__ import annotations

from pathlib import Path
from typing import List

from detection.scorer import VideoAnomalyDetector, VideoDetectionResult


def detect_folder(
    video_dir: str | Path,
    anomaly_text: str,
    checkpoint_path: str | Path,
    device: str | None = None,
    threshold: float = 0.5,
    frame_stride: int = 16,
    max_frames: int | None = None,
    window_stride: int | None = None,
    smoothing_kernel: int = 5,
    window_topk_ratio: float = 0.125,
    video_topk: int = 2,
    recursive: bool = True,
) -> List[VideoDetectionResult]:
    detector = VideoAnomalyDetector(
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=threshold,
        frame_stride=frame_stride,
        max_frames=max_frames,
        window_stride=window_stride,
        smoothing_kernel=smoothing_kernel,
        window_topk_ratio=window_topk_ratio,
        video_topk=video_topk,
    )
    return detector.predict_folder(video_dir, anomaly_text, recursive=recursive)


def detect_video(
    video_path: str | Path,
    anomaly_text: str,
    checkpoint_path: str | Path,
    device: str | None = None,
    threshold: float = 0.5,
    frame_stride: int = 16,
    max_frames: int | None = None,
    window_stride: int | None = None,
    smoothing_kernel: int = 5,
    window_topk_ratio: float = 0.125,
    video_topk: int = 2,
) -> VideoDetectionResult:
    detector = VideoAnomalyDetector(
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=threshold,
        frame_stride=frame_stride,
        max_frames=max_frames,
        window_stride=window_stride,
        smoothing_kernel=smoothing_kernel,
        window_topk_ratio=window_topk_ratio,
        video_topk=video_topk,
    )
    return detector.predict_video(video_path, anomaly_text)
