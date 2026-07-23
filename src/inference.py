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
    recursive: bool = True,
) -> List[VideoDetectionResult]:
    detector = VideoAnomalyDetector(
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=threshold,
        frame_stride=frame_stride,
    )
    return detector.predict_folder(video_dir, anomaly_text, recursive=recursive)


def detect_video(
    video_path: str | Path,
    anomaly_text: str,
    checkpoint_path: str | Path,
    device: str | None = None,
    threshold: float = 0.5,
    frame_stride: int = 16,
) -> VideoDetectionResult:
    detector = VideoAnomalyDetector(
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=threshold,
        frame_stride=frame_stride,
    )
    return detector.predict_video(video_path, anomaly_text)
