from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

VIDEO_SUFFIXES = {
    ".avi",
    ".flv",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


def iter_video_files(directory: str | Path, recursive: bool = True) -> List[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Video directory not found: {root}")

    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    return sorted(files)


def read_sampled_frames(
    video_path: str | Path,
    frame_stride: int = 16,
    max_frames: int | None = None,
) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    sampled_frames: List[np.ndarray] = []
    index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_stride <= 1 or index % frame_stride == 0:
                sampled_frames.append(frame)
                if max_frames is not None and len(sampled_frames) >= max_frames:
                    break

            index += 1
    finally:
        capture.release()

    if not sampled_frames:
        raise ValueError(f"No frames sampled from video: {video_path}")

    return sampled_frames
