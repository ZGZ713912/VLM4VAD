from __future__ import annotations

from typing import Tuple

import numpy as np


def chunk_feature_sequence(
    features: np.ndarray,
    window_size: int,
    window_stride: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim != 2:
        raise ValueError(f"Expected [T, D] features, got {feature_array.shape}")
    if feature_array.shape[0] == 0:
        raise ValueError("Feature sequence is empty")
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    stride = window_size if window_stride is None else int(window_stride)
    if stride < 1 or stride > window_size:
        raise ValueError("window_stride must be in [1, window_size]")

    windows = []
    lengths = []
    starts = []
    total = feature_array.shape[0]

    for start in range(0, total, stride):
        chunk = feature_array[start : start + window_size]
        lengths.append(int(chunk.shape[0]))
        starts.append(start)
        if chunk.shape[0] < window_size:
            padding = np.zeros((window_size - chunk.shape[0], feature_array.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, padding], axis=0)
        windows.append(chunk)
        if start + window_size >= total:
            break

    return (
        np.stack(windows, axis=0),
        np.asarray(lengths, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
    )
