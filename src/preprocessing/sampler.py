from __future__ import annotations

from typing import Tuple

import numpy as np


def chunk_feature_sequence(features: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim != 2:
        raise ValueError(f"Expected [T, D] features, got {feature_array.shape}")
    if feature_array.shape[0] == 0:
        raise ValueError("Feature sequence is empty")

    windows = []
    lengths = []
    total = feature_array.shape[0]

    for start in range(0, total, window_size):
        chunk = feature_array[start : start + window_size]
        lengths.append(int(chunk.shape[0]))
        if chunk.shape[0] < window_size:
            padding = np.zeros((window_size - chunk.shape[0], feature_array.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, padding], axis=0)
        windows.append(chunk)

    return np.stack(windows, axis=0), np.asarray(lengths, dtype=np.int64)
