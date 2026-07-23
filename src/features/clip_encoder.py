from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image


class CLIPFrameEncoder:
    def __init__(self, clip_model, preprocess, device: torch.device | str):
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

    def _crop_frame(self, frame: np.ndarray, crop_type: int) -> np.ndarray:
        resized = cv2.resize(frame, dsize=(340, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if crop_type == 0:
            crop = rgb[16:240, 58:282, :]
        elif crop_type == 1:
            crop = rgb[:224, :224, :]
        elif crop_type == 2:
            crop = rgb[:224, -224:, :]
        elif crop_type == 3:
            crop = rgb[-224:, :224, :]
        elif crop_type == 4:
            crop = rgb[-224:, -224:, :]
        elif crop_type == 5:
            crop = cv2.flip(rgb[16:240, 58:282, :], 1)
        elif crop_type == 6:
            crop = cv2.flip(rgb[:224, :224, :], 1)
        elif crop_type == 7:
            crop = cv2.flip(rgb[:224, -224:, :], 1)
        elif crop_type == 8:
            crop = cv2.flip(rgb[-224:, :224, :], 1)
        else:
            crop = cv2.flip(rgb[-224:, -224:, :], 1)

        return crop

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        crops = [Image.fromarray(self._crop_frame(frame, crop_type)) for crop_type in range(10)]
        return torch.stack([self.preprocess(crop) for crop in crops], dim=0)

    @torch.no_grad()
    def extract_features(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("No frames provided for feature extraction")

        frame_features = []
        for frame in frames:
            crops = self._frame_to_tensor(frame).to(self.device)
            features = self.clip_model.encode_image(crops)
            frame_features.append(features.mean(dim=0))

        stacked = torch.stack(frame_features, dim=0)
        return stacked.detach().cpu().numpy().astype(np.float32)
