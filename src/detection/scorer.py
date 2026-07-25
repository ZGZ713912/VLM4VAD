from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

from clip import clip
from features.clip_encoder import CLIPFrameEncoder
from model import CLIPVAD
from preprocessing.sampler import chunk_feature_sequence
from preprocessing.video_io import iter_video_files, read_sampled_frames
from prompts import build_prompt_pairs, normalize_anomaly_text


@dataclass
class VideoDetectionResult:
    video_path: str
    anomaly_text: str
    canonical_anomaly_text: str
    abnormal: bool
    score: float
    threshold: float
    sampled_frames: int
    window_scores: List[float]
    frame_scores: List[float]

    def to_dict(self) -> dict:
        return asdict(self)


class VideoAnomalyDetector:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        visual_length: int = 256,
        visual_width: int = 512,
        embed_dim: int = 512,
        visual_head: int = 1,
        visual_layers: int = 1,
        attn_window: int = 64,
        prompt_prefix: int = 10,
        prompt_postfix: int = 10,
        threshold: float = 0.5,
        frame_stride: int = 16,
    ):
        if device in (None, "auto"):
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device

        self.device = torch.device(resolved_device)
        self.threshold = threshold
        self.frame_stride = frame_stride
        self.visual_length = visual_length

        clip_model, preprocess = clip.load("ViT-B/16", self.device)
        self.model = CLIPVAD(
            num_class=2,
            embed_dim=embed_dim,
            visual_length=visual_length,
            visual_width=visual_width,
            visual_head=visual_head,
            visual_layers=visual_layers,
            attn_window=attn_window,
            prompt_prefix=prompt_prefix,
            prompt_postfix=prompt_postfix,
            device=self.device,
            clip_model=clip_model,
            clip_preprocess=preprocess,
        ).to(self.device)

        try:
            state = torch.load(str(checkpoint_path), map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(str(checkpoint_path), map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

        self.encoder = CLIPFrameEncoder(self.model.clipmodel, self.model.clip_preprocess, self.device)

    @staticmethod
    def _topk_mean(values: torch.Tensor) -> float:
        flat = values.flatten()
        if flat.numel() == 0:
            return 0.0
        k = min(flat.numel(), max(1, flat.numel() // 16 + 1))
        return float(torch.topk(flat, k=k).values.mean().item())

    def _score_feature_windows(self, windows: np.ndarray, lengths: np.ndarray, anomaly_text: str) -> tuple[torch.Tensor, List[float], List[float]]:
        prompt_pairs = build_prompt_pairs(anomaly_text)
        window_tensors = torch.tensor(windows, dtype=torch.float32, device=self.device)
        length_tensors = torch.tensor(lengths, dtype=torch.long, device=self.device)

        prompt_scores = []
        with torch.no_grad():
            for prompt_pair in prompt_pairs:
                _, logits1, logits2 = self.model(window_tensors, None, prompt_pair, length_tensors)
                prob1 = torch.sigmoid(logits1.squeeze(-1))
                prob2 = 1.0 - torch.softmax(logits2, dim=-1)[:, :, 0]
                prompt_scores.append((prob1 + prob2) * 0.5)

        combined = torch.stack(prompt_scores, dim=0).mean(dim=0)

        window_scores: List[float] = []
        frame_scores: List[float] = []
        for index, length in enumerate(lengths.tolist()):
            active = combined[index, :length]
            window_scores.append(self._topk_mean(active))
            frame_scores.extend(active.detach().cpu().tolist())

        return combined, window_scores, frame_scores

    def predict_video(self, video_path: str | Path, anomaly_text: str) -> VideoDetectionResult:
        frames = read_sampled_frames(video_path, frame_stride=self.frame_stride)
        features = self.encoder.extract_features(frames)
        windows, lengths = chunk_feature_sequence(features, self.visual_length)
        _, window_scores, frame_scores = self._score_feature_windows(windows, lengths, anomaly_text)

        score = max(window_scores) if window_scores else 0.0
        canonical = normalize_anomaly_text(anomaly_text)
        return VideoDetectionResult(
            video_path=str(video_path),
            anomaly_text=anomaly_text,
            canonical_anomaly_text=canonical,
            abnormal=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            sampled_frames=len(frames),
            window_scores=window_scores,
            frame_scores=frame_scores,
        )

    def predict_folder(self, video_dir: str | Path, anomaly_text: str, recursive: bool = True) -> List[VideoDetectionResult]:
        video_files = iter_video_files(video_dir, recursive=recursive)
        if not video_files:
            raise ValueError(f"No video files found in {video_dir}")
        return [self.predict_video(video_path, anomaly_text) for video_path in video_files]
