from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

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
        max_frames: int | None = None,
        window_stride: int | None = None,
        smoothing_kernel: int = 5,
        window_topk_ratio: float = 0.125,
        video_topk: int = 2,
    ):
        if device in (None, "auto"):
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device

        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be at least 1 when provided")

        resolved_window_stride = max(1, visual_length // 2) if window_stride is None else int(window_stride)
        if resolved_window_stride < 1 or resolved_window_stride > visual_length:
            raise ValueError("window_stride must be in [1, visual_length]")
        if smoothing_kernel < 1:
            raise ValueError("smoothing_kernel must be at least 1")
        if smoothing_kernel % 2 == 0:
            raise ValueError("smoothing_kernel must be an odd integer")
        if not 0 < window_topk_ratio <= 1:
            raise ValueError("window_topk_ratio must be in (0, 1]")
        if video_topk < 1:
            raise ValueError("video_topk must be at least 1")

        self.device = torch.device(resolved_device)
        self.threshold = threshold
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.visual_length = visual_length
        self.window_stride = resolved_window_stride
        self.smoothing_kernel = smoothing_kernel
        self.window_topk_ratio = window_topk_ratio
        self.window_topk_min = 2
        self.video_topk = video_topk

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

    def _aggregate_window_score(self, values: torch.Tensor) -> float:
        flat = values.flatten()
        if flat.numel() == 0:
            return 0.0
        k = min(flat.numel(), max(self.window_topk_min, math.ceil(flat.numel() * self.window_topk_ratio)))
        return float(torch.topk(flat, k=k).values.mean().item())

    def _smooth_frame_scores(self, frame_scores: torch.Tensor) -> torch.Tensor:
        if self.smoothing_kernel <= 1 or frame_scores.numel() < 2:
            return frame_scores

        values = frame_scores.view(1, 1, -1)
        padding = self.smoothing_kernel // 2
        padded = F.pad(values, (padding, padding), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=self.smoothing_kernel, stride=1).view(-1)

    def _aggregate_video_score(self, window_scores: List[float]) -> float:
        if not window_scores:
            return 0.0

        values = torch.tensor(window_scores, dtype=torch.float32)
        k = min(values.numel(), self.video_topk)
        return float(torch.topk(values, k=k).values.mean().item())

    def _score_feature_windows(
        self,
        windows: np.ndarray,
        lengths: np.ndarray,
        starts: np.ndarray,
        anomaly_text: str,
    ) -> tuple[torch.Tensor, List[float], List[float]]:
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

        total_frames = max(int(start + length) for start, length in zip(starts.tolist(), lengths.tolist()))
        frame_score_sum = torch.zeros(total_frames, dtype=combined.dtype, device=self.device)
        frame_score_count = torch.zeros(total_frames, dtype=combined.dtype, device=self.device)

        for index, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
            active = combined[index, :length]
            frame_score_sum[start : start + length] += active
            frame_score_count[start : start + length] += 1

        frame_sequence = frame_score_sum / frame_score_count.clamp_min(1.0)
        frame_sequence = self._smooth_frame_scores(frame_sequence)

        window_scores: List[float] = []
        for start, length in zip(starts.tolist(), lengths.tolist()):
            active = frame_sequence[start : start + length]
            window_scores.append(self._aggregate_window_score(active))

        frame_scores = frame_sequence.detach().cpu().tolist()

        return combined, window_scores, frame_scores

    def predict_video(self, video_path: str | Path, anomaly_text: str) -> VideoDetectionResult:
        frames = read_sampled_frames(video_path, frame_stride=self.frame_stride, max_frames=self.max_frames)
        features = self.encoder.extract_features(frames)
        windows, lengths, starts = chunk_feature_sequence(features, self.visual_length, self.window_stride)
        _, window_scores, frame_scores = self._score_feature_windows(windows, lengths, starts, anomaly_text)

        score = self._aggregate_video_score(window_scores)
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
