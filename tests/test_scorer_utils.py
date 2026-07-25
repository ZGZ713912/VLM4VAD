from __future__ import annotations

import unittest

import torch

from detection.scorer import VideoAnomalyDetector


class VideoAnomalyDetectorHelperTests(unittest.TestCase):
    def _build_detector(self) -> VideoAnomalyDetector:
        detector = VideoAnomalyDetector.__new__(VideoAnomalyDetector)
        detector.smoothing_kernel = 3
        detector.window_topk_ratio = 0.5
        detector.window_topk_min = 2
        detector.video_topk = 2
        return detector

    def test_smooth_frame_scores_uses_moving_average(self) -> None:
        detector = self._build_detector()

        smoothed = detector._smooth_frame_scores(torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]))

        expected = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        self.assertTrue(torch.allclose(smoothed, expected, atol=1e-6))

    def test_aggregate_video_score_uses_topk_mean(self) -> None:
        detector = self._build_detector()

        score = detector._aggregate_video_score([0.2, 0.6, 0.9])

        self.assertAlmostEqual(score, 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
