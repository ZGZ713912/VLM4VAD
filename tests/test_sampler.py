from __future__ import annotations

import unittest

import numpy as np

from preprocessing.sampler import chunk_feature_sequence


class ChunkFeatureSequenceTests(unittest.TestCase):
    def test_overlap_stride_keeps_tail_window(self) -> None:
        features = np.arange(10, dtype=np.float32).reshape(5, 2)

        windows, lengths, starts = chunk_feature_sequence(features, window_size=4, window_stride=2)

        self.assertEqual(windows.shape, (2, 4, 2))
        np.testing.assert_array_equal(lengths, np.array([4, 3], dtype=np.int64))
        np.testing.assert_array_equal(starts, np.array([0, 2], dtype=np.int64))
        np.testing.assert_array_equal(windows[0], features[:4])
        np.testing.assert_array_equal(windows[1][:3], features[2:5])
        np.testing.assert_array_equal(windows[1][3], np.zeros(2, dtype=np.float32))

    def test_invalid_stride_raises(self) -> None:
        features = np.ones((4, 2), dtype=np.float32)

        with self.assertRaises(ValueError):
            chunk_feature_sequence(features, window_size=4, window_stride=0)


if __name__ == "__main__":
    unittest.main()
