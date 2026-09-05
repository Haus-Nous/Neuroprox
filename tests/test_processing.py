import unittest

import numpy as np

from pathlib import Path
import tempfile

from emg.processing import (
    extract_features,
    load_trials,
    normalize_by_session_open_baseline,
    preprocess_signal,
    window_signal,
)


class ProcessingTests(unittest.TestCase):
    def test_session_open_baseline_normalization(self) -> None:
        rows = [
            {"session_id": "s", "gesture_label": "OPEN_PALM", "rms": 2.0,
             "mav": 1.0, "zero_crossings": 4, "waveform_length": 6.0, "variance": 4.0},
            {"session_id": "s", "gesture_label": "CLOSED_FIST", "rms": 4.0,
             "mav": 2.0, "zero_crossings": 5, "waveform_length": 8.0, "variance": 16.0},
        ]
        normalized, baselines = normalize_by_session_open_baseline(rows)
        self.assertEqual(baselines, {"s": 2.0})
        self.assertEqual(normalized[1]["rms"], 2.0)
        self.assertEqual(normalized[1]["mav"], 1.0)
        self.assertEqual(normalized[1]["zero_crossings"], 5)
        self.assertEqual(normalized[1]["waveform_length"], 4.0)
        self.assertEqual(normalized[1]["variance"], 4.0)

    def test_loader_drops_out_of_range_adc_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            path.write_text(
                "timestamp,raw_value,gesture_label,trial_number,session_id\n"
                "2026-01-01T00:00:00+00:00,500,OPEN_PALM,1,s\n"
                "2026-01-01T00:00:00.001+00:00,50515,OPEN_PALM,1,s\n"
                "2026-01-01T00:00:00.002+00:00,501,OPEN_PALM,1,s\n"
            )
            trial = load_trials([path])[0]
        np.testing.assert_array_equal(trial.raw_values, [500.0, 501.0])
        self.assertEqual(trial.dropped_raw_samples, 1)

    def test_filter_accepts_short_signal(self) -> None:
        result = preprocess_signal(np.array([1.0, 2.0, 1.0, 0.0]), 1300.0)
        self.assertEqual(result.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result >= 0))

    def test_constant_signal_features(self) -> None:
        features = extract_features(np.array([3.0, 3.0, 3.0, 3.0]))
        self.assertAlmostEqual(features["rms"], 3.0)
        self.assertAlmostEqual(features["mav"], 3.0)
        self.assertEqual(features["zero_crossings"], 0)
        self.assertAlmostEqual(features["waveform_length"], 0.0)
        self.assertAlmostEqual(features["variance"], 0.0)

    def test_known_wave_features(self) -> None:
        signed = np.array([-1.0, 1.0, -1.0, 1.0])
        features = extract_features(np.abs(signed), zero_crossing_values=signed)
        self.assertAlmostEqual(features["rms"], 1.0)
        self.assertAlmostEqual(features["mav"], 1.0)
        self.assertEqual(features["zero_crossings"], 3)
        self.assertAlmostEqual(features["waveform_length"], 0.0)
        self.assertAlmostEqual(features["variance"], 0.0)

    def test_window_count_with_half_overlap(self) -> None:
        windows = list(window_signal(np.arange(100), window_size=20, overlap=0.5))
        self.assertEqual(len(windows), 9)
        np.testing.assert_array_equal(windows[0], np.arange(20))
        np.testing.assert_array_equal(windows[-1], np.arange(80, 100))


if __name__ == "__main__":
    unittest.main()
