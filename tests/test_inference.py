import unittest

import numpy as np

from emg.inference import Debouncer, normalized_feature_vector


class InferenceTests(unittest.TestCase):
    def test_debouncer_emits_once_per_stable_transition(self) -> None:
        debouncer = Debouncer(3)
        self.assertIsNone(debouncer.update("OPEN_PALM", 0.8))
        self.assertIsNone(debouncer.update("OPEN_PALM", 0.9))
        result = debouncer.update("OPEN_PALM", 1.0)
        self.assertEqual(result[0], "OPEN_PALM")
        self.assertAlmostEqual(result[1], 0.9)
        self.assertIsNone(debouncer.update("OPEN_PALM", 0.9))
        self.assertIsNone(debouncer.update("CLOSED_FIST", 0.8))
        self.assertIsNone(debouncer.update("CLOSED_FIST", 0.8))
        result = debouncer.update("CLOSED_FIST", 0.8)
        self.assertEqual(result[0], "CLOSED_FIST")
        self.assertAlmostEqual(result[1], 0.8)

    def test_live_feature_normalization_matches_training_formula(self) -> None:
        vector = normalized_feature_vector(np.array([-2.0, 2.0, -2.0, 2.0]), 2.0)
        np.testing.assert_allclose(vector[0], [1.0, 1.0, 3.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
