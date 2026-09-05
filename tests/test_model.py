import unittest

import numpy as np

from emg.model import FeatureDataset, cross_validate_candidates, split_by_trial


class TrialSplitTests(unittest.TestCase):
    @staticmethod
    def make_dataset() -> FeatureDataset:
        keys = []
        labels = []
        for label in ("OPEN_PALM", "CLOSED_FIST"):
            for trial in range(1, 6):
                keys.extend([("session", label, trial)] * 3)
                labels.extend([label] * 3)
        features = np.arange(len(labels) * 5, dtype=float).reshape(len(labels), 5)
        return FeatureDataset(features, np.asarray(labels), tuple(keys))

    def test_split_has_no_trial_leakage_and_is_stratified(self) -> None:
        dataset = self.make_dataset()

        split = split_by_trial(dataset, test_size=0.2, random_state=42)

        self.assertFalse(split.train_trials & split.test_trials)
        self.assertEqual({key[1] for key in split.train_trials}, set(dataset.labels))
        self.assertEqual({key[1] for key in split.test_trials}, set(dataset.labels))
        self.assertEqual(len(split.train_trials), 8)
        self.assertEqual(len(split.test_trials), 2)

    def test_cross_validation_reports_five_folds_for_each_candidate(self) -> None:
        results = cross_validate_candidates(self.make_dataset(), random_state=42)
        self.assertEqual({result.name for result in results}, {"SVM (RBF)", "Random Forest"})
        self.assertTrue(all(len(result.folds) == 5 for result in results))

    def test_each_cv_fold_contains_every_session(self) -> None:
        keys = []
        labels = []
        for session in ("session-1", "session-2"):
            for label in ("OPEN_PALM", "CLOSED_FIST"):
                for trial in range(1, 6):
                    keys.extend([(session, label, trial)] * 2)
                    labels.extend([label] * 2)
        dataset = FeatureDataset(
            np.arange(len(labels) * 5, dtype=float).reshape(len(labels), 5),
            np.asarray(labels),
            tuple(keys),
        )

        results = cross_validate_candidates(dataset, random_state=42)

        self.assertTrue(
            all(
                set(fold.test_sessions) == {"session-1", "session-2"}
                for result in results
                for fold in result.folds
            )
        )


if __name__ == "__main__":
    unittest.main()
