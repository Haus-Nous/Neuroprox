"""Leakage-safe training and evaluation for the binary EMG classifier."""

from __future__ import annotations

import csv
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FEATURE_NAMES = ("rms", "mav", "zero_crossings", "waveform_length", "variance")
LABELS = ("CLOSED_FIST", "OPEN_PALM")
TrialKey = tuple[str, str, int]


@dataclass(frozen=True, slots=True)
class FeatureDataset:
    features: np.ndarray
    labels: np.ndarray
    trial_keys: tuple[TrialKey, ...]


@dataclass(frozen=True, slots=True)
class TrialSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_trials: frozenset[TrialKey]
    test_trials: frozenset[TrialKey]


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    name: str
    estimator: object
    accuracy: float
    confusion: np.ndarray
    report: dict[str, object]

    @property
    def macro_f1(self) -> float:
        return float(self.report["macro avg"]["f1-score"])


@dataclass(frozen=True, slots=True)
class FoldScore:
    fold: int
    accuracy: float
    macro_f1: float
    confusion: np.ndarray
    test_sessions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CrossValidationEvaluation:
    name: str
    folds: tuple[FoldScore, ...]

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean([fold.accuracy for fold in self.folds]))

    @property
    def std_accuracy(self) -> float:
        return float(np.std([fold.accuracy for fold in self.folds], ddof=1))

    @property
    def mean_macro_f1(self) -> float:
        return float(np.mean([fold.macro_f1 for fold in self.folds]))

    @property
    def std_macro_f1(self) -> float:
        return float(np.std([fold.macro_f1 for fold in self.folds], ddof=1))


def load_feature_csvs(paths: Iterable[Path]) -> FeatureDataset:
    """Load and combine one or more processed feature CSV files."""

    features: list[list[float]] = []
    labels: list[str] = []
    trial_keys: list[TrialKey] = []
    required = set(FEATURE_NAMES) | {"gesture_label", "session_id", "trial_number"}

    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path} does not contain the required feature columns")
            for line_number, row in enumerate(reader, start=2):
                try:
                    label = row["gesture_label"]
                    vector = [float(row[name]) for name in FEATURE_NAMES]
                    key = (row["session_id"], label, int(row["trial_number"]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid feature row in {path} at line {line_number}") from exc
                if label not in LABELS:
                    raise ValueError(f"Unsupported gesture label {label!r} in {path}")
                features.append(vector)
                labels.append(label)
                trial_keys.append(key)

    if not features:
        raise ValueError("No feature rows were loaded")
    return FeatureDataset(
        features=np.asarray(features, dtype=float),
        labels=np.asarray(labels),
        trial_keys=tuple(trial_keys),
    )


def split_by_trial(
    dataset: FeatureDataset,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrialSplit:
    """Create a stratified split with every trial wholly on one side."""

    trial_labels: OrderedDict[TrialKey, str] = OrderedDict()
    for key in dataset.trial_keys:
        trial_labels[key] = key[1]
    keys = list(trial_labels)
    train_keys, test_keys = train_test_split(
        keys,
        test_size=test_size,
        random_state=random_state,
        stratify=[trial_labels[key] for key in keys],
    )
    train_trials = frozenset(train_keys)
    test_trials = frozenset(test_keys)
    train_indices = np.asarray(
        [index for index, key in enumerate(dataset.trial_keys) if key in train_trials]
    )
    test_indices = np.asarray(
        [index for index, key in enumerate(dataset.trial_keys) if key in test_trials]
    )
    if train_trials & test_trials:
        raise AssertionError("trial leakage detected between train and test")
    return TrialSplit(train_indices, test_indices, train_trials, test_trials)


def candidate_models(random_state: int = 42) -> dict[str, object]:
    """Return fresh candidate estimators with deterministic configuration."""

    return {
        "SVM (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", random_state=random_state)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def train_and_evaluate(
    dataset: FeatureDataset,
    split: TrialSplit,
    *,
    random_state: int = 42,
) -> list[ModelEvaluation]:
    """Fit candidates on train windows and evaluate on held-out trials."""

    x_train = dataset.features[split.train_indices]
    y_train = dataset.labels[split.train_indices]
    x_test = dataset.features[split.test_indices]
    y_test = dataset.labels[split.test_indices]
    evaluations: list[ModelEvaluation] = []
    for name, estimator in candidate_models(random_state).items():
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_test)
        report = classification_report(
            y_test,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        evaluations.append(
            ModelEvaluation(
                name=name,
                estimator=estimator,
                accuracy=float(accuracy_score(y_test, predictions)),
                confusion=confusion_matrix(y_test, predictions, labels=LABELS),
                report=report,
            )
        )
    return evaluations


def choose_best(evaluations: Iterable[ModelEvaluation]) -> ModelEvaluation:
    """Select by held-out accuracy, then macro F1, preserving candidate order."""

    candidates = list(evaluations)
    if not candidates:
        raise ValueError("No model evaluations supplied")
    return max(candidates, key=lambda result: (result.accuracy, result.macro_f1))


def cross_validate_candidates(
    dataset: FeatureDataset,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[CrossValidationEvaluation]:
    """Evaluate candidates with stratified folds over whole trials."""

    trial_labels: OrderedDict[TrialKey, str] = OrderedDict()
    for key in dataset.trial_keys:
        trial_labels[key] = key[1]
    keys = list(trial_labels)
    labels = np.asarray([trial_labels[key] for key in keys])
    strata = np.asarray([f"{key[0]}::{key[1]}" for key in keys])
    if min(Counter(strata).values()) < n_splits:
        raise ValueError(
            f"Each session/gesture stratum needs at least {n_splits} trials for CV"
        )

    scores: dict[str, list[FoldScore]] = {
        name: [] for name in candidate_models(random_state)
    }
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_number, (train_trial_indices, test_trial_indices) in enumerate(
        folds.split(keys, strata), start=1
    ):
        train_trials = {keys[index] for index in train_trial_indices}
        test_trials = {keys[index] for index in test_trial_indices}
        if train_trials & test_trials:
            raise AssertionError("trial leakage detected between CV train and test")
        train_indices = np.asarray(
            [i for i, key in enumerate(dataset.trial_keys) if key in train_trials]
        )
        test_indices = np.asarray(
            [i for i, key in enumerate(dataset.trial_keys) if key in test_trials]
        )
        for name, estimator in candidate_models(random_state).items():
            estimator.fit(dataset.features[train_indices], dataset.labels[train_indices])
            predictions = estimator.predict(dataset.features[test_indices])
            scores[name].append(
                FoldScore(
                    fold=fold_number,
                    accuracy=float(accuracy_score(dataset.labels[test_indices], predictions)),
                    macro_f1=float(
                        f1_score(dataset.labels[test_indices], predictions, average="macro")
                    ),
                    confusion=confusion_matrix(
                        dataset.labels[test_indices], predictions, labels=LABELS
                    ),
                    test_sessions=tuple(sorted({key[0] for key in test_trials})),
                )
            )
    return [
        CrossValidationEvaluation(name=name, folds=tuple(fold_scores))
        for name, fold_scores in scores.items()
    ]


def choose_best_cv(
    evaluations: Iterable[CrossValidationEvaluation],
) -> CrossValidationEvaluation:
    """Select by mean CV accuracy, then mean macro F1."""

    candidates = list(evaluations)
    if not candidates:
        raise ValueError("No cross-validation evaluations supplied")
    return max(
        candidates,
        key=lambda result: (result.mean_accuracy, result.mean_macro_f1),
    )


def fit_candidate_on_all(
    name: str,
    dataset: FeatureDataset,
    *,
    random_state: int = 42,
    calibrate_probabilities: bool = False,
) -> object:
    """Fit a fresh selected candidate on every available feature window."""

    if name == "SVM (RBF)" and calibrate_probabilities:
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        SVC(kernel="rbf", random_state=random_state),
                        method="sigmoid",
                        cv=5,
                    ),
                ),
            ]
        )
    else:
        try:
            estimator = candidate_models(random_state)[name]
        except KeyError as exc:
            raise ValueError(f"Unknown candidate model: {name}") from exc
    estimator.fit(dataset.features, dataset.labels)
    return estimator


def class_split_counts(
    dataset: FeatureDataset, split: TrialSplit
) -> dict[str, dict[str, Counter[str]]]:
    """Count trials and windows per label on both sides of the split."""

    return {
        "train": {
            "trials": Counter(key[1] for key in split.train_trials),
            "windows": Counter(dataset.labels[split.train_indices]),
        },
        "test": {
            "trials": Counter(key[1] for key in split.test_trials),
            "windows": Counter(dataset.labels[split.test_indices]),
        },
    }


def save_model_bundle(
    classifier_name: str,
    estimator: object,
    dataset: FeatureDataset,
    path: Path,
    *,
    data_cleaned: bool = False,
    cv_evaluations: Iterable[CrossValidationEvaluation] = (),
    model_version: str = "v1",
    feature_normalization: str = "none",
    probability_calibrated: bool = False,
) -> None:
    """Persist the winning estimator and the metadata required for inference."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_version": model_version,
            "classifier_name": classifier_name,
            "estimator": estimator,
            "feature_names": FEATURE_NAMES,
            "labels": LABELS,
            "training_sessions": sorted({key[0] for key in dataset.trial_keys}),
            "data_cleaned": data_cleaned,
            "cleaning_policy": (
                "raw ADC samples outside the Arduino Uno range 0-1023 were dropped"
                if data_cleaned
                else None
            ),
            "evaluation_method": "5-fold trial-level CV",
            "trained_on_all_trials": True,
            "training_trial_count": len(set(dataset.trial_keys)),
            "training_window_count": int(dataset.features.shape[0]),
            "feature_normalization": feature_normalization,
            "probability_calibrated": probability_calibrated,
            "cross_validation": {
                result.name: {
                    "folds": [
                        {
                            "fold": fold.fold,
                            "accuracy": fold.accuracy,
                            "macro_f1": fold.macro_f1,
                            "confusion_matrix": fold.confusion.tolist(),
                            "test_sessions": list(fold.test_sessions),
                        }
                        for fold in result.folds
                    ],
                    "mean_accuracy": result.mean_accuracy,
                    "std_accuracy": result.std_accuracy,
                    "mean_macro_f1": result.mean_macro_f1,
                    "std_macro_f1": result.std_macro_f1,
                }
                for result in cv_evaluations
            },
        },
        path,
    )
