#!/usr/bin/env python3
"""Train, compare, and save the best binary EMG feature classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emg.model import (
    choose_best_cv,
    cross_validate_candidates,
    fit_candidate_on_all,
    load_feature_csvs,
    save_model_bundle,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "features_session_20260904T081307Z.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "emg_binary_v1_session_20260904T081307Z.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", nargs="*", type=Path, help="processed feature CSV files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-version", default="v1", help="artifact version metadata")
    parser.add_argument(
        "--feature-normalization",
        default="none",
        help="feature-normalization metadata stored with the artifact",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="run CV selection without fitting or saving a deployment artifact",
    )
    parser.add_argument(
        "--calibrate-probabilities",
        action="store_true",
        help="calibrate the final selected estimator to expose predict_proba",
    )
    parser.add_argument(
        "--data-cleaned",
        action="store_true",
        help="mark the artifact as trained from ADC-range-cleaned features",
    )
    args = parser.parse_args()
    return args


def print_cv_evaluation(result) -> None:
    print(f"\n{result.name}")
    print(f"{'Fold':>4} {'Accuracy':>10} {'Macro F1':>10}")
    for fold in result.folds:
        print(
            f"{fold.fold:>4} {fold.accuracy:>10.4f} {fold.macro_f1:>10.4f}"
        )
        print(
            f"     Confusion: {fold.confusion.tolist()} | "
            f"Test sessions: {', '.join(fold.test_sessions)}"
        )
    print(
        f"Mean {result.mean_accuracy:>10.4f} {result.mean_macro_f1:>10.4f}"
    )
    print(
        f"Std  {result.std_accuracy:>10.4f} {result.std_macro_f1:>10.4f}"
    )


def main() -> int:
    args = parse_args()
    paths = args.features or [DEFAULT_INPUT]
    try:
        dataset = load_feature_csvs(paths)
        evaluations = cross_validate_candidates(dataset, random_state=args.random_state)
        best = choose_best_cv(evaluations)
        if not args.evaluate_only:
            final_estimator = fit_candidate_on_all(
                best.name,
                dataset,
                random_state=args.random_state,
                calibrate_probabilities=args.calibrate_probabilities,
            )
            save_model_bundle(
                best.name,
                final_estimator,
                dataset,
                args.output,
                data_cleaned=args.data_cleaned,
                cv_evaluations=evaluations,
                model_version=args.model_version,
                feature_normalization=args.feature_normalization,
                probability_calibrated=args.calibrate_probabilities,
            )
    except (OSError, ValueError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    print("NeuroProx binary EMG model training")
    print(f"Feature files: {len(paths)}")
    total_trials = len(set(dataset.trial_keys))
    print(f"Total trials: {total_trials}")
    print(f"Total windows: {dataset.features.shape[0]}")
    print("Evaluation: 5-fold stratified trial-level CV (no window leakage)")
    for evaluation in evaluations:
        print_cv_evaluation(evaluation)
    print(f"\nSelected model: {best.name}")
    print("Selection rule: mean CV accuracy, then mean macro F1")
    if args.evaluate_only:
        print("Final fit: skipped (evaluation only)")
    else:
        print(
            f"Final fit: all {total_trials} trials / {dataset.features.shape[0]} windows (100%)"
        )
    print(f"Cleaned-data provenance: {args.data_cleaned}")
    print(f"Model version: {args.model_version}")
    print(f"Feature normalization: {args.feature_normalization}")
    print(f"Probability calibrated: {args.calibrate_probabilities}")
    print(f"Saved model: {args.output if not args.evaluate_only else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
