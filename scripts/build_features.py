#!/usr/bin/env python3
"""Build window-level EMG features from one or more raw sessions."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emg.processing import (
    build_feature_rows,
    detect_sample_rate,
    load_trials,
    normalize_by_session_open_baseline,
    write_feature_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sessions", nargs="*", type=Path,
        help="raw session CSVs (default: every session_*.csv in data/raw)",
    )
    parser.add_argument("--output", type=Path, help="processed CSV output path")
    parser.add_argument("--window-ms", type=float, default=175.0, help="window duration")
    parser.add_argument(
        "--normalize-open-baseline",
        action="store_true",
        help="normalize each session using its own OPEN_PALM mean RMS",
    )
    args = parser.parse_args()
    if args.window_ms <= 0:
        parser.error("--window-ms must be positive")
    return args


def main() -> int:
    args = parse_args()
    paths = args.sessions or sorted((PROJECT_ROOT / "data" / "raw").glob("session_*.csv"))
    if not paths:
        print("No raw session CSV files found.", file=sys.stderr)
        return 1
    missing = [path for path in paths if not path.is_file()]
    if missing:
        print(f"Input file not found: {missing[0]}", file=sys.stderr)
        return 1

    output = args.output or (
        PROJECT_ROOT / "data" / "processed"
        / f"features_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    try:
        trials = load_trials(paths)
        sample_rate_hz = detect_sample_rate(trials)
        rows = build_feature_rows(
            trials, sample_rate_hz,
            window_duration_seconds=args.window_ms / 1000.0,
            overlap=0.5,
        )
        baselines = {}
        if args.normalize_open_baseline:
            rows, baselines = normalize_by_session_open_baseline(rows)
        write_feature_csv(rows, output)
    except (OSError, ValueError) as exc:
        print(f"Feature build failed: {exc}", file=sys.stderr)
        return 1

    counts = Counter(str(row["gesture_label"]) for row in rows)
    print("Feature build complete")
    print(f"Input sessions: {len(paths)}")
    print(
        "Session IDs: "
        + ", ".join(sorted({trial.session_id for trial in trials}))
    )
    print(f"Trials processed: {len(trials)}")
    print("Dropped raw samples by trial:")
    for trial in trials:
        print(
            f"  {trial.session_id} {trial.gesture_label} "
            f"trial {trial.trial_number}: {trial.dropped_raw_samples}"
        )
    print(f"Total dropped raw samples: {sum(t.dropped_raw_samples for t in trials)}")
    print(f"Detected sample rate: {sample_rate_hz:.2f} Hz")
    print(f"Window: {args.window_ms:g} ms, 50% overlap")
    if baselines:
        print("Per-session OPEN_PALM RMS baselines:")
        for session_id, baseline in sorted(baselines.items()):
            print(f"  {session_id}: {baseline:.6f}")
        print("Session baseline normalization: applied")
    else:
        print("Session baseline normalization: not applied")
    print(f"Total windows: {len(rows)}")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
