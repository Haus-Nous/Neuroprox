#!/usr/bin/env python3
"""Interactively record a randomized, labeled NeuroProx EMG session."""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Make the project packages importable when invoked as `python scripts/...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emg.acquisition import (
    GESTURE_LABELS,
    create_session_file,
    open_serial,
    record_trial,
)


GESTURES = GESTURE_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", required=True, help="Arduino port, e.g. /dev/cu.usbmodem1101"
    )
    parser.add_argument(
        "--baud", type=int, default=115200, help="serial baud rate (default: 115200)"
    )
    parser.add_argument(
        "--trials-per-gesture",
        type=int,
        default=20,
        help="number of trials for each gesture (default: 20; 40 total)",
    )
    parser.add_argument(
        "--record-seconds", type=float, default=4.0, help="hold/record duration"
    )
    parser.add_argument(
        "--countdown-seconds", type=int, default=3, help="countdown before recording"
    )
    parser.add_argument(
        "--rest-seconds", type=float, default=2.0, help="rest between trials"
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=2.0,
        help="seconds to wait after opening the Uno (it may reset)",
    )
    args = parser.parse_args()
    if args.baud <= 0:
        parser.error("--baud must be greater than zero")
    if args.trials_per_gesture <= 0:
        parser.error("--trials-per-gesture must be greater than zero")
    if args.record_seconds <= 0 or args.rest_seconds < 0 or args.startup_delay < 0:
        parser.error("durations must be positive (rest/startup delay may be zero)")
    if args.countdown_seconds < 0:
        parser.error("--countdown-seconds may not be negative")
    return args


def make_trial_order(trials_per_gesture: int) -> list[tuple[str, int]]:
    """Build and randomize the full session, retaining per-gesture trial IDs."""

    trials = [
        (gesture, trial_number)
        for gesture in GESTURES
        for trial_number in range(1, trials_per_gesture + 1)
    ]
    random.SystemRandom().shuffle(trials)
    return trials


def countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"  Starting in {remaining}...", flush=True)
        time.sleep(1)


def print_counts(completed: Counter[str], target: int) -> None:
    status = " | ".join(
        f"{gesture}: {completed[gesture]}/{target}" for gesture in GESTURES
    )
    print(f"Completed — {status}")


def print_summary(
    completed: Counter[str], samples: Counter[str], malformed: int
) -> None:
    print("\nSession summary")
    print(f"{'Gesture':<16} {'Trials':>8} {'Samples':>10}")
    print(f"{'-' * 16} {'-' * 8} {'-' * 10}")
    for gesture in GESTURES:
        print(f"{gesture:<16} {completed[gesture]:>8} {samples[gesture]:>10}")
    print(f"Malformed serial lines skipped: {malformed}")


def main() -> int:
    args = parse_args()
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path, output_handle, writer = create_session_file(
        PROJECT_ROOT / "data" / "raw", session_id
    )
    completed: Counter[str] = Counter()
    samples: Counter[str] = Counter()
    malformed_total = 0
    invalid_adc_total = 0

    print(f"Session: {session_id}")
    print(f"Output:  {output_path}")
    print(f"Opening {args.port} at {args.baud} baud...")

    connection = None
    try:
        connection = open_serial(args.port, args.baud)
        time.sleep(args.startup_delay)
        trial_order = make_trial_order(args.trials_per_gesture)

        for session_index, (gesture, trial_number) in enumerate(trial_order, start=1):
            print(
                f"\nTrial {session_index}/{len(trial_order)}: prepare {gesture} "
                f"(gesture trial {trial_number}/{args.trials_per_gesture})"
            )
            input("Press Enter when ready...")
            countdown(args.countdown_seconds)
            print(
                f"  HOLD {gesture} — recording for {args.record_seconds:g} seconds",
                flush=True,
            )

            result = record_trial(
                connection,
                writer,
                duration_seconds=args.record_seconds,
                gesture_label=gesture,
                trial_number=trial_number,
                session_id=session_id,
            )
            output_handle.flush()
            completed[gesture] += 1
            samples[gesture] += result.sample_count
            malformed_total += result.malformed_line_count
            invalid_adc_total += result.invalid_adc_count
            print(f"  Recorded {result.sample_count} samples. Relax.")
            print_counts(completed, args.trials_per_gesture)

            if session_index < len(trial_order) and args.rest_seconds:
                time.sleep(args.rest_seconds)

    except KeyboardInterrupt:
        print(
            "\nSession stopped by user; completed trial data has been saved.",
            file=sys.stderr,
        )
    except RuntimeError as exc:
        print(f"\nAcquisition error: {exc}", file=sys.stderr)
        print(
            "Check the USB cable, port name, Arduino upload, and permissions.",
            file=sys.stderr,
        )
        return 1
    finally:
        output_handle.close()
        if connection is not None:
            try:
                connection.close()
            except Exception as exc:
                print(
                    f"Warning: could not close serial port cleanly: {exc}",
                    file=sys.stderr,
                )

    print_summary(completed, samples, malformed_total)
    print(f"Out-of-range ADC frames dropped: {invalid_adc_total}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
