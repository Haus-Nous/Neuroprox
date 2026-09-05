"""Per-trial signal processing and feature extraction for EMG classification."""

from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from scipy import signal


FEATURE_NAMES = ("rms", "mav", "zero_crossings", "waveform_length", "variance")
PROCESSED_FIELDS = FEATURE_NAMES + ("gesture_label", "session_id", "trial_number")


@dataclass(frozen=True, slots=True)
class TrialSignal:
    """One uninterrupted labeled trial loaded from a session CSV."""

    session_id: str
    gesture_label: str
    trial_number: int
    timestamps: np.ndarray
    raw_values: np.ndarray
    dropped_raw_samples: int = 0


def load_trials(paths: Iterable[Path]) -> list[TrialSignal]:
    """Load session CSVs and group samples without joining trial boundaries."""

    groups: OrderedDict[
        tuple[str, str, int], tuple[list[float], list[float], list[int]]
    ] = OrderedDict()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"timestamp", "raw_value", "gesture_label", "trial_number", "session_id"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path} does not contain the required raw-data columns")
            for line_number, row in enumerate(reader, start=2):
                try:
                    key = (row["session_id"], row["gesture_label"], int(row["trial_number"]))
                    timestamp = datetime.fromisoformat(row["timestamp"]).timestamp()
                    raw_value = float(row["raw_value"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid sample in {path} at line {line_number}") from exc
                timestamps, values, dropped = groups.setdefault(key, ([], [], [0]))
                if not 0 <= raw_value <= 1023:
                    dropped[0] += 1
                    continue
                timestamps.append(timestamp)
                values.append(raw_value)

    return [
        TrialSignal(
            key[0],
            key[1],
            key[2],
            np.asarray(timestamps),
            np.asarray(values),
            dropped[0],
        )
        for key, (timestamps, values, dropped) in groups.items()
    ]


def detect_sample_rate(trials: Iterable[TrialSignal]) -> float:
    """Estimate effective Hz from consecutive timestamps within each trial.

    Serial reads can arrive in bursts, so this uses total sample intervals
    divided by their total elapsed time rather than the median inter-arrival
    delta. Non-positive deltas are discarded.
    """

    interval_count = 0
    elapsed_seconds = 0.0
    for trial in trials:
        positive_deltas = np.diff(trial.timestamps)
        positive_deltas = positive_deltas[positive_deltas > 0]
        interval_count += int(positive_deltas.size)
        elapsed_seconds += float(np.sum(positive_deltas))
    if interval_count == 0 or elapsed_seconds <= 0:
        raise ValueError("Cannot detect sample rate: insufficient increasing timestamps")
    return interval_count / elapsed_seconds


def filter_signal(
    values: np.ndarray,
    sample_rate_hz: float,
    *,
    low_hz: float = 20.0,
    high_hz: float = 450.0,
    notch_hz: float = 50.0,
) -> np.ndarray:
    """Bandpass and notch a single trial without crossing boundaries."""

    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("values must be a non-empty one-dimensional signal")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    nyquist_hz = sample_rate_hz / 2.0
    adjusted_high_hz = min(high_hz, nyquist_hz * 0.95)
    if not 0 < low_hz < adjusted_high_hz:
        raise ValueError("sample rate is too low for the requested bandpass")

    bandpass = signal.butter(
        4,
        (low_hz, adjusted_high_hz),
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    try:
        filtered = signal.sosfiltfilt(bandpass, samples)
    except ValueError:
        filtered = signal.sosfilt(bandpass, samples)

    if notch_hz < nyquist_hz:
        notch_b, notch_a = signal.iirnotch(notch_hz, Q=30.0, fs=sample_rate_hz)
        try:
            filtered = signal.filtfilt(notch_b, notch_a, filtered)
        except ValueError:
            filtered = signal.lfilter(notch_b, notch_a, filtered)

    return filtered


def preprocess_signal(
    values: np.ndarray,
    sample_rate_hz: float,
    *,
    low_hz: float = 20.0,
    high_hz: float = 450.0,
    notch_hz: float = 50.0,
) -> np.ndarray:
    """Bandpass, notch, and rectify a single trial."""

    return np.abs(
        filter_signal(
            values,
            sample_rate_hz,
            low_hz=low_hz,
            high_hz=high_hz,
            notch_hz=notch_hz,
        )
    )


def window_signal(values: np.ndarray, window_size: int, overlap: float = 0.5) -> Iterator[np.ndarray]:
    """Yield full overlapping windows; an incomplete tail is discarded."""

    samples = np.asarray(values)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in the range [0, 1)")
    step = max(1, int(round(window_size * (1.0 - overlap))))
    for start in range(0, samples.size - window_size + 1, step):
        yield samples[start : start + window_size]


def extract_features(
    values: np.ndarray,
    *,
    zero_crossing_values: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Calculate five time-domain features for one window.

    ``values`` is normally rectified. A matching pre-rectification window can
    be supplied for meaningful zero crossings; otherwise ``values`` is used.
    """

    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("values must be a non-empty one-dimensional window")
    crossing_samples = (
        samples
        if zero_crossing_values is None
        else np.asarray(zero_crossing_values, dtype=float)
    )
    if crossing_samples.shape != samples.shape:
        raise ValueError("zero_crossing_values must have the same shape as values")
    return {
        "rms": float(np.sqrt(np.mean(np.square(samples)))),
        "mav": float(np.mean(np.abs(samples))),
        "zero_crossings": int(np.count_nonzero(np.diff(np.signbit(crossing_samples)))),
        "waveform_length": float(np.sum(np.abs(np.diff(samples)))),
        "variance": float(np.var(samples)),
    }


def build_feature_rows(
    trials: Iterable[TrialSignal],
    sample_rate_hz: float,
    *,
    window_duration_seconds: float = 0.175,
    overlap: float = 0.5,
) -> list[dict[str, object]]:
    """Run preprocessing, windowing, and feature extraction per trial."""

    window_size = max(1, int(round(window_duration_seconds * sample_rate_hz)))
    rows: list[dict[str, object]] = []
    for trial in trials:
        signed = filter_signal(trial.raw_values, sample_rate_hz)
        rectified_windows = window_signal(np.abs(signed), window_size, overlap)
        signed_windows = window_signal(signed, window_size, overlap)
        for rectified_window, signed_window in zip(rectified_windows, signed_windows):
            row: dict[str, object] = extract_features(
                rectified_window, zero_crossing_values=signed_window
            )
            row.update(
                gesture_label=trial.gesture_label,
                session_id=trial.session_id,
                trial_number=trial.trial_number,
            )
            rows.append(row)
    return rows


def normalize_by_session_open_baseline(
    rows: Iterable[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Normalize each session to its mean OPEN_PALM window RMS.

    Amplitude-linear features scale by baseline RMS, variance scales by its
    square, and the dimensionless zero-crossing count is preserved.
    """

    materialized = [dict(row) for row in rows]
    open_rms: dict[str, list[float]] = {}
    for row in materialized:
        if row["gesture_label"] == "OPEN_PALM":
            open_rms.setdefault(str(row["session_id"]), []).append(float(row["rms"]))
    sessions = {str(row["session_id"]) for row in materialized}
    missing = sessions - open_rms.keys()
    if missing:
        raise ValueError(f"No OPEN_PALM baseline windows for session(s): {sorted(missing)}")
    baselines = {session: float(np.mean(values)) for session, values in open_rms.items()}
    if any(baseline <= 0 for baseline in baselines.values()):
        raise ValueError("OPEN_PALM baseline RMS must be positive")

    for row in materialized:
        baseline = baselines[str(row["session_id"])]
        row["rms"] = float(row["rms"]) / baseline
        row["mav"] = float(row["mav"]) / baseline
        row["waveform_length"] = float(row["waveform_length"]) / baseline
        row["variance"] = float(row["variance"]) / (baseline * baseline)
    return materialized, baselines


def write_feature_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    """Write feature rows using the stable processed-data column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROCESSED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
