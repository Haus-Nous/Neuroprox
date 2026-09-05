"""Calibrated real-time EMG inference for the NeuroProx command bus."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

from core import Command, CommandBus, CommandMessage, CommandSource
from emg.acquisition import SerialLike, parse_adc_line
from emg.processing import FEATURE_NAMES, extract_features, filter_signal, window_signal


LABEL_TO_COMMAND = {
    "OPEN_PALM": Command.OPEN,
    "CLOSED_FIST": Command.CLOSE,
}


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    baseline_rms: float
    sample_rate_hz: float
    sample_count: int
    window_count: int
    dropped_frames: int


@dataclass(frozen=True, slots=True)
class LoadedInferenceModel:
    estimator: object
    feature_names: tuple[str, ...]
    metadata: dict[str, object]


class Debouncer:
    """Emit only after a label wins N consecutive windows and changes state."""

    def __init__(self, required_windows: int = 3) -> None:
        if required_windows <= 0:
            raise ValueError("required_windows must be positive")
        self.required_windows = required_windows
        self._recent: deque[tuple[str, float]] = deque(maxlen=required_windows)
        self._last_emitted: str | None = None

    def update(self, label: str, confidence: float) -> tuple[str, float] | None:
        self._recent.append((label, confidence))
        if len(self._recent) < self.required_windows:
            return None
        labels = {item[0] for item in self._recent}
        if len(labels) != 1 or label == self._last_emitted:
            return None
        self._last_emitted = label
        return label, float(np.mean([item[1] for item in self._recent]))


def load_inference_model(path: Path) -> LoadedInferenceModel:
    """Load and validate a probability-capable normalized v3 bundle."""

    bundle = joblib.load(path)
    required = {
        "estimator",
        "feature_names",
        "labels",
        "model_version",
        "feature_normalization",
    }
    missing = required - bundle.keys()
    if missing:
        raise ValueError(f"Model bundle is missing metadata: {sorted(missing)}")
    estimator = bundle["estimator"]
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("Model does not expose predict_proba; retrain with calibration")
    if bundle["feature_normalization"] != "per-session-open-palm-rms":
        raise ValueError("Model does not use the required OPEN_PALM normalization")
    if tuple(bundle["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Model feature order does not match the processing pipeline")
    return LoadedInferenceModel(estimator, tuple(bundle["feature_names"]), bundle)


def read_valid_sample(connection: SerialLike) -> tuple[int | None, bool]:
    """Read one frame, returning ``(sample, dropped)``."""

    try:
        raw_line = connection.readline()
    except (OSError, IOError) as exc:
        raise RuntimeError(f"Serial connection lost: {exc}") from exc
    if not raw_line:
        return None, False
    try:
        return parse_adc_line(raw_line), False
    except (ValueError, OverflowError):
        if not raw_line.endswith(b"\n"):
            try:
                connection.reset_input_buffer()
            except (OSError, IOError) as exc:
                raise RuntimeError(f"Could not resynchronize serial input: {exc}") from exc
        return None, True


def collect_calibration_samples(
    connection: SerialLike,
    duration_seconds: float,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Collect timestamped, validated ADC samples for calibration."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    connection.reset_input_buffer()
    started = clock()
    timestamps: list[float] = []
    samples: list[int] = []
    dropped = 0
    while clock() - started < duration_seconds:
        sample, was_dropped = read_valid_sample(connection)
        dropped += int(was_dropped)
        if sample is not None:
            timestamps.append(clock())
            samples.append(sample)
    if len(samples) < 2:
        raise RuntimeError("Calibration captured too few valid samples")
    return np.asarray(samples, dtype=float), np.asarray(timestamps), dropped


def calibrate(
    connection: SerialLike,
    *,
    duration_seconds: float = 4.0,
    window_duration_seconds: float = 0.175,
) -> CalibrationResult:
    """Collect OPEN_PALM data and calculate the live normalization baseline."""

    samples, timestamps, dropped = collect_calibration_samples(
        connection, duration_seconds
    )
    elapsed = float(timestamps[-1] - timestamps[0])
    if elapsed <= 0:
        raise RuntimeError("Cannot detect calibration sample rate")
    sample_rate_hz = (samples.size - 1) / elapsed
    window_size = max(1, int(round(window_duration_seconds * sample_rate_hz)))
    signed = filter_signal(samples, sample_rate_hz)
    rms_values = [
        float(extract_features(window)["rms"])
        for window in window_signal(np.abs(signed), window_size, overlap=0.5)
    ]
    if not rms_values:
        raise RuntimeError("Calibration did not contain one complete feature window")
    baseline = float(np.mean(rms_values))
    if not np.isfinite(baseline) or baseline <= 0:
        raise RuntimeError("Calibration produced an invalid OPEN_PALM baseline")
    return CalibrationResult(
        baseline,
        sample_rate_hz,
        int(samples.size),
        len(rms_values),
        dropped,
    )


def normalized_feature_vector(
    signed_window: np.ndarray,
    baseline_rms: float,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> np.ndarray:
    """Extract and normalize one live window exactly like v3 training data."""

    if baseline_rms <= 0:
        raise ValueError("baseline_rms must be positive")
    features = extract_features(
        np.abs(signed_window), zero_crossing_values=signed_window
    )
    features["rms"] = float(features["rms"]) / baseline_rms
    features["mav"] = float(features["mav"]) / baseline_rms
    features["waveform_length"] = float(features["waveform_length"]) / baseline_rms
    features["variance"] = float(features["variance"]) / (baseline_rms**2)
    return np.asarray([[float(features[name]) for name in feature_names]])


def run_live_inference(
    connection: SerialLike,
    model: LoadedInferenceModel,
    calibration: CalibrationResult,
    bus: CommandBus,
    *,
    debounce_windows: int = 3,
    window_duration_seconds: float = 0.175,
) -> None:
    """Read, classify, debounce, and emit commands until interrupted."""

    window_size = max(1, int(round(window_duration_seconds * calibration.sample_rate_hz)))
    step_size = max(1, int(round(window_size * 0.5)))
    buffer: deque[float] = deque()
    debouncer = Debouncer(debounce_windows)
    classes = list(model.estimator.classes_)

    while True:
        sample, _ = read_valid_sample(connection)
        if sample is None:
            continue
        buffer.append(float(sample))
        if len(buffer) < window_size:
            continue

        raw_window = np.asarray(list(buffer)[:window_size])
        signed_window = filter_signal(raw_window, calibration.sample_rate_hz)
        vector = normalized_feature_vector(
            signed_window, calibration.baseline_rms, model.feature_names
        )
        probabilities = model.estimator.predict_proba(vector)[0]
        prediction = str(model.estimator.predict(vector)[0])
        confidence = float(probabilities[classes.index(prediction)])
        stable = debouncer.update(prediction, confidence)
        if stable is not None:
            label, stable_confidence = stable
            bus.emit(
                CommandMessage.create(
                    LABEL_TO_COMMAND[label], stable_confidence, CommandSource.EMG
                )
            )
        for _ in range(min(step_size, len(buffer))):
            buffer.popleft()
