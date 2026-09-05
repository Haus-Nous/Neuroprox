"""Serial acquisition primitives for raw, single-channel EMG recordings.

The Arduino is expected to send exactly one integer ADC reading per line. This
module deliberately records raw values only; filtering and ML belong elsewhere.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TextIO


class SerialLike(Protocol):
    """Small subset of ``serial.Serial`` used by the recorder."""

    def readline(self) -> bytes: ...

    def reset_input_buffer(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Outcome of one recording trial."""

    sample_count: int
    malformed_line_count: int
    invalid_adc_count: int = 0


CSV_FIELDS = (
    "timestamp",
    "raw_value",
    "gesture_label",
    "trial_number",
    "session_id",
)

# Dataset labels for the current binary-classification phase.
GESTURE_LABELS = ("OPEN_PALM", "CLOSED_FIST")


def open_serial(port: str, baudrate: int, timeout: float = 0.25) -> SerialLike:
    """Open the Arduino serial stream with a bounded read timeout.

    Importing pyserial lazily keeps non-hardware code and unit tests importable
    before acquisition dependencies are installed.
    """

    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is not installed; run: python -m pip install -r requirements.txt"
        ) from exc

    try:
        return serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
    except serial.SerialException as exc:
        raise RuntimeError(f"Could not open serial port {port!r}: {exc}") from exc


def create_session_file(data_dir: Path, session_id: str) -> tuple[Path, TextIO, csv.DictWriter]:
    """Create a raw-session CSV and write its stable header."""

    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"session_{session_id}.csv"
    handle = path.open("x", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    handle.flush()
    return path, handle, writer


def parse_adc_line(raw_line: bytes) -> int:
    """Parse one complete Arduino line and enforce the Uno's 10-bit range.

    ``Serial.readline`` may return a partial frame when its timeout expires.
    Requiring the newline prevents such fragments from becoming samples. The
    range check rejects merged frames if a delimiter was lost in transit.
    """

    if not raw_line.endswith(b"\n"):
        raise ValueError("unterminated serial frame")
    payload = raw_line[:-1]
    if payload.endswith(b"\r"):
        payload = payload[:-1]
    if not payload or not payload.isdigit():
        raise ValueError("ADC frame must contain ASCII digits only")
    value = int(payload)
    if not 0 <= value <= 1023:
        raise OverflowError(f"ADC value outside 0-1023: {value}")
    return value


def record_trial(
    connection: SerialLike,
    writer: csv.DictWriter,
    *,
    duration_seconds: float,
    gesture_label: str,
    trial_number: int,
    session_id: str,
    clock=time.monotonic,
) -> TrialResult:
    """Record valid integer lines for one timed trial into an existing CSV."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    if gesture_label not in GESTURE_LABELS:
        raise ValueError(
            f"gesture_label must be one of: {', '.join(GESTURE_LABELS)}"
        )

    try:
        connection.reset_input_buffer()
    except (OSError, IOError) as exc:
        raise RuntimeError(f"Could not prepare serial input for recording: {exc}") from exc
    deadline = clock() + duration_seconds
    sample_count = 0
    malformed_line_count = 0
    invalid_adc_count = 0

    while clock() < deadline:
        try:
            raw_line = connection.readline()
        except (OSError, IOError) as exc:
            raise RuntimeError(f"Serial connection lost during recording: {exc}") from exc

        if not raw_line:
            continue
        try:
            raw_value = parse_adc_line(raw_line)
        except OverflowError:
            invalid_adc_count += 1
            continue
        except ValueError:
            malformed_line_count += 1
            if not raw_line.endswith(b"\n"):
                # Discard the remainder of a timed-out partial frame and resume
                # at a clean delimiter boundary.
                try:
                    connection.reset_input_buffer()
                except (OSError, IOError) as exc:
                    raise RuntimeError(f"Could not resynchronize serial input: {exc}") from exc
            continue

        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_value": raw_value,
                "gesture_label": gesture_label,
                "trial_number": trial_number,
                "session_id": session_id,
            }
        )
        sample_count += 1

    return TrialResult(sample_count, malformed_line_count, invalid_adc_count)
