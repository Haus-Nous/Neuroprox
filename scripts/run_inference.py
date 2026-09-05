#!/usr/bin/env python3
"""Calibrate and run real-time NeuroProx EMG inference."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import CommandBus
from emg.acquisition import open_serial
from emg.inference import calibrate, load_inference_model, run_live_inference


DEFAULT_MODEL = PROJECT_ROOT / "models" / "emg_binary_v3_combined_normalized.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Arduino port, e.g. /dev/cu.usbmodem1101")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration-seconds", type=float, default=4.0)
    parser.add_argument("--debounce-windows", type=int, default=3)
    parser.add_argument("--startup-delay", type=float, default=2.0)
    args = parser.parse_args()
    if args.calibration_seconds <= 0 or args.startup_delay < 0:
        parser.error("calibration must be positive and startup delay may not be negative")
    if args.debounce_windows <= 0:
        parser.error("--debounce-windows must be positive")
    return args


def main() -> int:
    args = parse_args()
    connection = None
    try:
        model = load_inference_model(args.model)
        connection = open_serial(args.port, args.baud)
        time.sleep(args.startup_delay)
        print("Calibration: relax your hand in OPEN_PALM and hold still.")
        input("Press Enter when ready...")
        for remaining in range(3, 0, -1):
            print(f"Starting calibration in {remaining}...", flush=True)
            time.sleep(1)
        print(f"Recording OPEN_PALM baseline for {args.calibration_seconds:g} seconds...")
        calibration = calibrate(
            connection, duration_seconds=args.calibration_seconds
        )
        print(
            f"Calibration complete: baseline RMS={calibration.baseline_rms:.6f}, "
            f"sample rate={calibration.sample_rate_hz:.2f} Hz, "
            f"samples={calibration.sample_count}, windows={calibration.window_count}, "
            f"dropped frames={calibration.dropped_frames}"
        )
        bus = CommandBus()
        bus.subscribe(lambda message: print(f"EMIT {message.to_json()}", flush=True))
        print("Live inference started. Press Ctrl-C to stop.")
        run_live_inference(
            connection,
            model,
            calibration,
            bus,
            debounce_windows=args.debounce_windows,
        )
    except KeyboardInterrupt:
        print("\nInference stopped.")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Inference error: {exc}", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception as exc:
                print(f"Warning: could not close serial port: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
