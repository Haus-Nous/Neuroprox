#!/usr/bin/env python3
"""Run NeuroProx NEOS with mutually exclusive EMG-or-voice control."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import CommandBus
from emg.acquisition import SerialLike, open_serial, parse_adc_line
from emg.inference import calibrate, load_inference_model, run_live_inference
from voice.recognizer import listen, model_is_ready


DEFAULT_EMG_MODEL = (
    PROJECT_ROOT / "models" / "emg_binary_v3_combined_normalized.joblib"
)

DEFAULT_VOICE_MODEL = (
    PROJECT_ROOT / "models" / "vosk" / "vosk-model-small-en-us-0.15"
)

ARDUINO_PORT_HINTS = (
    "arduino",
    "usbmodem",
    "usbserial",
    "wchusbserial",
    "ch340",
    "ttyacm",
    "ttyusb",
)


@dataclass(frozen=True, slots=True)
class EMGConnection:
    connection: SerialLike
    port: str


def arduino_like_ports(port_infos: Iterable[object]) -> list[str]:
    """Return unique device names whose metadata looks USB/Arduino-like."""

    devices: list[str] = []

    for info in port_infos:
        device = str(getattr(info, "device", ""))

        searchable = " ".join(
            str(getattr(info, attribute, "") or "")
            for attribute in (
                "device",
                "description",
                "manufacturer",
                "product",
            )
        ).lower()

        if device and any(
            hint in searchable
            for hint in ARDUINO_PORT_HINTS
        ):
            if device not in devices:
                devices.append(device)

    return devices


def probe_emg_port(
    port: str,
    *,
    baud: int = 115200,
    timeout_seconds: float = 2.0,
    serial_factory: Callable[..., SerialLike] = open_serial,
    clock: Callable[[], float] = time.monotonic,
) -> EMGConnection | None:
    """Open a candidate and require one complete valid ADC frame."""

    connection = None

    try:
        connection = serial_factory(
            port,
            baud,
            timeout=0.1,
        )

        deadline = clock() + timeout_seconds

        while clock() < deadline:

            raw_line = connection.readline()

            if not raw_line:
                continue

            try:
                parse_adc_line(raw_line)

            except (ValueError, OverflowError):

                if not raw_line.endswith(b"\n"):
                    connection.reset_input_buffer()

                continue

            return EMGConnection(
                connection,
                port,
            )

    except (OSError, IOError, RuntimeError):
        pass

    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass

    return None


def detect_emg_connection(
    requested_port: str | None,
    *,
    baud: int = 115200,
    timeout_seconds: float = 2.0,
    port_provider: Callable[[], Iterable[object]] | None = None,
    serial_factory: Callable[..., SerialLike] = open_serial,
    clock: Callable[[], float] = time.monotonic,
) -> EMGConnection | None:
    """Probe the requested port or auto-detected Arduino-like candidates."""

    if requested_port:

        candidates = [requested_port]

    else:

        if port_provider is None:
            from serial.tools import list_ports

            port_provider = list_ports.comports

        candidates = arduino_like_ports(
            port_provider()
        )

    for port in candidates:

        detected = probe_emg_port(
            port,
            baud=baud,
            timeout_seconds=timeout_seconds,
            serial_factory=serial_factory,
            clock=clock,
        )

        if detected is not None:
            return detected

    return None


def send_motor_command(
    connection: SerialLike,
    command: str,
) -> None:
    """Send a motor command to the Arduino."""

    message = f"{command}\n".encode("ascii")

    connection.write(message)

    # Flush so the command is sent immediately.
    try:
        connection.flush()
    except AttributeError:
        pass


def run_with_fallback(
    detected: EMGConnection | None,
    bus: CommandBus,
    *,
    emg_runner: Callable[
        [EMGConnection, CommandBus],
        None
    ],
    voice_runner: Callable[[CommandBus], None],
    quiet: bool = False,
) -> str:
    """Run exactly one source, switching to voice only after an EMG failure."""

    if detected is None:

        print(
            "No responsive EMG sensor detected — "
            "falling back to voice control",
            flush=True,
        )

        voice_runner(bus)

        return "voice"

    print(
        f"EMG sensor detected on {detected.port} "
        f"— using EMG control",
        flush=True,
    )

    try:

        emg_runner(
            detected,
            bus,
        )

        return "emg"

    except (OSError, IOError, RuntimeError) as exc:

        if not quiet:
            print(
                f"EMG control failed: {exc}",
                file=sys.stderr,
            )

        print(
            "EMG connection lost — switching to voice control",
            file=sys.stderr,
            flush=True,
        )

        try:
            detected.connection.close()
        except Exception:
            pass

        voice_runner(bus)

        return "voice"


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--emg-port",
        help="Arduino serial port; omit to auto-detect",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
    )

    parser.add_argument(
        "--probe-seconds",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--emg-model",
        type=Path,
        default=DEFAULT_EMG_MODEL,
    )

    parser.add_argument(
        "--emg-calibration-seconds",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--emg-debounce-windows",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--voice-model",
        type=Path,
        default=DEFAULT_VOICE_MODEL,
    )

    parser.add_argument(
        "--voice-confidence",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--voice-debounce-seconds",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--voice-device",
        help="optional sounddevice input device",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="show only mode selection and emitted commands",
    )

    args = parser.parse_args()

    if args.baud <= 0 or args.probe_seconds <= 0:
        parser.error(
            "baud and probe duration must be positive"
        )

    if (
        args.emg_calibration_seconds <= 0
        or args.emg_debounce_windows <= 0
    ):
        parser.error(
            "EMG calibration and debounce values "
            "must be positive"
        )

    if (
        not 0 <= args.voice_confidence <= 1
        or args.voice_debounce_seconds < 0
    ):
        parser.error(
            "invalid voice confidence or debounce value"
        )

    return args


def main() -> int:

    args = parse_args()

    bus = CommandBus()

    # Normal NEOS command output.
    bus.subscribe(
        lambda message: print(
            f"EMIT {message.to_json()}",
            flush=True,
        )
    )

    if not args.quiet:

        print(
            "=== NeuroProx NEOS — starting ===",
            flush=True,
        )

        print(
            "Checking for a responsive EMG sensor...",
            flush=True,
        )

    detected = detect_emg_connection(
        args.emg_port,
        baud=args.baud,
        timeout_seconds=args.probe_seconds,
    )

    def run_emg(
        selection: EMGConnection,
        shared_bus: CommandBus,
    ) -> None:

        # -------------------------------------------------
        # MOTOR CONTROL
        # -------------------------------------------------

        def motor_control(message) -> None:

            command = getattr(
                message,
                "command",
                None,
            )

            if command == "CLOSE":

                print(
                    "MOTOR → LEFT",
                    flush=True,
                )

                send_motor_command(
                    selection.connection,
                    "CLOSE",
                )
            else:
                print(
                    "MOTOR → Stop",
                    flush=True,
                )

                send_motor_command(
                    selection.connection,
                    "STOP",
                )

        # Listen for commands emitted by the EMG model.
        shared_bus.subscribe(
            motor_control
        )

        try:

            model = load_inference_model(
                args.emg_model
            )

            if not args.quiet:

                print(
                    "Calibration: relax your hand "
                    "in OPEN_PALM and hold still."
                )

                input(
                    "Press Enter when ready..."
                )

                for remaining in range(3, 0, -1):

                    print(
                        f"Starting calibration in "
                        f"{remaining}...",
                        flush=True,
                    )

                    time.sleep(1)

                print(
                    f"Recording OPEN_PALM baseline for "
                    f"{args.emg_calibration_seconds:g} "
                    f"seconds..."
                )

            calibration = calibrate(
                selection.connection,
                duration_seconds=(
                    args.emg_calibration_seconds
                ),
            )

            if not args.quiet:

                print(
                    f"Calibration complete: "
                    f"baseline RMS="
                    f"{calibration.baseline_rms:.6f}, "
                    f"sample rate="
                    f"{calibration.sample_rate_hz:.2f} Hz"
                )

                print(
                    "Live EMG inference started. "
                    "Press Ctrl-C to stop."
                )

            run_live_inference(
                selection.connection,
                model,
                calibration,
                shared_bus,
                debounce_windows=(
                    args.emg_debounce_windows
                ),
            )

        finally:

            try:
                # Make sure motor is stopped before closing.
                send_motor_command(
                    selection.connection,
                    "STOP",
                )
            except Exception:
                pass

            try:
                selection.connection.close()
            except Exception:
                pass

    def run_voice(
        shared_bus: CommandBus,
    ) -> None:

        if not model_is_ready(
            args.voice_model
        ):

            raise RuntimeError(
                "Voice model is missing. Run: "
                ".venv/bin/python "
                "scripts/download_voice_model.py"
            )

        # Suppress native Vosk diagnostics.
        try:

            from vosk import SetLogLevel

            SetLogLevel(-1)

        except ImportError:
            pass

        if not args.quiet:

            print(
                'Listening offline for '
                '"open" or "close". '
                "Press Ctrl-C to stop."
            )

        listen(
            args.voice_model,
            shared_bus,
            minimum_confidence=(
                args.voice_confidence
            ),
            debounce_seconds=(
                args.voice_debounce_seconds
            ),
            device=args.voice_device,
            verbose=not args.quiet,
        )

    try:

        run_with_fallback(
            detected,
            bus,
            emg_runner=run_emg,
            voice_runner=run_voice,
            quiet=args.quiet,
        )

    except KeyboardInterrupt:

        if not args.quiet:
            print(
                "\nNEOS stopped."
            )

    except RuntimeError as exc:

        print(
            f"NEOS error: {exc}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())