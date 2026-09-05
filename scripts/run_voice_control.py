#!/usr/bin/env python3
"""Run offline NeuroProx voice control from the default microphone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import CommandBus
from voice.recognizer import listen, model_is_ready


DEFAULT_MODEL = PROJECT_ROOT / "models" / "vosk" / "vosk-model-small-en-us-0.15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.65)
    parser.add_argument("--debounce-seconds", type=float, default=1.5)
    parser.add_argument("--device", help="optional sounddevice input device ID or name")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.debounce_seconds < 0:
        parser.error("--debounce-seconds may not be negative")
    return args


def main() -> int:
    args = parse_args()
    if not model_is_ready(args.model):
        print(f"Voice model not found at: {args.model}", file=sys.stderr)
        print(
            "Download it first with: .venv/bin/python scripts/download_voice_model.py",
            file=sys.stderr,
        )
        return 1

    bus = CommandBus()
    bus.subscribe(lambda message: print(f"EMIT {message.to_json()}", flush=True))
    print('Listening offline for "open" or "close". Press Ctrl-C to stop.')
    try:
        listen(
            args.model,
            bus,
            minimum_confidence=args.confidence,
            debounce_seconds=args.debounce_seconds,
            device=args.device,
        )
    except KeyboardInterrupt:
        print("\nVoice control stopped.")
    except RuntimeError as exc:
        print(f"Voice control error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
