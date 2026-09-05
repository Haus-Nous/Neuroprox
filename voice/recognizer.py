"""Offline, grammar-constrained Vosk control for OPEN and CLOSE commands."""

from __future__ import annotations

import json
import queue
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from core import Command, CommandBus, CommandMessage, CommandSource


VOICE_GRAMMAR = ("open", "close", "[unk]")
WORD_TO_COMMAND = {"open": Command.OPEN, "close": Command.CLOSE}
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_BLOCK_SIZE = 4_000


def model_is_ready(model_path: Path) -> bool:
    """Check for key files from an extracted Vosk model."""

    return (
        model_path.is_dir()
        and (model_path / "am" / "final.mdl").is_file()
        and (model_path / "conf" / "model.conf").is_file()
    )


class RecognitionEmitter:
    """Validate Vosk results, debounce them, and publish command messages."""

    def __init__(
        self,
        bus: CommandBus,
        *,
        minimum_confidence: float = 0.65,
        debounce_seconds: float = 1.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if debounce_seconds < 0:
            raise ValueError("debounce_seconds may not be negative")
        self.bus = bus
        self.minimum_confidence = minimum_confidence
        self.debounce_seconds = debounce_seconds
        self.clock = clock
        self._last_command: Command | None = None
        self._last_emitted_at = float("-inf")

    def process_result(
        self, result: str | Mapping[str, Any]
    ) -> CommandMessage | None:
        """Process one final Vosk result and return an emitted message, if any."""

        payload = json.loads(result) if isinstance(result, str) else dict(result)
        recognized_text = str(payload.get("text", "")).strip().lower()
        command = WORD_TO_COMMAND.get(recognized_text)
        if command is None:
            return None

        words = payload.get("result", [])
        confidences = [
            float(word["conf"])
            for word in words
            if isinstance(word, Mapping) and "conf" in word
        ]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        if confidence < self.minimum_confidence:
            return None

        now = self.clock()
        if (
            command == self._last_command
            and now - self._last_emitted_at < self.debounce_seconds
        ):
            return None

        message = CommandMessage.create(command, confidence, CommandSource.VOICE)
        self.bus.emit(message)
        self._last_command = command
        self._last_emitted_at = now
        return message


def listen(
    model_path: Path,
    bus: CommandBus,
    *,
    minimum_confidence: float = 0.65,
    debounce_seconds: float = 1.5,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    block_size: int = DEFAULT_BLOCK_SIZE,
    device: int | str | None = None,
    verbose: bool = True,
) -> None:
    """Capture default-microphone audio and emit commands until interrupted."""

    if not model_is_ready(model_path):
        raise RuntimeError(f"Vosk model is missing or incomplete: {model_path}")
    try:
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model, SetLogLevel
    except ImportError as exc:
        raise RuntimeError(
            "Voice dependencies are missing; run: "
            "python -m pip install -r requirements.txt"
        ) from exc

    # Vosk otherwise writes verbose native-library startup diagnostics to stderr.
    SetLogLevel(-1)
    model = Model(str(model_path))
    recognizer = KaldiRecognizer(model, sample_rate, json.dumps(VOICE_GRAMMAR))
    recognizer.SetWords(True)
    emitter = RecognitionEmitter(
        bus,
        minimum_confidence=minimum_confidence,
        debounce_seconds=debounce_seconds,
    )
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=32)

    def audio_callback(indata, frames, timing, status) -> None:
        del frames, timing
        if status and verbose:
            print(f"Audio status: {status}", file=sys.stderr)
        chunk = bytes(indata)
        try:
            audio_queue.put_nowait(chunk)
        except queue.Full:
            # Drop the oldest chunk rather than allowing latency to grow.
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
            audio_queue.put_nowait(chunk)

    try:
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            device=device,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            while True:
                audio = audio_queue.get()
                if recognizer.AcceptWaveform(audio):
                    emitter.process_result(recognizer.Result())
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Could not use microphone: {exc}") from exc
