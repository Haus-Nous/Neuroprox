#!/usr/bin/env python3
"""Download and safely extract the small offline English Vosk model."""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice.recognizer import model_is_ready


MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
MODEL_PARENT = PROJECT_ROOT / "models" / "vosk"
MODEL_PATH = MODEL_PARENT / MODEL_NAME


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract only members that remain within the destination directory."""

    destination = destination.resolve()
    for member in archive.infolist():
        member_path = (destination / member.filename).resolve()
        if destination not in member_path.parents and member_path != destination:
            raise RuntimeError(f"Unsafe path in model archive: {member.filename}")
    archive.extractall(destination)


def main() -> int:
    if model_is_ready(MODEL_PATH):
        print(f"Vosk model is already installed: {MODEL_PATH}")
        return 0

    MODEL_PARENT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_URL}")
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            archive_path = temporary / f"{MODEL_NAME}.zip"
            urllib.request.urlretrieve(MODEL_URL, archive_path)
            print("Download complete. Extracting...")
            with zipfile.ZipFile(archive_path) as archive:
                safe_extract(archive, temporary)
            extracted = temporary / MODEL_NAME
            if not model_is_ready(extracted):
                raise RuntimeError("Downloaded archive does not contain a complete Vosk model")
            if MODEL_PATH.exists():
                shutil.rmtree(MODEL_PATH)
            shutil.move(str(extracted), str(MODEL_PATH))
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, RuntimeError) as exc:
        print(f"Voice model setup failed: {exc}", file=sys.stderr)
        return 1

    print(f"Vosk model installed: {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
