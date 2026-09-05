# NeuroProx NEOS

NeuroProx is a bionic-arm project, and NEOS is its Python control software. It
turns either live muscle activity from a single-channel EMG sensor or the spoken
words **“open”** and **“close”** into one stable command interface for the arm.
The unified runner uses EMG whenever responsive hardware is connected and uses
offline Vosk voice control otherwise; the two input modes never run together.

## Hardware integration contract

Both input modes publish an immutable `core.CommandMessage` through the same
`core.CommandBus`. This is the interface Ishank's motor-control code should
consume:

| Field | Type | Values |
|---|---|---|
| `command` | string enum | `OPEN`, `CLOSE` |
| `confidence` | float | `0.0` through `1.0` |
| `source` | string enum | `EMG`, `VOICE` |
| `timestamp` | datetime | timezone-aware ISO 8601 UTC |

Example wire representation:

```json
{"command":"OPEN","confidence":0.99,"source":"EMG","timestamp":"2026-09-04T10:30:00+00:00"}
```

In-process consumers subscribe once and receive messages from either source:

```python
from core import CommandBus

bus = CommandBus()
unsubscribe = bus.subscribe(lambda message: motor_controller.handle(message))
latest = bus.latest()  # most recent message, or None
```

Callbacks run synchronously in registration order. `latest()` is thread-safe.
The serial/socket transport contracts are defined but remain stubs. The schema
and field names are the hardware/motor integration contract; coordinate with the
hardware team before changing them.

## Fresh-clone setup

Python 3.10 or newer is required. On macOS, install PortAudio once (needed by
the microphone library), then clone and set up the project:

```bash
brew install portaudio
git clone https://github.com/Haus-Nous/Neuroprox.git
cd Neuroprox
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_voice_model.py
```

The last command downloads the approximately 40 MB
`vosk-model-small-en-us-0.15` model into `models/vosk/`. That directory is
gitignored. macOS may ask for microphone permission the first time voice mode
runs; allow it for the terminal application. Voice mode also requires an
available microphone input; `python -m sounddevice` lists the audio devices that
PortAudio can see. The trained EMG v3 model is already included in the
repository.

For EMG, upload `arduino/emg_stream/emg_stream.ino` to an Arduino Uno and wire
the Muscle BioAmp Patchy as `OUT -> A0`, `VCC -> 5V`, `GND -> GND`. No EMG
hardware setup is needed for voice-only operation.

## Run NEOS

From the activated virtual environment, the single demo command is:

```bash
python scripts/run_neos.py
```

NEOS probes Arduino-like serial devices for valid ADC samples. If one responds,
it selects EMG control and guides you through an OPEN_PALM calibration before
live inference. If none responds, it automatically falls back to offline voice
control. Specify a known port when auto-detection is unsuitable:

```bash
python scripts/run_neos.py --emg-port /dev/cu.usbmodem1101
```

For a minimal demo console showing only mode selection and `EMIT` messages:

```bash
python scripts/run_neos.py --quiet
```

In quiet EMG mode, hold a relaxed OPEN_PALM immediately after the mode-selection
message: calibration begins without the usual prompt or countdown and lasts four
seconds. Stop either mode with Ctrl-C.

The v3 EMG model requires this per-run OPEN_PALM baseline because electrode
placement and skin contact shift signal levels between sessions. Its honest
trial-level, session-aware 5-fold cross-validation estimate is **70.35% mean
accuracy with a 7.05 percentage-point standard deviation**. This is an
experimental estimate on the recorded two-session dataset, not a claim of
clinical performance or guaranteed live accuracy.

## Project status

Phases 0–7 are complete: shared command contract and bus, timed Arduino EMG
streaming, data acquisition, signal processing and feature extraction, model
training/evaluation, calibrated real-time EMG inference, offline constrained
voice recognition, and the mutually exclusive unified runner with automatic
fallback. Raw and processed datasets and the downloaded Vosk model stay local;
the deployable EMG v3 model is tracked.

Ishank's remaining integration work is the motor/hardware consumer that
subscribes to `CommandBus` (or completes one of the separate-process transports)
and maps `OPEN`/`CLOSE` messages to safe actuator behavior. He does not need to
depend on the EMG or voice internals.

## Tests and layout

```bash
python -m unittest discover -s tests -v
```

```text
core/                   command schema, bus, and transport contracts
emg/                    acquisition, processing, training, and inference
voice/                  offline Vosk recognition
arduino/emg_stream/     Arduino Uno sampling firmware
scripts/                unified runner and supporting CLI tools
models/                 tracked EMG v3 artifact; ignored Vosk download
data/                   ignored raw and processed recordings
tests/                  automated tests
```
