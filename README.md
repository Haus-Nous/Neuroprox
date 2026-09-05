# NeuroProx NEOS

NEOS is the software-side interface for the NeuroProx bionic arm. The command
contract between EMG/voice producers and Ishank's hardware/motor-control
consumer remains frozen. Phase 1 adds raw EMG acquisition, but deliberately
contains no signal processing, machine learning, or speech recognition.

## Integration contract

Every producer emits a `core.CommandMessage` with exactly these fields:

| Field | Type | Allowed value / meaning |
|---|---|---|
| `command` | `Command` string enum | `OPEN` or `CLOSE` |
| `confidence` | float | Inclusive range `0.0` to `1.0` |
| `source` | `CommandSource` string enum | `EMG` or `VOICE` |
| `timestamp` | timezone-aware datetime | Serialized as an ISO 8601 UTC string |

Example serialized message:

```json
{"command":"OPEN","confidence":0.99,"source":"EMG","timestamp":"2026-09-04T10:30:00+00:00"}
```

This schema and its serialized field names are the hardware/motor integration
contract. Discuss and coordinate any change with the hardware team before
altering it.

## Consuming commands in the same process

```python
from core import CommandBus

bus = CommandBus()

def move_arm(message):
    print(message.command, message.confidence, message.source, message.timestamp)

unsubscribe = bus.subscribe(move_arm)

# EMG or voice code receives the same bus instance and calls bus.emit(message).
# A polling consumer may instead read:
latest_message = bus.latest()  # None until the first command is emitted

# Stop callback delivery when appropriate:
unsubscribe()
```

Callbacks run synchronously in registration order. A callback exception is
propagated to the emitter. `latest()` is thread-safe and returns the most recent
immutable message without removing it.

## Mock producer

From the project root, run:

```bash
python mock_emitter.py --interval 1 --source EMG
```

It alternates between `OPEN` and `CLOSE`, emitting each through a real
`CommandBus`. Its sample subscriber prints the JSON wire representation. Stop it
with Ctrl-C. Ishank can replace the printing callback with motor-control logic.

## Separate-process transport

`CommandBus(transport=...)` accepts any object implementing
`CommandTransport.send(CommandMessage)`. `SerialTransport` and `SocketTransport`
reserve newline-delimited JSON as the wire direction, but both currently raise
`NotImplementedError`. They are intentional stubs; no serial port or socket is
opened in this milestone.

```python
from core import CommandBus, SocketTransport

# Contract is ready, but emit() will raise NotImplementedError until implemented.
bus = CommandBus(transport=SocketTransport("127.0.0.1", 8765))
```

## Layout and status

```text
neuroprox-neos/
├── core/                 shared schema, command bus, transport contracts
├── arduino/emg_stream/   timed 500 Hz Arduino Uno EMG streamer
├── emg/                  serial acquisition plus later-stage placeholders
├── voice/                voice-recognition placeholder
├── data/                 local recorded datasets (contents gitignored)
├── models/               local trained artifacts (contents gitignored)
├── scripts/               runnable data-recording tools
├── tests/                standard-library contract tests
├── mock_emitter.py       timed fake command producer
├── requirements.txt      dependency scaffold (no packages required yet)
└── README.md
```

Implemented now: validated command schema, command bus, transport interface,
mock producer, a fixed-rate Arduino EMG stream, and randomized labeled raw-data
recording.

Stubbed for later: EMG acquisition/processing/training/inference, voice command
recognition, serial output, and socket output. Planned packages are documented
in `requirements.txt` but intentionally not installed.

## Environment and tests

Python 3.10 or newer is required (the scaffold uses modern type syntax).

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Only `pyserial` is installed for Phase 1; no ML or voice libraries are included.

## EMG hardware bring-up and recording

Wire Muscle BioAmp Patchy `OUT -> A0`, `VCC -> 5V`, and `GND -> GND`, then upload
`arduino/emg_stream/emg_stream.ino` to the Arduino Uno. It emits one ADC integer
per line at a timed target of 500 Hz and 115200 baud.

On macOS, list serial ports with pyserial:

```bash
.venv/bin/python -m serial.tools.list_ports
```

Arduino Uno devices commonly appear as `/dev/cu.usbmodem...`. Use the exact
value printed on your Mac. Acquisition has two labels: `OPEN_PALM` maps to the
`OPEN` command and `CLOSED_FIST` maps to the `CLOSE` command. For a short first
test (one randomized trial for each gesture), run:

```bash
.venv/bin/python scripts/record_session.py --port /dev/cu.usbmodem1101 --trials-per-gesture 1
```

Replace `/dev/cu.usbmodem1101` with the discovered port. A full session omits
the final option and records 20 trials per gesture (40 randomized trials total).
Each trial uses a 3-second countdown, 4-second recording hold, and 2-second rest.
CSV output is written to `data/raw/session_<UTC timestamp>.csv`; dataset contents
remain gitignored.
