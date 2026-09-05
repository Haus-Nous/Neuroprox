"""Emit fake commands so motor-control integration can begin immediately."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator

from core import Command, CommandBus, CommandMessage, CommandSource


def fake_commands(source: CommandSource) -> Iterator[CommandMessage]:
    """Yield the supported commands forever with stable fake confidence."""

    commands = (Command.OPEN, Command.CLOSE)
    while True:
        for command in commands:
            yield CommandMessage.create(command, confidence=0.99, source=source)


def run(interval: float, source: CommandSource) -> None:
    """Start a mock producer and a sample in-process consumer."""

    bus = CommandBus()
    bus.subscribe(lambda message: print(message.to_json(), flush=True))
    try:
        for message in fake_commands(source):
            bus.emit(message)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Mock emitter stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between commands")
    parser.add_argument(
        "--source",
        type=CommandSource,
        choices=list(CommandSource),
        default=CommandSource.EMG,
        help="fake producer source",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.interval, arguments.source)
