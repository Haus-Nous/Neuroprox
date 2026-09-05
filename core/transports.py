"""External-process transport contracts; implementations are deferred."""

from __future__ import annotations

from typing import Protocol

from .schema import CommandMessage


class CommandTransport(Protocol):
    """Interface implemented by any external command transport."""

    def send(self, command: CommandMessage) -> None:
        """Send one command to an external consumer."""


class SerialTransport:
    """Reserved for newline-delimited JSON over a serial connection."""

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self.port = port
        self.baudrate = baudrate

    def send(self, command: CommandMessage) -> None:
        raise NotImplementedError("Serial transport is scaffolded but not implemented")


class SocketTransport:
    """Reserved for newline-delimited JSON over a TCP socket."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def send(self, command: CommandMessage) -> None:
        raise NotImplementedError("Socket transport is scaffolded but not implemented")

