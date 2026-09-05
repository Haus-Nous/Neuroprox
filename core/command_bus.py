"""In-process command distribution with an optional external transport."""

from __future__ import annotations

from threading import RLock
from typing import Callable

from .schema import CommandMessage
from .transports import CommandTransport

CommandCallback = Callable[[CommandMessage], None]


class CommandBus:
    """Publish commands to callbacks and, optionally, an external transport.

    Callback delivery is synchronous: ``emit`` returns after every subscriber
    and configured transport has handled the message.
    """

    def __init__(self, transport: CommandTransport | None = None) -> None:
        self._transport = transport
        self._subscribers: list[CommandCallback] = []
        self._latest: CommandMessage | None = None
        self._lock = RLock()

    def subscribe(self, callback: CommandCallback) -> Callable[[], None]:
        """Register a callback and return a function that unsubscribes it."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def latest(self) -> CommandMessage | None:
        """Poll the most recently emitted command, or ``None`` if empty."""

        with self._lock:
            return self._latest

    def emit(self, command: CommandMessage) -> None:
        """Publish a validated command to all configured consumers."""

        if not isinstance(command, CommandMessage):
            raise TypeError("command must be a CommandMessage")
        with self._lock:
            self._latest = command
            subscribers = tuple(self._subscribers)

        for callback in subscribers:
            callback(command)
        if self._transport is not None:
            self._transport.send(command)

