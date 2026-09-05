"""Canonical command schema shared by EMG, voice, and motor-control code."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class Command(str, Enum):
    """Motor-level gestures exposed by NEOS."""

    OPEN = "OPEN"
    CLOSE = "CLOSE"


class CommandSource(str, Enum):
    """NEOS modules allowed to produce commands."""

    EMG = "EMG"
    VOICE = "VOICE"


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class CommandMessage:
    """Immutable command passed across the NeuroProx integration boundary."""

    command: Command
    confidence: float
    source: CommandSource
    timestamp: datetime

    def __post_init__(self) -> None:
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise TypeError("confidence must be a number between 0 and 1")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", float(self.confidence))

        if not isinstance(self.command, Command):
            raise TypeError("command must be a Command enum value")
        if not isinstance(self.source, CommandSource):
            raise TypeError("source must be a CommandSource enum value")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")

    @classmethod
    def create(
        cls, command: Command, confidence: float, source: CommandSource
    ) -> "CommandMessage":
        """Create a message stamped with the current UTC time."""

        return cls(command, confidence, source, utc_now())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the contract."""

        payload = asdict(self)
        payload["command"] = self.command.value
        payload["source"] = self.source.value
        payload["timestamp"] = self.timestamp.astimezone(timezone.utc).isoformat()
        return payload

    def to_json(self) -> str:
        """Serialize the command for a process or hardware boundary."""

        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CommandMessage":
        """Validate and deserialize a mapping received at the boundary."""

        return cls(
            command=Command(payload["command"]),
            confidence=payload["confidence"],
            source=CommandSource(payload["source"]),
            timestamp=datetime.fromisoformat(payload["timestamp"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> "CommandMessage":
        """Validate and deserialize JSON received at the boundary."""

        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise TypeError("command JSON must contain an object")
        return cls.from_dict(decoded)
