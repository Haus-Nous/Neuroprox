from datetime import datetime, timezone
import unittest

from core import Command, CommandBus, CommandMessage, CommandSource


class CommandContractTests(unittest.TestCase):
    def test_json_round_trip(self) -> None:
        message = CommandMessage(
            Command.CLOSE,
            0.75,
            CommandSource.EMG,
            datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        )

        self.assertEqual(CommandMessage.from_json(message.to_json()), message)

    def test_binary_command_set_is_frozen(self) -> None:
        self.assertEqual(list(Command), [Command.OPEN, Command.CLOSE])

    def test_confidence_is_bounded(self) -> None:
        with self.assertRaises(ValueError):
            CommandMessage.create(Command.OPEN, 1.01, CommandSource.VOICE)

    def test_bus_callback_latest_and_unsubscribe(self) -> None:
        bus = CommandBus()
        received: list[CommandMessage] = []
        unsubscribe = bus.subscribe(received.append)
        first = CommandMessage.create(Command.OPEN, 0.9, CommandSource.VOICE)
        second = CommandMessage.create(Command.CLOSE, 0.8, CommandSource.EMG)

        bus.emit(first)
        unsubscribe()
        bus.emit(second)

        self.assertEqual(received, [first])
        self.assertEqual(bus.latest(), second)


if __name__ == "__main__":
    unittest.main()
