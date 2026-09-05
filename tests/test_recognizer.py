import unittest

from core import Command, CommandBus, CommandSource
from voice.recognizer import RecognitionEmitter


class MutableClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class RecognitionEmitterTests(unittest.TestCase):
    def test_open_and_close_emit_valid_voice_messages(self) -> None:
        bus = CommandBus()
        received = []
        bus.subscribe(received.append)
        clock = MutableClock()
        emitter = RecognitionEmitter(bus, clock=clock)

        opened = emitter.process_result(
            {"text": "open", "result": [{"word": "open", "conf": 0.91}]}
        )
        clock.value = 0.1
        closed = emitter.process_result(
            {"text": "close", "result": [{"word": "close", "conf": 0.87}]}
        )

        self.assertEqual(received, [opened, closed])
        self.assertEqual(opened.command, Command.OPEN)
        self.assertEqual(closed.command, Command.CLOSE)
        self.assertEqual(opened.source, CommandSource.VOICE)
        self.assertAlmostEqual(opened.confidence, 0.91)

    def test_rapid_identical_command_is_debounced(self) -> None:
        bus = CommandBus()
        received = []
        bus.subscribe(received.append)
        clock = MutableClock()
        emitter = RecognitionEmitter(bus, debounce_seconds=1.5, clock=clock)
        result = {"text": "open", "result": [{"word": "open", "conf": 0.9}]}

        emitter.process_result(result)
        clock.value = 0.5
        self.assertIsNone(emitter.process_result(result))
        clock.value = 1.6
        self.assertIsNotNone(emitter.process_result(result))
        self.assertEqual(len(received), 2)


if __name__ == "__main__":
    unittest.main()
