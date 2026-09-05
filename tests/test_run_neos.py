import unittest
from types import SimpleNamespace

from core import CommandBus
from scripts.run_neos import EMGConnection, detect_emg_connection, run_with_fallback


class StepClock:
    def __init__(self, step: float = 0.05) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


class FakeSerial:
    def __init__(self, lines=()) -> None:
        self.lines = iter(lines)
        self.closed = False

    def readline(self) -> bytes:
        return next(self.lines, b"")

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class UnifiedRunnerTests(unittest.TestCase):
    def test_valid_mock_serial_selects_emg(self) -> None:
        serial = FakeSerial([b"noise\n", b"512\r\n"])
        result = detect_emg_connection(
            None,
            port_provider=lambda: [
                SimpleNamespace(
                    device="/dev/cu.usbmodem-test",
                    description="Arduino Uno",
                    manufacturer="Arduino",
                    product="Uno",
                )
            ],
            serial_factory=lambda *args, **kwargs: serial,
            clock=StepClock(),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.port, "/dev/cu.usbmodem-test")
        self.assertFalse(serial.closed)

    def test_no_device_selects_voice(self) -> None:
        result = detect_emg_connection(None, port_provider=lambda: [])
        calls = []
        mode = run_with_fallback(
            result,
            CommandBus(),
            emg_runner=lambda selection, bus: calls.append("emg"),
            voice_runner=lambda bus: calls.append("voice"),
        )
        self.assertEqual(mode, "voice")
        self.assertEqual(calls, ["voice"])

    def test_mid_run_emg_failure_falls_back_to_voice(self) -> None:
        serial = FakeSerial()
        calls = []

        def fail_emg(selection, bus) -> None:
            calls.append("emg")
            raise RuntimeError("sensor disconnected")

        mode = run_with_fallback(
            EMGConnection(serial, "/dev/cu.usbmodem-test"),
            CommandBus(),
            emg_runner=fail_emg,
            voice_runner=lambda bus: calls.append("voice"),
        )
        self.assertEqual(mode, "voice")
        self.assertEqual(calls, ["emg", "voice"])
        self.assertTrue(serial.closed)


if __name__ == "__main__":
    unittest.main()
