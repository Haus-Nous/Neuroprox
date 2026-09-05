import csv
import io
import unittest

from emg.acquisition import CSV_FIELDS, parse_adc_line, record_trial


class FakeSerial:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = iter(lines)
        self.was_reset = False

    def reset_input_buffer(self) -> None:
        self.was_reset = True

    def readline(self) -> bytes:
        return next(self.lines, b"")


class StepClock:
    def __init__(self) -> None:
        self.value = -0.1

    def __call__(self) -> float:
        self.value += 0.1
        return self.value


class AcquisitionTests(unittest.TestCase):
    def test_parser_requires_delimiter_and_adc_range(self) -> None:
        self.assertEqual(parse_adc_line(b"512\r\n"), 512)
        with self.assertRaises(ValueError):
            parse_adc_line(b"51")
        with self.assertRaises(OverflowError):
            parse_adc_line(b"50515\n")

    def test_record_trial_writes_valid_samples_and_skips_noise(self) -> None:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        connection = FakeSerial([b"512\r\n", b"noise\n", b"700\n"])

        result = record_trial(
            connection,
            writer,
            duration_seconds=0.4,
            gesture_label="OPEN_PALM",
            trial_number=2,
            session_id="test-session",
            clock=StepClock(),
        )
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))

        self.assertTrue(connection.was_reset)
        self.assertEqual(result.sample_count, 2)
        self.assertEqual(result.malformed_line_count, 1)
        self.assertEqual([row["raw_value"] for row in rows], ["512", "700"])
        self.assertTrue(all(row["gesture_label"] == "OPEN_PALM" for row in rows))
        self.assertTrue(all(row["trial_number"] == "2" for row in rows))

    def test_non_binary_gesture_label_is_rejected(self) -> None:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        with self.assertRaises(ValueError):
            record_trial(
                FakeSerial([]),
                writer,
                duration_seconds=1.0,
                gesture_label="HALF_CLAW",
                trial_number=1,
                session_id="test-session",
            )


if __name__ == "__main__":
    unittest.main()
