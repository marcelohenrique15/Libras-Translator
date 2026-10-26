import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from application.runtime.channels import receive_latest
from application.config import RuntimeConfig
from application.types import AnalysisResult, FramePacket
from application.runtime.worker import AnalysisWorker


class FakePipeline:
    """Pipeline leve, importável pelo spawn e observável por arquivos pequenos."""

    device = "fake-cpu"

    def __init__(self, config):
        self.directory = config.checkpoint
        self.resets = 0

    def _record(self, event):
        with (self.directory / "events.jsonl").open("a") as file:
            file.write(json.dumps(event) + "\n")

    def __enter__(self):
        self._record({"event": "open", "pid": os.getpid()})
        return self

    def __exit__(self, *_):
        self._record({"event": "close", "pid": os.getpid()})

    def reset(self):
        self.resets += 1
        self._record({"event": "reset", "count": self.resets})

    def process(self, frame):
        if frame.image.flags.writeable:
            raise RuntimeError("O processo recebeu um frame mutável.")
        return AnalysisResult(
            frame.frame_id, frame.timestamp, box=None,
            caption=f"resets:{self.resets}", status="Processado",
        )


class GatedPipeline(FakePipeline):
    def process(self, frame):
        if frame.frame_id == 1:
            (self.directory / "processing").touch()
            deadline = time.monotonic() + 10
            while not (self.directory / "release").exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("O teste não liberou a inferência em trânsito.")
                time.sleep(0.01)
        return super().process(frame)


class FailingPipeline(FakePipeline):
    def process(self, frame):
        raise RuntimeError("Falha simulada na inferência.")


class CrashedPipeline(FakePipeline):
    def process(self, frame):
        os._exit(7)


class BlockedPipeline(FakePipeline):
    def process(self, frame):
        (self.directory / "processing").touch()
        time.sleep(60)


class AnalysisWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = RuntimeConfig(checkpoint=self.directory, device="cpu")

    def _wait(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.01)
        self.fail("O processo não respondeu dentro do prazo do teste.")

    def _worker(self, factory=FakePipeline):
        worker = AnalysisWorker(self.config, pipeline_factory=factory)
        self.addCleanup(worker.stop)
        worker.start()
        self._wait(lambda: worker.ready or worker.error)
        self.assertIsNone(worker.error)
        return worker

    def _frame(self, identifier):
        return FramePacket(identifier, time.monotonic(), np.zeros((8, 8, 3), dtype=np.uint8))

    def _events(self):
        return [json.loads(line) for line in (self.directory / "events.jsonl").read_text().splitlines()]

    def test_spawn_round_trip_owns_pipeline_in_child_and_closes_it(self):
        worker = self._worker()
        self.assertNotEqual(worker._process.pid, os.getpid())
        self.assertEqual(worker.device, "fake-cpu")
        process = worker._process
        worker.start()
        self.assertIs(worker._process, process)
        self.assertTrue(worker.submit(self._frame(4)))
        result = self._wait(worker.read_latest)
        self.assertEqual(result.frame_id, 4)
        self.assertEqual(result.caption, "resets:1")
        worker.stop()
        worker.stop()
        self.assertFalse(process.is_alive())
        events = self._events()
        self.assertEqual(events[0], {"event": "open", "pid": process.pid})
        self.assertEqual(events[-1], {"event": "close", "pid": process.pid})
        self.assertFalse(worker.submit(self._frame(5)))
        self.assertIsNone(worker.read_latest())

    def test_reset_discards_a_real_prediction_already_in_flight(self):
        worker = self._worker(GatedPipeline)
        self.assertTrue(worker.submit(self._frame(1)))
        self._wait(lambda: (self.directory / "processing").exists())
        worker.reset()
        (self.directory / "release").touch()

        received = []

        def observe_output(channel):
            message = receive_latest(channel)
            if message is not None:
                received.append(message)
            return message

        with patch("application.runtime.worker.receive_latest", side_effect=observe_output):
            deadline = time.monotonic() + 10
            while not received and time.monotonic() < deadline:
                self.assertIsNone(worker.read_latest())
                time.sleep(0.01)
        self.assertEqual(len(received), 1)
        stale_generation, stale_result = received[0]
        self.assertEqual(stale_generation, 0)
        self.assertEqual(stale_result.frame_id, 1)

        self.assertTrue(worker.submit(self._frame(2)))
        current = self._wait(worker.read_latest)
        self.assertEqual(current.frame_id, 2)
        self.assertEqual(current.caption, "resets:2")
        worker.stop()
        resets = [event["count"] for event in self._events() if event["event"] == "reset"]
        self.assertEqual(resets, [1, 2])

    def test_inference_failure_reaches_parent_and_releases_pipeline(self):
        worker = self._worker(FailingPipeline)
        self.assertTrue(worker.submit(self._frame(1)))
        self._wait(lambda: worker.error)
        self.assertIn("Falha simulada", worker.error)
        worker.stop()
        self.assertFalse(worker._process.is_alive())
        self.assertEqual(self._events()[-1]["event"], "close")

    def test_abrupt_child_failure_is_reported_without_waiting_forever(self):
        worker = self._worker(CrashedPipeline)
        worker.submit(self._frame(1))
        self._wait(lambda: worker.error)
        self.assertIn("código 7", worker.error)
        worker.stop()
        self.assertFalse(worker._process.is_alive())

    def test_native_call_that_ignores_stop_is_terminated_within_deadline(self):
        worker = self._worker(BlockedPipeline)
        worker.submit(self._frame(1))
        self._wait(lambda: (self.directory / "processing").exists())
        process = worker._process
        started = time.monotonic()
        worker.stop()
        self.assertLess(time.monotonic() - started, 4)
        self.assertFalse(process.is_alive())
        self.assertIsNone(worker.read_latest())

    def test_input_before_start_does_not_create_a_hidden_queue(self):
        worker = AnalysisWorker(self.config, pipeline_factory=FakePipeline)
        self.assertFalse(worker.ready)
        self.assertEqual(worker.device, "cpu")
        self.assertFalse(worker.submit(self._frame(1)))
        self.assertIsNone(worker.read_latest())
        self.assertIsNone(worker.error)
        worker.stop()
        worker.stop()


if __name__ == "__main__":
    unittest.main()
