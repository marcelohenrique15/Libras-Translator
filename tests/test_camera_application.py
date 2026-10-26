"""Verifica o controlador com componentes locais, sem abrir janela ou câmera."""

import os
import threading
import unittest
from unittest.mock import patch

import numpy as np

from application.runtime.app import CameraApplication
from application.config import RuntimeConfig
from application.types import AnalysisResult, Candidate, FramePacket, PersonBox


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 0.001)


class FakeCamera:
    def __init__(self, events, clock):
        self.events = events
        self.clock = clock
        self.error = ""
        self.finished = False
        self.frames_read = 0
        self.start_error = None
        self.read_error = None
        self.stop_error = None

    def start(self):
        self.events.append("camera.start")
        if self.start_error:
            raise self.start_error

    def read_latest(self):
        if self.read_error:
            raise self.read_error
        packet = FramePacket(self.frames_read, self.clock.now, np.zeros((24, 32, 3), dtype=np.uint8))
        self.frames_read += 1
        return packet

    def stop(self):
        self.events.append("camera.stop")
        if self.stop_error:
            raise self.stop_error


class FakeAnalysis:
    def __init__(self, events, ready_after=0):
        self.events = events
        self.ready_after = ready_after
        self.reads = 0
        self.started = False
        self.reported_ready = False
        self.device = "cpu"
        self.error = ""
        self.submitted = []
        self.pending = None
        self.reset_count = 0
        self.start_error = None
        self.read_error = None
        self.stop_error = None
        self.predict_first_only = False

    @property
    def ready(self):
        ready = self.started and self.reads >= self.ready_after
        if ready and not self.reported_ready:
            self.reported_ready = True
            self.events.append("analysis.ready")
        return ready

    def start(self):
        self.events.append("analysis.start")
        if self.start_error:
            raise self.start_error
        self.started = True

    def submit(self, packet):
        self.submitted.append(packet.frame_id)
        if not self.predict_first_only or packet.frame_id == 0:
            self.pending = AnalysisResult(
                packet.frame_id, packet.timestamp, PersonBox(0.2, 0.1, 0.8, 0.9),
                (Candidate("02", "Aluno", 0.9),), caption="Aluno",
                prediction_timestamp=packet.timestamp,
            )

    def read_latest(self):
        if self.read_error:
            raise self.read_error
        self.reads += 1
        result, self.pending = self.pending, None
        return result

    def reset(self):
        self.events.append("analysis.reset")
        self.reset_count += 1
        self.pending = None

    def stop(self):
        self.events.append("analysis.stop")
        if self.stop_error:
            raise self.stop_error


class FakeRenderer:
    def __init__(self, events):
        self.events = events
        self.calls = []
        self.reset_count = 0

    def render(self, frame, result, **options):
        self.calls.append((frame, result, options))
        return frame.image.copy()

    def reset(self):
        self.events.append("renderer.reset")
        self.reset_count += 1


class FakeDisplay:
    def __init__(self, events, keys):
        self.events = events
        self.keys = iter(keys)
        self.contexts = []
        self.images = []

    def _record(self, operation):
        self.events.append("display." + operation)
        self.contexts.append((operation, os.getpid(), threading.get_ident()))

    def open(self):
        self._record("open")

    def show(self, image):
        self._record("show")
        self.images.append(image)

    def poll_key(self):
        return next(self.keys, 27)

    def close(self):
        self._record("close")


class CameraApplicationTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.clock = FakeClock()
        self.camera = FakeCamera(self.events, self.clock)
        self.analysis = FakeAnalysis(self.events)
        self.renderer = FakeRenderer(self.events)

    def run_application(self, keys=(27,)):
        self.display = FakeDisplay(self.events, keys)
        application = CameraApplication(
            RuntimeConfig(width=32, height=24), camera=self.camera,
            analysis=self.analysis, renderer=self.renderer, display=self.display,
        )
        with patch("application.runtime.app.time.monotonic", side_effect=self.clock.monotonic), patch(
            "application.runtime.app.time.sleep", side_effect=self.clock.sleep,
        ):
            return application.run()

    def assert_resources_closed(self):
        self.assertEqual(self.events[-3:], ["camera.stop", "analysis.stop", "display.close"])
        for operation in ("camera.stop", "analysis.stop", "display.close"):
            self.assertEqual(self.events.count(operation), 1)

    def test_escape_closes_all_resources_and_returns_session_statistics(self):
        stats = self.run_application()
        self.assertEqual(stats.captured_frames, 1)
        self.assertEqual(stats.analyzed_frames, 1)
        self.assertEqual(stats.predictions, 1)
        self.assertEqual(stats.last_candidates, [{"label": "Aluno", "confidence": 0.9}])
        self.assertEqual(self.analysis.submitted, [0])
        self.assertEqual(len(self.display.images), 1)
        self.assert_resources_closed()

    def test_window_lifecycle_stays_in_calling_main_process_and_thread(self):
        self.run_application()
        self.assertEqual(self.events[0], "display.open")
        for _, process_id, thread_id in self.display.contexts:
            self.assertEqual(process_id, os.getpid())
            self.assertEqual(thread_id, threading.main_thread().ident)
        self.assertEqual(self.display.contexts[0][0], "open")
        self.assertEqual(self.display.contexts[-1][0], "close")

    def test_analysis_and_camera_reported_errors_close_resources(self):
        for component in ("analysis", "camera"):
            with self.subTest(component=component):
                self.setUp()
                getattr(self, component).error = "Falha simulada em " + component
                with self.assertRaisesRegex(RuntimeError, "Falha simulada"):
                    self.run_application()
                self.assert_resources_closed()

    def test_start_and_read_failures_close_resources(self):
        for component, operation in (("analysis", "start"), ("camera", "start"),
                                     ("analysis", "read"), ("camera", "read")):
            with self.subTest(component=component, operation=operation):
                self.setUp()
                setattr(getattr(self, component), operation + "_error", RuntimeError("Falha simulada"))
                with self.assertRaisesRegex(RuntimeError, "Falha simulada"):
                    self.run_application()
                self.assert_resources_closed()

    def test_pause_resume_and_clear_reset_state_and_skip_paused_frames(self):
        self.analysis.predict_first_only = True
        self.run_application(keys=(32, -1, 32, ord("r"), ord("R"), 27))
        self.assertEqual(self.analysis.submitted, [0, 3, 4, 5])
        self.assertEqual(self.analysis.reset_count, 4)
        self.assertEqual(self.renderer.reset_count, 4)
        self.assertEqual([call[2]["paused"] for call in self.renderer.calls],
                         [False, True, True, False, False, False])
        self.assertEqual(self.renderer.calls[0][1].caption, "Aluno")
        self.assertTrue(all(call[1] is None for call in self.renderer.calls[1:]))
        self.assert_resources_closed()

    def test_camera_starts_only_after_model_is_ready_preserving_first_video_frame(self):
        self.analysis.ready_after = 2
        self.run_application(keys=(-1, -1, 27))
        self.assertLess(self.events.index("analysis.ready"), self.events.index("camera.start"))
        self.assertEqual(self.events.count("camera.start"), 1)
        self.assertEqual(self.analysis.submitted, [0])
        self.assertEqual(self.camera.frames_read, 1)
        self.assertEqual(self.renderer.calls[0][0].frame_id, -1)
        self.assertEqual(self.renderer.calls[0][1].status, "Carregando modelo…")
        self.assertEqual(self.renderer.calls[-1][0].frame_id, 0)
        self.assert_resources_closed()

    def test_analysis_and_display_close_even_when_camera_stop_fails(self):
        self.camera.stop_error = RuntimeError("Não foi possível fechar captura")
        with self.assertRaisesRegex(RuntimeError, "fechar captura"):
            self.run_application()
        self.assert_resources_closed()

    def test_display_closes_even_when_analysis_stop_fails(self):
        self.analysis.stop_error = RuntimeError("Não foi possível fechar análise")
        with self.assertRaisesRegex(RuntimeError, "fechar análise"):
            self.run_application()
        self.assert_resources_closed()


if __name__ == "__main__":
    unittest.main()
