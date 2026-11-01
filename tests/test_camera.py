import queue
import multiprocessing as mp
import os
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from application.capture.camera import CameraSource, _capture_worker, _replace_frame
from application.types import FramePacket


def _blocked_capture(*args):
    """Simula um read() nativo que não responde ao pedido cooperativo."""
    time.sleep(60)


def _read_large_video(video, results):
    """O supervisor do teste pode encerrar este leitor se um pipe travar."""
    with CameraSource(video, width=960, height=540, fps=30) as source:
        results.put(("started", source._process.pid))
        packets = 0
        last_shape = None
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            packet = source.read_latest()
            if packet is not None:
                packets += 1
                last_shape = packet.image.shape
            if source.finished:
                break
            time.sleep(0.01)
        results.put(("finished", source.finished, source.error, packets, last_shape))
    results.close()
    results.join_thread()


class LocalQueue(queue.Queue):
    def cancel_join_thread(self):
        pass

    def close(self):
        pass

    def join_thread(self):
        pass


class CameraSourceTests(unittest.TestCase):
    def test_invalid_dimensions_and_frame_rate_are_rejected(self):
        for options in ({"width": 0}, {"height": -1}, {"fps": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CameraSource(**options)

    def test_replacement_keeps_only_the_newest_waiting_frame(self):
        frames = queue.Queue(maxsize=1)
        first = FramePacket(1, 1.0, np.zeros((2, 2, 3), dtype=np.uint8))
        newest = FramePacket(2, 2.0, np.ones((2, 2, 3), dtype=np.uint8))
        _replace_frame(frames, first)
        _replace_frame(frames, newest)
        self.assertIs(frames.get_nowait(), newest)
        with self.assertRaises(queue.Empty):
            frames.get_nowait()

    def test_replacement_does_not_wait_for_a_multiprocessing_feeder(self):
        class PendingQueue:
            def put_nowait(self, value):
                raise queue.Full

            def get_nowait(self):
                raise queue.Empty

        _replace_frame(PendingQueue(), FramePacket(0, 0.0, np.zeros((2, 2, 3))))

    def test_read_is_nonblocking_and_returns_a_read_only_image(self):
        source = CameraSource()
        source._frames = LocalQueue(maxsize=1)
        self.assertIsNone(source.read_latest())
        source._frames.put_nowait(FramePacket(3, 3.0, np.zeros((2, 2, 3))))
        packet = source.read_latest()
        self.assertEqual(packet.frame_id, 3)
        with self.assertRaises(ValueError):
            packet.image[0, 0] = 1
        self.assertIsNone(source.read_latest())
        source.stop()
        source.stop()
        self.assertTrue(source.finished)

    def test_camera_read_failure_has_finite_retries_and_releases_resource(self):
        class Capture:
            reads = 0
            released = False

            def isOpened(self):
                return True

            def set(self, *_):
                return True

            def read(self):
                self.reads += 1
                return False, None

            def release(self):
                self.released = True

        capture = Capture()
        cv2 = SimpleNamespace(
            VideoCapture=lambda *_: capture,
            CAP_PROP_FRAME_WIDTH=1,
            CAP_PROP_FRAME_HEIGHT=2,
            CAP_PROP_FPS=3,
            CAP_PROP_BUFFERSIZE=4,
        )
        finished = threading.Event()
        error = SimpleNamespace(value="")
        with patch.dict("sys.modules", {"cv2": cv2}):
            _capture_worker(
                0, 80, 64, 30, None, LocalQueue(maxsize=1), threading.Event(),
                finished, error,
            )
        self.assertEqual(capture.reads, 5)
        self.assertTrue(capture.released)
        self.assertTrue(finished.is_set())
        self.assertIn("5 tentativas", error.value)

    def test_unavailable_source_reports_error_in_the_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            source = CameraSource(str(Path(directory) / "missing.avi"))
            self.addCleanup(source.stop)
            source.start()
            deadline = time.monotonic() + 10
            while not source.finished and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(source.finished)
            self.assertIn("Não foi possível abrir", source.error)
            source.stop()
            self.assertFalse(source._process.is_alive())

    def test_video_uses_recorded_fps_and_falls_back_if_it_is_invalid(self):
        for recorded_fps, expected_fps in ((12.0, 12.0), (float("nan"), 60.0)):
            with self.subTest(recorded_fps=recorded_fps):
                clock = [0.0]
                images = iter([np.zeros((2, 2, 3), dtype=np.uint8)] * 2)

                def read_frame():
                    image = next(images, None)
                    return image is not None, image

                capture = SimpleNamespace(
                    isOpened=lambda: True,
                    get=lambda _: recorded_fps,
                    read=read_frame,
                    release=lambda: None,
                )
                cv2 = SimpleNamespace(VideoCapture=lambda _: capture, CAP_PROP_FPS=3)

                def advance(seconds):
                    clock[0] += seconds

                stopped = SimpleNamespace(is_set=lambda: False, wait=advance)
                finished = threading.Event()
                with patch.dict("sys.modules", {"cv2": cv2}), patch(
                    "application.capture.camera.time.monotonic", side_effect=lambda: clock[0],
                ):
                    _capture_worker(
                        "video.avi", 2, 2, 60, None, LocalQueue(maxsize=1), stopped,
                        finished, SimpleNamespace(value=""),
                    )
                self.assertTrue(finished.is_set())
                self.assertAlmostEqual(clock[0], 2 / expected_fps)

    def test_small_video_is_read_in_a_spawned_process_until_eof(self):
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            video = str(Path(directory) / "capture.avi")
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 15, (80, 64))
            if not writer.isOpened():
                self.skipTest("Codec MJPG indisponível neste OpenCV.")
            try:
                for intensity in range(8):
                    writer.write(np.full((64, 80, 3), intensity * 20, dtype=np.uint8))
            finally:
                writer.release()

            source = CameraSource(video, width=960, height=540, fps=15)
            self.addCleanup(source.stop)
            source.start()
            process = source._process
            source.start()
            self.assertIs(source._process, process)
            packets = []
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                packet = source.read_latest()
                if packet is not None:
                    packets.append(packet)
                if source.finished:
                    break
                time.sleep(0.01)
            self.assertTrue(source.finished)
            self.assertIsNone(source.error)
            self.assertGreater(len(packets), 0)
            ids = [packet.frame_id for packet in packets]
            self.assertEqual(ids, sorted(set(ids)))
            self.assertTrue(all(packet.image.shape == (64, 80, 3) for packet in packets))
            self.assertTrue(all(not packet.image.flags.writeable for packet in packets))
            self.assertTrue(all(packet.timestamp <= time.monotonic() for packet in packets))
            source.stop()
            self.assertFalse(process.is_alive())

    def test_stop_terminates_only_its_own_unresponsive_process(self):
        source = CameraSource()
        self.addCleanup(source.stop)
        with patch("application.capture.camera._capture_worker", _blocked_capture):
            source.start()
        process = source._process
        started = time.monotonic()
        source.stop()
        source.stop()
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertFalse(process.is_alive())
        self.assertTrue(source.finished)
        self.assertIsNone(source.read_latest())

    def test_large_video_is_downscaled_without_distortion_and_finishes_eof(self):
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            video = str(Path(directory) / "large.avi")
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 15, (1280, 960))
            if not writer.isOpened():
                self.skipTest("Codec MJPG indisponível neste OpenCV.")
            try:
                for intensity in range(8):
                    writer.write(np.full((960, 1280, 3), intensity * 20, dtype=np.uint8))
            finally:
                writer.release()

            context = mp.get_context("spawn")
            results = context.Queue(maxsize=2)
            reader = context.Process(target=_read_large_video, args=(video, results))
            capture_pid = None
            try:
                reader.start()
                started, capture_pid = results.get(timeout=10)
                self.assertEqual(started, "started")
                event, finished, error, count, shape = results.get(timeout=10)
                self.assertEqual(event, "finished")
                self.assertTrue(finished)
                self.assertIsNone(error)
                self.assertGreater(count, 0)
                self.assertEqual(shape, (540, 720, 3))
                self.assertAlmostEqual(shape[1] / shape[0], 1280 / 960)
                reader.join(timeout=4)
                self.assertFalse(reader.is_alive())
                self.assertEqual(reader.exitcode, 0)
            finally:
                if reader.is_alive():
                    reader.terminate()
                    reader.join(timeout=1)
                    if capture_pid is not None:
                        try:
                            os.kill(capture_pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                results.close()
                results.cancel_join_thread()

    def test_webcam_that_ignores_resolution_is_downscaled_before_ipc(self):
        import cv2

        stopped = threading.Event()
        finished = threading.Event()
        frames = LocalQueue(maxsize=1)
        original = np.zeros((960, 1280, 3), dtype=np.uint8)

        def read_frame():
            stopped.set()
            return True, original

        capture = SimpleNamespace(
            isOpened=lambda: True, set=lambda *_: True,
            read=read_frame, release=lambda: None,
        )
        with patch("cv2.VideoCapture", return_value=capture):
            _capture_worker(
                0, 960, 540, 30, None, frames, stopped, finished,
                SimpleNamespace(value=""),
            )
        image = frames.get_nowait().image
        self.assertEqual(image.shape, (540, 720, 3))
        self.assertEqual(original.shape, (960, 1280, 3))
        self.assertTrue(finished.is_set())


if __name__ == "__main__":
    unittest.main()
