import os
import unittest
from unittest.mock import patch

from application.ui.display import OpenCVDisplay


class OpenCVDisplayTests(unittest.TestCase):
    def test_default_window_is_larger_than_capture(self):
        display = OpenCVDisplay()
        self.assertEqual((display.width, display.height), (1280, 720))

    def test_open_uses_configured_window_size(self):
        display = OpenCVDisplay("Teste", width=1600, height=900)
        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
                patch("cv2.namedWindow") as create, patch("cv2.resizeWindow") as resize:
            display.open()
        create.assert_called_once()
        resize.assert_called_once_with("Teste", 1600, 900)
        self.assertTrue(display._opened)

    def test_invalid_window_size_is_rejected(self):
        for width, height in ((0, 720), (1280, -1), (True, 720), (1280, 720.5)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                OpenCVDisplay(width=width, height=height)

    def test_close_is_idempotent(self):
        display = OpenCVDisplay()
        display._opened = True
        with patch("cv2.destroyWindow") as destroy:
            display.close()
            display.close()
        destroy.assert_called_once_with(display.title)


if __name__ == "__main__":
    unittest.main()
