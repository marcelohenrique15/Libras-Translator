import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from application.ui.overlay import OverlayRenderer
from application.types import AnalysisResult, Candidate, FramePacket, PersonBox


class OverlayRendererTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((480, 640, 3), 100, dtype=np.uint8)
        self.frame = FramePacket(10, 10.0, self.image)
        self.box = PersonBox(0.1, 0.05, 0.5, 0.85)
        self.result = AnalysisResult(
            frame_id=10, timestamp=10.0, box=self.box,
            candidates=(Candidate("01", "Acontecer", 0.8), Candidate("02", "Aluno", 0.15),
                        Candidate("03", "Amarelo", 0.05), Candidate("04", "Não exibir", 0.0)),
            caption="Acontecer", status="Sinal reconhecido", prediction_timestamp=10.0,
        )

    def _texts(self, renderer, result, now=10.1):
        with patch.object(renderer, "_text", wraps=renderer._text) as draw:
            renderer.render(self.frame, result, now=now)
        return [call.args[1] for call in draw.call_args_list]

    def test_render_does_not_mutate_shared_camera_frame(self):
        before = self.image.copy()
        output = OverlayRenderer().render(self.frame, self.result, now=10.1)
        np.testing.assert_array_equal(self.image, before)
        self.assertFalse(np.shares_memory(output, self.image))
        self.assertEqual(output.shape, self.image.shape)
        self.assertFalse(np.array_equal(output, before))

    def test_mirror_changes_only_display_and_person_position(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[:, :10] = (10, 20, 30)
        packet = FramePacket(1, 10.0, image)
        renderer = OverlayRenderer(mirror=True)
        output = renderer.render(packet, None, now=10.0)
        np.testing.assert_array_equal(output, image[:, ::-1])
        box = renderer._display_box(self.box)
        self.assertAlmostEqual(box.left, 0.5)
        self.assertAlmostEqual(box.right, 0.9)
        self.assertEqual(self.box.left, 0.1)

    def test_three_candidates_and_unicode_labels(self):
        texts = self._texts(OverlayRenderer(), self.result)
        self.assertIn("Acontecer", texts)
        self.assertIn("Aluno", texts)
        self.assertIn("Amarelo", texts)
        self.assertNotIn("Não exibir", texts)
        self.assertIn("80%", texts)
        self.assertIn("MAIORES CONFIANÇAS", texts)

    def test_candidates_stay_in_fixed_corner_when_person_moves(self):
        renderer = OverlayRenderer(mirror=False)
        positions = []
        for box in (self.box, PersonBox(0.5, 0.05, 0.9, 0.85)):
            with patch.object(renderer, "_text", wraps=renderer._text) as draw:
                renderer.render(self.frame, replace(self.result, box=box), now=10.1)
            title = next(call for call in draw.call_args_list if call.args[1] == "MAIORES CONFIANÇAS")
            positions.append(title.args[2:4])
        self.assertEqual(positions[0], positions[1])
        self.assertGreater(positions[0][0], self.image.shape[1] // 2)

    def test_only_confirmed_leading_word_follows_person(self):
        renderer = OverlayRenderer()
        with patch.object(renderer, "_draw_caption", wraps=renderer._draw_caption) as draw:
            renderer.render(self.frame, self.result, now=10.1)
            self.assertEqual(draw.call_count, 1)
            self.assertEqual(draw.call_args.args[2], "Acontecer")
        for caption in ("", "Aluno"):
            with patch.object(renderer, "_draw_caption") as draw:
                renderer.render(self.frame, replace(self.result, caption=caption), now=10.1)
                draw.assert_not_called()

    def test_upper_left_corner_has_no_brand_badge(self):
        renderer = OverlayRenderer(mirror=False)
        output = renderer.render(self.frame, None, now=10.1)
        np.testing.assert_array_equal(output[:80, :200], self.image[:80, :200])
        texts = self._texts(renderer, self.result)
        self.assertFalse(any("LIBRAS" in text or "AO VIVO" in text for text in texts))

    def test_confirmed_caption_survives_missing_or_changed_candidates_until_its_expiry(self):
        renderer = OverlayRenderer()
        held = replace(
            self.result, timestamp=10.6, candidates=(), prediction_timestamp=9.0,
            caption_timestamp=10.0, caption_expires_at=12.0,
        )
        self.assertIn("Acontecer", self._texts(renderer, held, now=10.7))
        self.assertNotIn("80%", self._texts(renderer, held, now=10.7))
        changed = replace(held, candidates=(Candidate("02", "Aluno", 0.9),), prediction_timestamp=10.6)
        self.assertIn("Acontecer", self._texts(renderer, changed, now=10.7))
        # Um resultado novo não prolonga o TTL da legenda anterior.
        expired = replace(held, timestamp=12.1)
        self.assertNotIn("Acontecer", self._texts(renderer, expired, now=12.1))

    def test_partial_caption_is_visibly_revisable_and_stable_caption_has_no_ellipsis(self):
        renderer = OverlayRenderer()
        partial = replace(
            self.result, caption="Aluno", caption_is_partial=True,
            caption_timestamp=10.0, caption_expires_at=10.6,
        )
        with patch.object(renderer, "_text", wraps=renderer._text) as draw:
            renderer.render(self.frame, partial, now=10.1)
        label = next(call for call in draw.call_args_list if call.args[1] == "Aluno…")
        self.assertEqual(label.args[5], renderer.MUTED)
        stable = replace(partial, caption_is_partial=False)
        texts = self._texts(renderer, stable)
        self.assertIn("Aluno", texts)
        self.assertNotIn("Aluno…", texts)

    def test_clocked_caption_resumes_after_brief_tracking_loss_without_extending_its_ttl(self):
        renderer = OverlayRenderer()
        retained = replace(self.result, caption_timestamp=10.0, caption_expires_at=12.0)
        self.assertIn("Acontecer", self._texts(renderer, retained))
        missing = replace(retained, timestamp=10.2, box=None, caption="")
        self.assertNotIn("Acontecer", self._texts(renderer, missing, now=10.2))
        returned = replace(retained, timestamp=10.3, candidates=())
        self.assertIn("Acontecer", self._texts(renderer, returned, now=10.3))
        expired = replace(returned, timestamp=12.1)
        self.assertNotIn("Acontecer", self._texts(renderer, expired, now=12.1))

    def test_person_jump_after_brief_tracking_loss_still_hides_retained_caption(self):
        renderer = OverlayRenderer()
        retained = replace(self.result, caption_timestamp=10.0, caption_expires_at=12.0)
        self._texts(renderer, retained)
        missing = replace(retained, timestamp=10.2, box=None, caption="")
        self._texts(renderer, missing, now=10.2)
        moved = replace(retained, timestamp=10.3, box=PersonBox(0.75, 0.05, 0.95, 0.85))
        self.assertNotIn("Acontecer", self._texts(renderer, moved, now=10.3))

    def test_pause_hides_retained_caption_and_future_caption_is_not_displayed(self):
        renderer = OverlayRenderer()
        retained = replace(self.result, caption_timestamp=10.0, caption_expires_at=12.0)
        with patch.object(renderer, "_draw_caption") as draw:
            renderer.render(self.frame, retained, paused=True, now=10.1)
            draw.assert_not_called()
        future = replace(retained, caption_timestamp=10.5)
        with patch.object(renderer, "_draw_caption") as draw:
            renderer.render(self.frame, future, now=10.1)
            draw.assert_not_called()

    def test_stale_or_missing_person_removes_old_caption(self):
        renderer = OverlayRenderer()
        self.assertIn("Acontecer", self._texts(renderer, self.result))
        self.assertNotIn("Acontecer", self._texts(renderer, self.result, now=11.0))
        absent = AnalysisResult(11, 11.0, None, self.result.candidates, "Acontecer")
        self.assertNotIn("Acontecer", self._texts(renderer, absent, now=11.1))
        self.assertIsNone(renderer._position)

    def test_current_result_status_is_visible_without_person(self):
        for status in ("Carregando modelo…", "Sem pessoa visível"):
            result = AnalysisResult(11, 10.0, None, self.result.candidates, "Acontecer", status)
            texts = self._texts(OverlayRenderer(), result)
            self.assertIn(status, texts)
            self.assertNotIn("Acontecer", texts)

    def test_footer_shortcuts_and_statistics_do_not_overlap_at_960_pixels(self):
        renderer = OverlayRenderer()
        frame = FramePacket(10, 10.0, np.zeros((540, 960, 3), dtype=np.uint8))
        result = AnalysisResult(10, 10.0, self.box, inference_ms=12.3, analysis_ms=42.0)
        with patch.object(renderer, "_text", wraps=renderer._text) as draw:
            renderer.render(frame, result, now=10.1, device="cpu", capture_fps=30.0, analysis_fps=7.4)
        shortcuts = next(call for call in draw.call_args_list if call.args[1].startswith("ESC sair"))
        statistics = next(call for call in draw.call_args_list if "fps câmera" in call.args[1])
        self.assertLess(shortcuts.args[2] + shortcuts.kwargs["max_width"], statistics.args[2])
        self.assertLessEqual(statistics.args[2] + statistics.kwargs["max_width"], 960)
        self.assertIn("atraso 100 ms", statistics.args[1])
        self.assertIn("modelo 12 ms", statistics.args[1])

    def test_old_prediction_does_not_follow_new_person(self):
        renderer = OverlayRenderer()
        self._texts(renderer, self.result)
        self._texts(renderer, AnalysisResult(11, 10.2, None), now=10.2)
        returned = AnalysisResult(12, 10.3, self.box, self.result.candidates,
                                  "Acontecer", "Coletando movimento", prediction_timestamp=10.0)
        self.assertNotIn("Acontecer", self._texts(renderer, returned, now=10.3))
        updated = AnalysisResult(13, 10.4, self.box, self.result.candidates,
                                 "Acontecer", prediction_timestamp=10.4)
        self.assertIn("Acontecer", self._texts(renderer, updated, now=10.4))

    def test_recent_person_with_expired_prediction_hides_candidates(self):
        result = AnalysisResult(30, 12.0, self.box, self.result.candidates,
                                "Acontecer", "Coletando movimento", prediction_timestamp=10.0)
        texts = self._texts(OverlayRenderer(), result, now=12.1)
        self.assertNotIn("Acontecer", texts)
        self.assertNotIn("80%", texts)
        self.assertIn("Coletando movimento", texts)

    def test_large_person_jump_does_not_carry_previous_prediction(self):
        renderer = OverlayRenderer()
        self._texts(renderer, self.result)
        moved = AnalysisResult(12, 10.3, PersonBox(0.75, 0.05, 0.95, 0.85),
                               self.result.candidates, "Acontecer", "Coletando movimento",
                               prediction_timestamp=10.0)
        self.assertNotIn("Acontecer", self._texts(renderer, moved, now=10.3))

    def test_pause_hides_candidates_and_identifies_paused_display(self):
        renderer = OverlayRenderer()
        with patch.object(renderer, "_text", wraps=renderer._text) as draw:
            renderer.render(self.frame, self.result, paused=True, now=10.1)
        texts = [call.args[1] for call in draw.call_args_list]
        self.assertNotIn("Acontecer", texts)
        self.assertIn("Pausado", texts)
        self.assertFalse(any("LIBRAS" in text for text in texts))

    def test_panel_stays_inside_image_at_all_edges_and_after_resize(self):
        renderer = OverlayRenderer()
        for width, height in [(640, 480), (160, 120), (1920, 1080), (200, 150)]:
            panel_width, panel_height = min(130, width), min(60, height)
            for box in (PersonBox(-1, -1, 0, 0), PersonBox(1, 1, 2, 2), self.box):
                x, y = renderer._panel_position(box, panel_width, panel_height, width, height)
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x + panel_width, width)
                self.assertLessEqual(y + panel_height, height)

    def test_caption_tracks_chest_center_without_drifting_to_free_sides(self):
        renderer = OverlayRenderer(mirror=False)
        for anchor in ((.3, .5), (.4, .6), (.65, .5)):
            x, y = renderer._panel_position(self.box, 200, 50, 1000, 600, anchor)
            self.assertEqual((x + 100, y + 25), (round(anchor[0] * 1000), round(anchor[1] * 600)))

    def test_caption_anchor_respects_mirror_and_does_not_depend_on_arm_box(self):
        anchor = (.3, .55)
        for mirror, expected_x in ((False, .3), (True, .7)):
            renderer = OverlayRenderer(mirror=mirror)
            for box in (self.box, PersonBox(0, .05, 1, .85)):
                result = replace(self.result, box=box, caption_anchor=anchor)
                with patch.object(renderer, "_panel_position", wraps=renderer._panel_position) as position:
                    renderer.render(self.frame, result, now=10.1)
                np.testing.assert_allclose(position.call_args.args[-1], (expected_x, .55))

    def test_caption_keeps_its_anchor_when_near_fixed_candidates(self):
        renderer = OverlayRenderer()
        x, y = renderer._panel_position(self.box, 100, 50, 1000, 600, (.8, .2))
        self.assertEqual((x + 50, y + 25), (800, 120))

    def test_tiny_frames_and_long_labels_do_not_fail(self):
        renderer = OverlayRenderer()
        for width, height in [(1, 1), (30, 20), (160, 120), (200, 150)]:
            frame = FramePacket(1, 10.0, np.zeros((height, width, 3), dtype=np.uint8))
            result = AnalysisResult(1, 10.0, self.box, (Candidate("1", "Água " * 200, 1.0),), "Legenda " * 200)
            output = renderer.render(frame, result, now=10.1)
            self.assertEqual(output.shape, frame.image.shape)

    def test_reset_forgets_tracking_and_font_is_cached(self):
        renderer = OverlayRenderer()
        self._texts(renderer, self.result)
        self.assertIsNotNone(renderer._position)
        self.assertIs(renderer._font(14), renderer._font(14))
        renderer.reset()
        self.assertIsNone(renderer._position)
        self.assertIsNone(renderer._last_box)


if __name__ == "__main__":
    unittest.main()
