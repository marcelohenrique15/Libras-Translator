import unittest

from application.recognition.captions import CaptionPolicy, CaptionState
from application.types import Candidate


class CaptionPolicyTests(unittest.TestCase):
    first = (Candidate("01", "Acontecer", 0.9),)
    second = (Candidate("02", "Aluno", 0.9),)

    def test_first_prediction_is_visible_as_partial_before_confirmation(self):
        policy = CaptionPolicy()
        first = policy.update(self.first, 1)
        self.assertEqual(first.text, "Acontecer")
        self.assertTrue(first.is_partial)
        second = policy.update(self.first, 1.2)
        self.assertEqual(second.text, "Acontecer")
        self.assertFalse(second.is_partial)
        self.assertEqual(second.timestamp, 1.2)
        self.assertEqual(second.expires_at, 3.2)

    def test_short_high_confidence_prefixes_remain_partial_until_confirmation_allowed(self):
        policy = CaptionPolicy()
        for timestamp in (0, 0.1, 0.2):
            state = policy.update(self.first, timestamp, allow_confirmation=False)
            self.assertEqual(state.text, "Acontecer")
            self.assertTrue(state.is_partial)
        self.assertFalse(policy.update(self.first, 0.3, allow_confirmation=True).is_partial)

    def test_render_queries_and_tracking_failures_do_not_extend_stable_caption(self):
        policy = CaptionPolicy()
        policy.update(self.first, 0)
        confirmed = policy.update(self.first, 0.1)
        policy.break_agreement()
        for timestamp in (0.2, 1, 2):
            self.assertEqual(policy.get(timestamp), confirmed)
        self.assertEqual(policy.get(2.1), CaptionState())
        self.assertEqual(policy.get(3), CaptionState())

    def test_low_confidence_breaks_agreement_but_preserves_existing_stable_text(self):
        policy = CaptionPolicy()
        policy.update(self.first, 0)
        confirmed = policy.update(self.first, 0.1)
        self.assertEqual(policy.update((Candidate("01", "Acontecer", 0.6),), 0.2), confirmed)
        self.assertEqual(policy.update((), 0.3), confirmed)
        self.assertEqual(policy.update((Candidate("02", "Aluno", 0.1),), 0.4), confirmed)
        self.assertEqual(policy.get(2.1), CaptionState())

    def test_partial_text_is_cleared_on_missing_or_weak_prediction(self):
        for candidates in ((), (Candidate("01", "Acontecer", 0.2),)):
            with self.subTest(candidates=candidates):
                policy = CaptionPolicy()
                policy.update(self.first, 0)
                self.assertEqual(policy.update(candidates, 0.1), CaptionState())
                self.assertTrue(policy.update(self.first, 0.2).is_partial)
        policy.break_agreement()
        self.assertEqual(policy.get(0.3), CaptionState())

    def test_new_hypothesis_replaces_old_stable_word_without_reviving_it_after_expiry(self):
        policy = CaptionPolicy()
        policy.update(self.first, 0)
        policy.update(self.first, 0.1)
        new = policy.update(self.second, 0.2)
        self.assertEqual(new.text, "Aluno")
        self.assertTrue(new.is_partial)
        self.assertEqual(policy.get(0.9), CaptionState())
        self.assertEqual(policy.get(1.0), CaptionState())
        self.assertTrue(policy.update(self.second, 1.1).is_partial)

    def test_new_confirmations_refresh_expiry_without_accumulating_duplicate_words(self):
        policy = CaptionPolicy()
        for timestamp in (0, 0.1, 0.2, 0.3):
            policy.update(self.first, timestamp)
        state = policy.get(1)
        self.assertEqual(state.text, "Acontecer")
        self.assertEqual(state.timestamp, 0.3)
        self.assertEqual(state.expires_at, 2.3)
        self.assertFalse(state.is_partial)

    def test_reset_clears_text_and_allows_a_new_session_clock(self):
        policy = CaptionPolicy()
        policy.update(self.first, 3)
        policy.update(self.first, 3.1)
        policy.reset()
        self.assertEqual(policy.get(3.2), CaptionState())
        self.assertTrue(policy.update(self.first, 0).is_partial)

    def test_repeated_or_old_prediction_timestamp_is_rejected(self):
        policy = CaptionPolicy()
        policy.update(self.first, 1)
        for timestamp in (1, 0.9, float("nan")):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                policy.update(self.first, timestamp)

    def test_expired_partial_cannot_complete_an_old_confirmation_sequence(self):
        policy = CaptionPolicy(partial_seconds=0.5)
        policy.update(self.first, 0)
        self.assertEqual(policy.get(0.6), CaptionState())
        self.assertTrue(policy.update(self.first, 0.7).is_partial)


if __name__ == "__main__":
    unittest.main()
