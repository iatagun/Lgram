"""
Tests for token-conscious LLM helpers.
"""

from __future__ import annotations

import unittest

from lgram.essay.deep_grammar import DeepGrammarCheck
from lgram.essay.layer_grammar import GrammarLayer
from lgram.essay.layer_llm_content import LLMContentAnalyzer


class TestTokenConsciousHelpers(unittest.TestCase):

    def test_content_focus_text_keeps_structure(self):
        text = (
            "Introduction sentence one. Introduction sentence two. "
            + "Middle sentence. " * 20
            + "Conclusion sentence one. Conclusion sentence two."
        )
        excerpt = LLMContentAnalyzer._select_focus_text(text, max_chars=200)
        self.assertLessEqual(len(excerpt), 200)
        self.assertIn("Introduction sentence one.", excerpt)
        self.assertIn("Conclusion sentence two.", excerpt)

    def test_deep_grammar_focus_text_keeps_structure(self):
        text = (
            "Opening idea. " + "Body sentence. " * 30 + "Final idea. Closing sentence."
        )
        excerpt = DeepGrammarCheck._select_focus_text(text, max_chars=160)
        self.assertLessEqual(len(excerpt), 160)
        self.assertIn("Opening idea.", excerpt)
        self.assertIn("Closing sentence.", excerpt)

    def test_grammar_layer_deep_check_gating(self):
        layer = GrammarLayer()
        layer._lt = object()
        layer._init_attempted = True

        spelling_only = {
            "total_errors": 3,
            "grammar_errors": 0,
            "spelling_errors": 3,
            "pronoun_errors": 0,
        }
        grammar_issues = {
            "total_errors": 1,
            "grammar_errors": 1,
            "spelling_errors": 0,
            "pronoun_errors": 0,
        }

        self.assertFalse(layer._should_run_deep_check(spelling_only, word_count=120))
        self.assertTrue(layer._should_run_deep_check(grammar_issues, word_count=120))

    def test_grammar_layer_allows_deep_check_when_lt_unavailable(self):
        layer = GrammarLayer()
        layer._lt = None
        layer._init_attempted = True

        self.assertTrue(layer._should_run_deep_check({}, word_count=120))


if __name__ == "__main__":
    unittest.main()
