"""
Test suite for Centering-Lgram.
"""

import os
import tempfile
import unittest


class TestCentering(unittest.TestCase):

    def test_import(self):
        import lgram
        self.assertTrue(hasattr(lgram, "__version__"))
        self.assertTrue(hasattr(lgram, "EnhancedCenteringTheory"))
        self.assertTrue(hasattr(lgram, "TransitionType"))
        self.assertTrue(hasattr(lgram, "CenteringState"))
        self.assertTrue(hasattr(lgram, "DiscourseEntity"))

    def test_transition_enum(self):
        from lgram import TransitionType
        self.assertIn(TransitionType.ESTABLISH, list(TransitionType))
        self.assertEqual(TransitionType.CONTINUE.value, "Continue")
        self.assertEqual(TransitionType.RETAIN.value, "Retain")
        self.assertEqual(TransitionType.SMOOTH_SHIFT.value, "Smooth-Shift")
        self.assertEqual(TransitionType.ROUGH_SHIFT.value, "Rough-Shift")

    def test_centering_state(self):
        from lgram import CenteringState, TransitionType
        state = CenteringState(
            utterance="John went to the store.",
            forward_centers=["john", "store"],
            backward_center="john",
            preferred_center="john",
            transition=TransitionType.CONTINUE,
        )
        self.assertEqual(state.preferred_center, "john")
        self.assertEqual(len(state.forward_centers), 2)

    def test_first_utterance_establish(self):
        import spacy
        from lgram import EnhancedCenteringTheory, TransitionType

        nlp = spacy.load("en_core_web_sm")
        ct = EnhancedCenteringTheory(nlp)

        state = ct.analyze_utterance("John went to the store.")
        self.assertEqual(state.preferred_center, "john")
        self.assertEqual(state.transition, TransitionType.ESTABLISH)

    def test_continue_transition(self):
        import spacy
        from lgram import EnhancedCenteringTheory, TransitionType

        nlp = spacy.load("en_core_web_sm")
        ct = EnhancedCenteringTheory(nlp)

        ct.update_discourse("John went to the store.")
        state = ct.update_discourse("He bought milk.")
        self.assertEqual(state.transition, TransitionType.CONTINUE)

    def test_cohesion_scoring(self):
        import spacy
        from lgram import EnhancedCenteringTheory

        nlp = spacy.load("en_core_web_sm")
        ct = EnhancedCenteringTheory(nlp)

        result = ct.evaluate_cohesion([
            "John went to the store.",
            "He bought milk.",
            "The store was busy.",
            "Then John left.",
        ])
        self.assertIn("cohesion_score", result)
        self.assertIn("transition_distribution", result)
        self.assertGreaterEqual(result["cohesion_score"], 0.0)
        self.assertLessEqual(result["cohesion_score"], 1.0)

    def test_evaluate_does_not_destroy_state(self):
        import spacy
        from lgram import EnhancedCenteringTheory

        nlp = spacy.load("en_core_web_sm")
        ct = EnhancedCenteringTheory(nlp)

        ct.update_discourse("Alice went to the park.")
        ct.update_discourse("She saw Bob.")

        before = len(ct.discourse_history)

        ct.evaluate_cohesion([
            "Unrelated text about weather.",
            "It was sunny outside.",
        ])

        self.assertEqual(len(ct.discourse_history), before)
        self.assertEqual(ct.discourse_history[0].utterance, "Alice went to the park.")

    def test_save_load(self):
        import spacy
        from lgram import EnhancedCenteringTheory

        nlp = spacy.load("en_core_web_sm")
        ct = EnhancedCenteringTheory(nlp)
        ct.update_discourse("John went to the store.")
        ct.update_discourse("He bought milk.")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp = f.name
        try:
            ct.save(tmp)

            ct2 = EnhancedCenteringTheory(nlp)
            ct2.load(tmp)
            self.assertEqual(len(ct2.discourse_history), 2)
            self.assertEqual(
                ct2.discourse_history[0].utterance,
                ct.discourse_history[0].utterance,
            )
        finally:
            os.unlink(tmp)

    def test_custom_weights(self):
        import spacy
        from lgram import EnhancedCenteringTheory

        nlp = spacy.load("en_core_web_sm")
        ct = EnhancedCenteringTheory(
            nlp,
            salience_weights={"nsubj": 10.0},
            pos_weights={"NOUN": 5.0},
        )
        self.assertEqual(ct.salience_weights["nsubj"], 10.0)
        self.assertEqual(ct.pos_weights["NOUN"], 5.0)

    def test_logging(self):
        from lgram.utils import setup_logging
        setup_logging("WARNING")  # smoke test

    def test_extract_clauses(self):
        import spacy
        from lgram import EnhancedCenteringTheory

        nlp = spacy.load("en_core_web_sm")
        ct = EnhancedCenteringTheory(nlp)

        clauses = ct.extract_clauses(
            "John went to the store because he needed milk, but the store was closed."
        )
        self.assertGreaterEqual(len(clauses), 2)

    def test_intra_sentential_analysis(self):
        import spacy
        from lgram import EnhancedCenteringTheory

        nlp = spacy.load("en_core_web_sm")
        ct = EnhancedCenteringTheory(nlp)

        result = ct.analyze_intra_sentential(
            "John went to the store because he needed milk."
        )
        self.assertIn("cohesion_score", result)
        self.assertIn("transitions", result)
        self.assertGreaterEqual(result["clause_count"], 2)

    def test_analyze_full(self):
        import spacy
        from lgram import EnhancedCenteringTheory

        nlp = spacy.load("en_core_web_sm")
        ct = EnhancedCenteringTheory(nlp)

        result = ct.analyze_full(
            "John went to the store. He bought milk because he was hungry."
        )
        self.assertIn("inter_sentential", result)
        self.assertIn("intra_sentential", result)
        self.assertGreaterEqual(result["sentence_count"], 1)

    def test_get_coherent_next_center(self):
        import spacy
        from lgram import EnhancedCenteringTheory

        nlp = spacy.load("en_core_web_sm")
        ct = EnhancedCenteringTheory(nlp)
        self.assertIsNone(ct.get_coherent_next_center())

        ct.update_discourse("John went to the store.")
        center = ct.get_coherent_next_center()
        self.assertIsNotNone(center)

    def test_get_discourse_summary(self):
        import spacy
        from lgram import EnhancedCenteringTheory

        nlp = spacy.load("en_core_web_sm")
        ct = EnhancedCenteringTheory(nlp)
        summary = ct.get_discourse_summary()
        self.assertIn("message", summary)

        ct.update_discourse("John went to the store.")
        summary = ct.get_discourse_summary()
        self.assertIn("current_cb", summary)
        self.assertIn("discourse_length", summary)


if __name__ == "__main__":
    unittest.main()
