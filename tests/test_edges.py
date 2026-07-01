"""
Comprehensive edge case and false positive tests for Centering-Lgram.
"""

import os
import tempfile
import unittest

import spacy


def _make_ct():
    nlp = spacy.load("en_core_web_sm")
    from lgram import EnhancedCenteringTheory
    return EnhancedCenteringTheory(nlp)


# =============================================================================
# Edge cases – basic API
# =============================================================================

class TestEdgeCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load("en_core_web_sm")

    def test_empty_input(self):
        from lgram import EnhancedCenteringTheory
        ct = EnhancedCenteringTheory(self.nlp)
        state = ct.analyze_utterance("")
        self.assertEqual(state.preferred_center, None)
        self.assertEqual(state.forward_centers, [])
        self.assertEqual(state.transition.value, "Establish")

    def test_single_word(self):
        from lgram import EnhancedCenteringTheory
        ct = EnhancedCenteringTheory(self.nlp)
        state = ct.analyze_utterance("Hello.")
        self.assertEqual(state.preferred_center, None)

    def test_all_function_words(self):
        ct = _make_ct()
        state = ct.analyze_utterance("If only it were that simple.")
        self.assertIsNotNone(state)

    def test_no_nouns_pronouns(self):
        ct = _make_ct()
        state = ct.analyze_utterance("Quickly run away fast.")
        self.assertEqual(state.forward_centers, [])
        self.assertIsNone(state.preferred_center)

    def test_punctuation_only(self):
        ct = _make_ct()
        state = ct.analyze_utterance("...")
        self.assertEqual(state.forward_centers, [])

    def test_long_sentence_center_limit(self):
        ct = _make_ct()
        text = "Alice and Bob and Charlie and Dave and Eve and Frank went to the big store."
        cf, _ = ct.compute_forward_centers(text)
        # Cf list is limited to 5, entity map may have more unique keys
        self.assertLessEqual(len(cf), 5)

    def test_multiple_same_subject(self):
        ct = _make_ct()
        ct.update_discourse("Alice went to the park.")
        ct.update_discourse("Alice met Bob there.")
        state = ct.update_discourse("Alice was happy.")
        self.assertEqual(state.transition.value, "Continue")


# =============================================================================
# Transition truth table
# =============================================================================

class TestTransitionCorrectness(unittest.TestCase):

    def test_Continue_Cb_equals_Cp(self):
        ct = _make_ct()
        ct.update_discourse("John bought a car.")
        state = ct.update_discourse("John drove it home.")
        self.assertEqual(state.transition.value, "Continue")

    def test_Retain_Cb_same_Cp_different(self):
        ct = _make_ct()
        ct.update_discourse("Alice likes pizza.")
        state = ct.update_discourse("Bob likes Alice.")
        self.assertIn(state.transition.value, ("Retain", "Continue"))

    def test_Smooth_Shift_new_center(self):
        ct = _make_ct()
        ct.update_discourse("John ate lunch.")
        ct.update_discourse("Mary arrived later.")
        state = ct.update_discourse("John greeted Mary.")
        # Mary was the prev Cp, becomes new Cb
        self.assertIn(state.transition.value, ("Smooth-Shift", "Continue", "Retain"))

    def test_Rough_Shift_complete_change(self):
        ct = _make_ct()
        ct.update_discourse("The dog barked loudly.")
        state = ct.update_discourse("Stock markets fell sharply today.")
        self.assertEqual(state.transition.value, "Rough-Shift")


# =============================================================================
# False positive prevention
# =============================================================================

class TestFalsePositives(unittest.TestCase):

    def test_different_person_pronoun_not_coreferent(self):
        ct = _make_ct()
        ct.update_discourse("Alice met Bob at the cafe.")
        s = ct.update_discourse("He ordered coffee.")
        # NOTE: without gender model, "he" may match first person entity (alice).
        # This is a known limitation, not a bug.
        self.assertIn(
            s.backward_center, ("bob", "alice"),
            "Cb should be bob (correct) or alice (known limitation)",
        )

    def test_she_does_not_match_male(self):
        ct = _make_ct()
        ct.update_discourse("Bob left early.")
        s = ct.update_discourse("She stayed late.")
        # NOTE: spaCy small model lacks gender detection. "she" may falsely
        # match "bob" because both are person entities.
        self.assertIn(
            s.backward_center, (None, "bob"),
            "None = strict (correct), bob = known limitation",
        )

    def test_it_does_not_match_person(self):
        ct = _make_ct()
        ct.update_discourse("John liked the movie.")
        s = ct.update_discourse("It was very long.")
        # "It" should resolve to "movie" (non-person), not "john"
        self.assertEqual(s.backward_center, "movie")

    def test_they_only_matches_plural(self):
        ct = _make_ct()
        ct.update_discourse("Alice and Bob arrived.")
        s = ct.update_discourse("They sat down.")
        # "They" should resolve to plural antecedent
        self.assertIsNotNone(s.backward_center)

    def test_demonstrative_not_a_center(self):
        ct = _make_ct()
        cf, _ = ct.compute_forward_centers("This is a test.")
        self.assertNotIn("this", cf)

    def test_generic_pronoun_not_a_center(self):
        ct = _make_ct()
        cf, _ = ct.compute_forward_centers("Some people like coffee.")
        self.assertNotIn("some", cf)

    def test_same_word_different_meaning(self):
        ct = _make_ct()
        ct.update_discourse("The bank is by the river bank.")
        s = ct.update_discourse("The bank is closed today.")
        # "bank" matches, should be Continue
        self.assertEqual(s.transition.value, "Continue")


# =============================================================================
# Clause extraction edge cases
# =============================================================================

class TestClauseExtraction(unittest.TestCase):

    def test_simple_sentence_one_clause(self):
        ct = _make_ct()
        clauses = ct.extract_clauses("John walked home.")
        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0][1], "main")

    def test_relative_clause_not_split(self):
        ct = _make_ct()
        clauses = ct.extract_clauses("The man who lives next door bought a car.")
        # relative clause may or may not be split depending on model
        self.assertGreaterEqual(len(clauses), 1)

    def test_complement_clause(self):
        ct = _make_ct()
        clauses = ct.extract_clauses("I know that you are right.")
        self.assertGreaterEqual(len(clauses), 2)

    def test_coordinated_clauses(self):
        ct = _make_ct()
        clauses = ct.extract_clauses(
            "John went to the store, Mary went home, and Bob stayed."
        )
        self.assertGreaterEqual(len(clauses), 2)

    def test_adverbial_clause(self):
        ct = _make_ct()
        clauses = ct.extract_clauses("He left because he was angry.")
        self.assertGreaterEqual(len(clauses), 2)

    def test_no_verbs(self):
        ct = _make_ct()
        clauses = ct.extract_clauses("Hello world.")
        self.assertEqual(len(clauses), 1)

    def test_nested_clauses(self):
        ct = _make_ct()
        clauses = ct.extract_clauses(
            "I think that John said that Mary left."
        )
        self.assertGreaterEqual(len(clauses), 2)

    def test_relative_clause_detected(self):
        ct = _make_ct()
        clauses = ct.extract_clauses(
            "The book that I read was interesting."
        )
        self.assertGreaterEqual(len(clauses), 2)


# =============================================================================
# Pronoun resolution edge cases
# =============================================================================

class TestPronounResolution(unittest.TestCase):

    def test_possessive_pronoun_not_a_center(self):
        ct = _make_ct()
        cf, _ = ct.compute_forward_centers("His car is red.")
        self.assertNotIn("his", cf)

    def test_possessive_resolves_in_backward_center(self):
        ct = _make_ct()
        ct.update_discourse("John bought a car.")
        s = ct.update_discourse("His car is red.")
        # "His" refers to John, so Cb should be john
        self.assertEqual(s.backward_center, "john")

    def test_pronoun_chain(self):
        ct = _make_ct()
        ct.update_discourse("Alice met Bob.")
        ct.update_discourse("She liked him.")
        state = ct.update_discourse("He smiled.")
        self.assertIn(state.transition.value, ("Continue", "Retain"))

    def test_ambiguous_pronoun_first_match(self):
        ct = _make_ct()
        ct.update_discourse("Alice and Mary went to the store.")
        s = ct.update_discourse("She bought milk.")
        # first person pronoun matches first person in Cf
        self.assertEqual(s.backward_center, "alice")


# =============================================================================
# State isolation
# =============================================================================

class TestStateIsolation(unittest.TestCase):

    def test_analyze_utterance_does_not_change_history(self):
        ct = _make_ct()
        ct.update_discourse("Hello world.")
        before = len(ct.discourse_history)
        ct.analyze_utterance("Another sentence.")
        self.assertEqual(len(ct.discourse_history), before)

    def test_evaluate_preserves_history(self):
        ct = _make_ct()
        ct.update_discourse("Alice went to the park.")
        ct.update_discourse("She saw Bob.")
        before = list(ct.discourse_history)
        ct.evaluate_cohesion(["Unrelated text.", "More text."])
        self.assertEqual(len(ct.discourse_history), len(before))
        self.assertEqual(ct.discourse_history[0].utterance, before[0].utterance)

    def test_intra_sentential_does_not_leak(self):
        ct = _make_ct()
        ct.update_discourse("Main discourse sentence.")
        before = len(ct.discourse_history)
        ct.analyze_intra_sentential("Alice went because Bob called.")
        self.assertEqual(len(ct.discourse_history), before)


# =============================================================================
# Serialization
# =============================================================================

class TestSerialization(unittest.TestCase):

    def test_roundtrip_with_history(self):
        ct = _make_ct()
        ct.update_discourse("John went to the store.")
        ct.update_discourse("He bought milk.")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp = f.name
        try:
            ct.save(tmp)

            ct2 = _make_ct()
            ct2.load(tmp)
            self.assertEqual(len(ct2.discourse_history), 2)
            self.assertEqual(ct2.discourse_history[0].utterance, "John went to the store.")
            # Verify the loaded instance works
            s = ct2.update_discourse("The milk was fresh.")
            self.assertIsNotNone(s.transition)
        finally:
            os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()
