"""
Turkish Centering — logic tests on hand-built UD parses (no neural model needed).

The model-backed smoke test is opt-in: LGRAM_TR_MODEL_TESTS=1 (downloads ~440 MB).
"""

from __future__ import annotations

import os
import unittest

import pytest

pytest.importorskip("snowballstemmer", reason="optional: pip install .[tr]")

from lgram.models.centering_theory import TransitionType as TT  # noqa: E402
from lgram.tr.centering import (  # noqa: E402
    _STEM,
    _base,
    _lower,
    analyze_parsed,
    entity_key,
)
from lgram.tr.parser import Token, parse_feats  # noqa: E402

FIN3 = {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}


def tok(i, form, upos, head, deprel, **feats):
    return Token(i, form, upos, feats, head, deprel)


def verb(i, form, **extra):
    return tok(i, form, "VERB", 0, "root", **{**FIN3, **extra})


def run(*sentences):
    return analyze_parsed([f"s{i}" for i in range(len(sentences))], sentences)


def stem(word):
    return _STEM.stemWord(_lower(word))


# "Ayşe pazara gitti."
AYSE_GITTI = [
    tok(1, "Ayşe", "PROPN", 3, "nsubj"),
    tok(2, "pazara", "NOUN", 3, "obl", Case="Dat"),
    verb(3, "gitti"),
]


class TestHelpers(unittest.TestCase):
    def test_turkish_lowercase(self):
        self.assertEqual(_lower("İstanbul"), "istanbul")
        self.assertEqual(_lower("ISPARTA"), "ısparta")
        self.assertEqual(_base("İstanbul’a"), "istanbul")

    def test_parse_feats(self):
        self.assertEqual(parse_feats("_"), {})
        self.assertEqual(
            parse_feats("Case=Nom|Person[psor]=3"),
            {"Case": "Nom", "Person[psor]": "3"},
        )


class TestZeroPronoun(unittest.TestCase):
    def test_dropped_subject_continues_previous_topic(self):
        # "Ayşe pazara gitti. Elma aldı."  (subject of 2nd sentence = Ayşe)
        report = run(AYSE_GITTI, [tok(1, "Elma", "NOUN", 2, "obj"), verb(2, "aldı")])
        u2 = report.utterances[1]
        self.assertTrue(u2.zero_subject)
        self.assertEqual(u2.cb, stem("Ayşe"))
        self.assertEqual(u2.cp, stem("Ayşe"))
        self.assertEqual(u2.transition, TT.CONTINUE)

    def test_unlinkable_zero_subject_is_detected_but_unresolved(self):
        u = run([verb(1, "geldi")]).utterances[0]  # nothing before it to link to
        self.assertTrue(u.zero_detected)
        self.assertFalse(u.zero_subject)
        self.assertEqual(u.unresolved, ["zero"])

    def test_overt_subject_blocks_zero_pronoun(self):
        report = run(AYSE_GITTI, [tok(1, "Ali", "PROPN", 2, "nsubj"), verb(2, "geldi")])
        self.assertFalse(report.utterances[1].zero_subject)

    def test_imperative_is_not_a_zero_pronoun(self):
        report = run(AYSE_GITTI, [verb(1, "Gel", Mood="Imp")])
        self.assertFalse(report.utterances[1].zero_subject)

    def test_first_person_zero_subject_is_speaker_not_previous_entity(self):
        first = [
            tok(1, "Eve", "NOUN", 2, "obl", Case="Dat"),
            verb(2, "gittim", Person="1"),
        ]
        second = [verb(1, "Yoruldum", Person="1")]
        report = run(first, second)
        self.assertEqual(report.utterances[0].cp, "@1sg")
        self.assertEqual(report.utterances[1].transition, TT.CONTINUE)

    def test_copular_predicate_takes_person_from_copula(self):
        # "Ayşe geldi. Hastaydı."  root = ADJ, agreement lives on the copula
        report = run(
            [tok(1, "Ayşe", "PROPN", 2, "nsubj"), verb(2, "geldi")],
            [
                tok(1, "hasta", "ADJ", 0, "root"),
                tok(2, "ydı", "AUX", 1, "cop", **FIN3),
            ],
        )
        self.assertTrue(report.utterances[1].zero_subject)


class TestAnaphora(unittest.TestCase):
    def test_pronoun_resolves_to_previous_cf(self):
        # "Ali geldi. Ayşe onu gördü."
        report = run(
            [tok(1, "Ali", "PROPN", 2, "nsubj"), verb(2, "geldi")],
            [
                tok(1, "Ayşe", "PROPN", 3, "nsubj"),
                tok(2, "onu", "PRON", 3, "obj", PronType="Prs"),
                verb(3, "gördü"),
            ],
        )
        u2 = report.utterances[1]
        self.assertEqual(u2.cb, stem("Ali"))
        self.assertEqual(u2.cp, stem("Ayşe"))
        self.assertEqual(u2.transition, TT.RETAIN)

    def test_disjoint_reference_pronoun_is_not_the_named_subject(self):
        # "Ali geldi. Ali onu gördü."  -> onu != Ali
        report = run(
            [tok(1, "Ali", "PROPN", 2, "nsubj"), verb(2, "geldi")],
            [
                tok(1, "Ali", "PROPN", 3, "nsubj"),
                tok(2, "onu", "PRON", 3, "obj", PronType="Prs"),
                verb(3, "gördü"),
            ],
        )
        self.assertEqual([m.key for m in report.utterances[1].cf], [stem("Ali")])

    def test_plural_pronoun_needs_plural_antecedent(self):
        report = run(
            [tok(1, "Ali", "PROPN", 2, "nsubj"), verb(2, "geldi")],
            [
                tok(1, "Onlar", "PRON", 2, "nsubj", PronType="Prs"),
                verb(2, "güldü", Number="Plur"),
            ],
        )
        self.assertEqual(report.utterances[1].transition, TT.NOCB)

    def test_possessive_suffix_realises_possessor(self):
        # "Ayşe geldi. Annesi hastaydı."  (Annesi = Ayşe's mother)
        report = run(
            [tok(1, "Ayşe", "PROPN", 2, "nsubj"), verb(2, "geldi")],
            [
                tok(1, "Annesi", "NOUN", 2, "nsubj", **{"Person[psor]": "3"}),
                tok(2, "hasta", "ADJ", 0, "root"),
                tok(3, "ydı", "AUX", 2, "cop", **FIN3),
            ],
        )
        u2 = report.utterances[1]
        self.assertEqual(u2.cb, stem("Ayşe"))
        self.assertEqual(u2.cp, stem("Annesi"))
        self.assertEqual(u2.transition, TT.RETAIN)


class TestResolution(unittest.TestCase):
    def test_possessor_binds_to_its_clause_subject(self):
        # "Ayşe pazara gitti. Ali annesini aradı."  (annesi = Ali's, not Ayşe's)
        report = run(
            AYSE_GITTI,
            [
                tok(1, "Ali", "PROPN", 3, "nsubj"),
                tok(2, "annesini", "NOUN", 3, "obj", **{"Person[psor]": "3"}),
                verb(3, "aradı"),
            ],
        )
        u2 = report.utterances[1]
        self.assertIsNone(u2.cb)  # no link to Ayşe
        self.assertEqual(u2.transition, TT.NOCB)

    def test_possessed_subject_takes_possessor_from_context(self):
        # "Ayşe pazara gitti. Annesi kızdı."  (annesi = Ayşe's: it IS the subject)
        report = run(
            AYSE_GITTI,
            [
                tok(1, "Annesi", "NOUN", 2, "nsubj", **{"Person[psor]": "3"}),
                verb(2, "kızdı"),
            ],
        )
        self.assertEqual(report.utterances[1].cb, stem("Ayşe"))

    def test_zero_prefers_previous_cb_over_previous_cp(self):
        # "Ali geldi. Ayşe Ali'yi gördü. Gülümsedi."  Cb of U2 is Ali, Cp is Ayşe;
        # the zero subject of U3 follows the Cb.
        report = run(
            [tok(1, "Ali", "PROPN", 2, "nsubj"), verb(2, "geldi")],
            [
                tok(1, "Ayşe", "PROPN", 3, "nsubj"),
                tok(2, "Ali'yi", "PROPN", 3, "obj"),
                verb(3, "gördü"),
            ],
            [verb(1, "gülümsedi")],
        )
        self.assertEqual(report.utterances[1].cb, "ali")
        self.assertEqual(report.utterances[1].cp, "ayşe")
        self.assertEqual(report.utterances[2].cb, "ali")


class TestEntityKey(unittest.TestCase):
    def test_proper_names_are_not_stemmed(self):
        # snowball turns "Çadır" into "ça"; names keep their base form
        self.assertEqual(entity_key(tok(1, "Çadır'ın", "PROPN", 0, "root")), "çadır")
        noun = tok(1, "kapısı", "NOUN", 0, "root")
        self.assertEqual(entity_key(noun), _STEM.stemWord("kapısı"))


class TestTransitions(unittest.TestCase):
    def test_no_shared_entity_is_nocb(self):
        report = run(
            [tok(1, "Ali", "PROPN", 2, "nsubj"), verb(2, "geldi")],
            [tok(1, "Ayşe", "PROPN", 2, "nsubj"), verb(2, "gitti")],
        )
        self.assertEqual(report.utterances[1].transition, TT.NOCB)
        # rough_shift_ratio keeps counting NOCB, as before the split
        self.assertEqual(report.rough_shift_ratio, 1.0)

    def test_first_utterance_is_establish_and_not_counted(self):
        report = run(AYSE_GITTI)
        self.assertEqual(report.utterances[0].transition, TT.ESTABLISH)
        self.assertEqual(report.transition_distribution, {})

    def test_empty_parses_are_skipped(self):
        self.assertEqual(run([], AYSE_GITTI).utterances[0].transition, TT.ESTABLISH)


@unittest.skipUnless(
    os.environ.get("LGRAM_TR_MODEL_TESTS"),
    "set LGRAM_TR_MODEL_TESTS=1 (downloads model)",
)
class TestWithModel(unittest.TestCase):
    def test_zero_subject_detected_end_to_end(self):
        from lgram.tr import TurkishCenteringAnalyzer

        report = TurkishCenteringAnalyzer().analyze("Ayşe pazara gitti. Elma aldı.")
        self.assertEqual(len(report.utterances), 2)
        self.assertTrue(report.utterances[1].zero_subject)
        self.assertEqual(report.utterances[1].transition, TT.CONTINUE)


if __name__ == "__main__":
    unittest.main()
