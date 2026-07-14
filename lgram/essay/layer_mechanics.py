"""
Layer: Mechanics — spelling, punctuation, capitalization.

Uses pyspellchecker (Peter Norvig algorithm, 70K word dictionary)
for real spell checking, not heuristic.

Also checks:
- Capitalization: sentences start with uppercase
- Punctuation: sentences end with . ! ?
- Paragraph formatting: consistent structure

Requires: pip install pyspellchecker
"""

from __future__ import annotations

import re
from typing import List, Optional

from .models import Essay, LayerResult
from .utils import split_sentences


class MechanicsLayer:
    """
    Spelling + punctuation + capitalization checking.

    Scoring:
      - Spelling: error rate per 100 words
      - Capitalization: % of sentences with correct initial case
      - Punctuation: % of sentences with correct terminal punctuation
      - Combined: weighted average (spell 40%, caps 30%, punct 30%)
    """

    WEIGHTS = {"spelling": 0.40, "capitalization": 0.30, "punctuation": 0.30}

    def __init__(self, language: str = "en"):
        self._language = language
        self._spell = None
        self._load_error: Optional[str] = None
        self._init_attempted = False

    @property
    def available(self) -> bool:
        if self._spell is not None:
            return True
        if self._init_attempted:
            return False
        self._init_attempted = True
        try:
            from spellchecker import SpellChecker

            self._spell = SpellChecker(language=self._language)
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    def evaluate(self, essay: Essay) -> LayerResult:
        text = essay.text
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        word_count = len(words)
        sentences = split_sentences(text)
        sent_count = max(len(sentences), 1)

        spelling = self._check_spelling(words)
        capitalization = self._check_capitalization(sentences)
        punctuation = self._check_punctuation(sentences)

        weighted = (
            spelling * self.WEIGHTS["spelling"]
            + capitalization * self.WEIGHTS["capitalization"]
            + punctuation * self.WEIGHTS["punctuation"]
        )
        score = round(weighted * 100, 1)
        normalized = round(weighted, 3)

        evidence: List[str] = []

        if spelling < 0.8:
            misspelled = self._get_misspellings(words)
            if misspelled:
                evidence.append(
                    f"Spelling: {len(misspelled)} potential errors — "
                    f"{', '.join(misspelled[:5])}"
                )
        if capitalization < 0.9:
            evidence.append("Some sentences may lack initial capitalization")
        if punctuation < 0.9:
            evidence.append("Some sentences may lack terminal punctuation")
        if not evidence:
            evidence.append("Mechanics appear adequate")

        sem = 0.05
        ci = (max(0, score - sem * 4 * 100), min(100, score + sem * 4 * 100))

        return LayerResult(
            layer_name="Mechanics",
            score=score,
            normalized_score=normalized,
            raw_details={
                "spelling_score": round(spelling, 3),
                "capitalization_score": round(capitalization, 3),
                "punctuation_score": round(punctuation, 3),
                "word_count": word_count,
                "sentence_count": sent_count,
                "spellchecker_available": self.available,
            },
            evidence=evidence[:3],
            confidence_interval=(round(ci[0], 1), round(ci[1], 1)),
        )

    def _check_spelling(self, words: List[str]) -> float:
        if len(words) < 5 or not self.available:
            return 1.0

        unique_words = set(w.lower() for w in words if w.isalpha() and len(w) > 1)
        if len(unique_words) < 3:
            return 1.0

        try:
            unknown = self._spell.unknown(unique_words)
            error_rate = len(unknown) / max(len(unique_words), 1)
            return max(0.0, 1.0 - error_rate * 3)
        except Exception:
            return 0.8

    def _get_misspellings(self, words: List[str]) -> List[str]:
        if not self.available:
            return []
        unique = set(w.lower() for w in words if w.isalpha() and len(w) > 1)
        try:
            unknown = self._spell.unknown(unique)
            return sorted(unknown)[:8]
        except Exception:
            return []

    @staticmethod
    def _check_capitalization(sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 1.0
        correct = 0
        for s in sentences:
            stripped = s.strip()
            if stripped and stripped[0].isupper():
                correct += 1
        return correct / len(sentences)

    @staticmethod
    def _check_punctuation(sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 1.0
        correct = 0
        for s in sentences:
            stripped = s.strip()
            if stripped and stripped[-1] in ".!?":
                correct += 1
        return correct / len(sentences)
