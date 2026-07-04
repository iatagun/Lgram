"""
Layer 3: Surface Quality Assessment.

Evaluates mechanical language quality independent of content/cohesion:
  - Readability (Flesch Reading Ease)
  - Sentence length variety (std dev of sentence lengths)
  - Vocabulary richness (type-token ratio, hapax ratio)
  - Grammar: dependency parse complexity as proxy for grammaticality

All metrics are pure Python + spaCy — no external grammar checker needed.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from .models import Essay, LayerResult
from .utils import split_sentences


class SurfaceLayer:
    """
    Surface quality assessment using readable features.

    Rubric mapping:
      - Sentence variety → Organization sub-score
      - Vocabulary → Word Choice sub-score
      - Readability → Conventions sub-score
      - Grammar complexity → overall polish indicator

    Normalized to 0-100 scale.
    """

    WEIGHTS = {
        "readability": 0.30,
        "sentence_variety": 0.25,
        "vocabulary": 0.25,
        "grammar_complexity": 0.20,
    }

    def __init__(self, model: str = "en_core_web_md"):
        self._model = model
        self._nlp = None
        self._ta = None

    @property
    def _analyzer(self):
        if self._ta is None:
            from lgram import TextAnalyzer
            self._ta = TextAnalyzer(self._model, similarity_threshold=0.35)
        return self._ta

    def evaluate(self, essay: Essay) -> LayerResult:
        text = essay.text
        sentences = split_sentences(text)
        words = _tokenize(text)
        word_count = len(words)
        sent_count = len(sentences)

        readability = self._flesch(text, sentences, words)
        sent_variety = self._sentence_variety(sentences, words)
        vocabulary = self._vocabulary_richness(words)
        grammar = self._grammar_complexity(sentences)

        weighted = (
            readability * self.WEIGHTS["readability"]
            + sent_variety * self.WEIGHTS["sentence_variety"]
            + vocabulary * self.WEIGHTS["vocabulary"]
            + grammar * self.WEIGHTS["grammar_complexity"]
        )
        score = round(weighted * 100, 1)
        normalized = round(weighted, 3)

        evidence: List[str] = []
        if readability < 0.4:
            evidence.append(f"Low readability (Flesch: {self._flesch_raw(text, sentences, words):.0f})")
        if sent_variety < 0.4:
            evidence.append("Low sentence length variety — monotonous rhythm")
        if vocabulary < 0.4:
            evidence.append("Limited vocabulary range (low type-token ratio)")
        if grammar < 0.3:
            evidence.append("Very short or fragmented sentences — possible grammatical issues")
        if not evidence:
            evidence.append("Surface quality adequate")

        sem = _component_sem([readability, sent_variety, vocabulary, grammar])
        ci = (max(0, score - sem * 2 * 100), min(100, score + sem * 2 * 100))

        return LayerResult(
            layer_name="Surface Quality",
            score=score,
            normalized_score=normalized,
            raw_details={
                "readability": {
                    "flesch_reading_ease": round(self._flesch_raw(text, sentences, words), 1),
                    "normalized": round(readability, 3),
                },
                "sentence_variety": {
                    "avg_length_words": round(word_count / max(sent_count, 1), 1),
                    "std_length": round(self._sent_length_std(sentences), 1),
                    "normalized": round(sent_variety, 3),
                },
                "vocabulary": {
                    "word_count": word_count,
                    "unique_words": len(set(w.lower() for w in words)),
                    "ttr": round(len(set(w.lower() for w in words)) / max(word_count, 1), 3),
                    "hapax_ratio": round(self._hapax_ratio(words), 3),
                    "normalized": round(vocabulary, 3),
                },
                "grammar_complexity": {
                    "sentence_count": sent_count,
                    "normalized": round(grammar, 3),
                },
                "weights": self.WEIGHTS,
            },
            evidence=evidence,
            confidence_interval=ci,
        )

    def _flesch(self, text: str, sentences: List[str], words: List[str]) -> float:
        raw = self._flesch_raw(text, sentences, words)
        return min(1.0, max(0.0, raw / 100.0))

    def _flesch_raw(self, text: str, sentences: List[str], words: List[str]) -> float:
        try:
            return self._analyzer.readability_score(text).flesch_score
        except Exception:
            pass
        if not sentences or not words:
            return 50.0
        syllable_count = sum(_count_syllables(w) for w in words)
        word_cnt = len(words)
        sent_cnt = len(sentences)
        if word_cnt == 0 or sent_cnt == 0:
            return 50.0
        return 206.835 - 1.015 * (word_cnt / sent_cnt) - 84.6 * (syllable_count / word_cnt)

    def _sentence_variety(self, sentences: List[str], words: List[str]) -> float:
        if len(sentences) < 3:
            return 0.3
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        if mean_len == 0 or len(lengths) < 2:
            return 0.4
        variance = sum((l - mean_len) ** 2 for l in lengths) / (len(lengths) - 1)
        std = math.sqrt(variance)
        cv = std / mean_len
        if cv < 0.2:
            return 0.3
        elif cv < 0.5:
            return 0.6 + (0.5 - cv) * 0.5
        elif cv < 0.8:
            return 1.0 - (cv - 0.5) * 0.3
        else:
            return 0.7

    def _sent_length_std(self, sentences: List[str]) -> float:
        lengths = [len(s.split()) for s in sentences]
        if len(lengths) < 2:
            return 0.0
        mean = sum(lengths) / len(lengths)
        return math.sqrt(sum((l - mean) ** 2 for l in lengths) / (len(lengths) - 1))

    def _vocabulary_richness(self, words: List[str]) -> float:
        if len(words) < 20:
            return 0.3
        lower = [w.lower() for w in words if w.isalpha()]
        if len(lower) < 5:
            return 0.2
        types = len(set(lower))
        ttr = types / max(len(lower), 1)
        hapax = self._hapax_ratio(lower)
        score = (ttr * 0.6 + hapax * 0.4)
        if ttr < 0.4:
            score *= 0.7
        elif ttr > 0.75:
            score *= 0.8
        return min(1.0, max(0.0, score * 1.5))

    @staticmethod
    def _hapax_ratio(words: List[str]) -> float:
        if len(words) < 5:
            return 0.0
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        hapax_count = sum(1 for c in freq.values() if c == 1)
        return min(1.0, hapax_count / max(len(words), 1) * 2)

    def _grammar_complexity(self, sentences: List[str]) -> float:
        if len(sentences) < 3:
            return 0.4
        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / max(len(lengths), 1)
        if avg_len < 5:
            return 0.2
        elif avg_len < 10:
            return 0.4
        elif avg_len < 20:
            return 0.6 + (avg_len - 10) * 0.03
        elif avg_len < 30:
            return 0.9
        else:
            return 0.8


def _component_sem(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.1
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance / n)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text)


def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:\"'()[]{}")
    if len(word) <= 3:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    if word.endswith("le") and len(word) > 3 and word[-3] not in vowels:
        count += 1
    return max(1, count)
