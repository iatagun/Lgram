"""
Layer 1: Content Analysis (pluggable analyzer interface).

Analyzes essay content quality using rubric-specific criteria:
- Thesis clarity and argument structure
- Evidence use and reasoning
- Topic development

Uses a pluggable ContentAnalyzer interface. Includes:
- MockContentAnalyzer (heuristic-based, no API needed — testing/demo)
- Designed for real LLM-based analyzers (OpenAI, Claude, etc.)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .models import ContentAnalyzer, Essay, LayerResult, RubricCriterion
from .utils import split_sentences as _get_sentences

_CACHE: Dict[str, LayerResult] = {}
_CACHE_MAX_SIZE = 256


class MockContentAnalyzer(ContentAnalyzer):
    """
    Heuristic content analyzer using surface features.

    WARNING: This is a stand-in for demonstration. A real deployment
    MUST use a rubric-calibrated analyzer. Surface heuristics correlate
    weakly (r ~ 0.3-0.5) with human content evaluations.

    DEPRECATED alias: MockContentJudge = MockContentAnalyzer
    """

    def __init__(self):
        self._transition_phrases = {
            "however", "therefore", "moreover", "furthermore", "consequently",
            "thus", "nevertheless", "nonetheless", "accordingly", "hence",
            "in contrast", "on the other hand", "in addition", "for example",
            "for instance", "in conclusion", "to summarize", "as a result",
            "because", "although", "while", "whereas", "unless", "since",
        }

    def analyze(self, essay: Essay, rubric: List[RubricCriterion]) -> LayerResult:
        key = essay.text.strip()
        if key in _CACHE:
            return _CACHE[key]

        sentences = _get_sentences(essay.text)
        words = essay.text.split()

        thesis_score = self._assess_thesis(sentences)
        evidence_score = self._assess_evidence(words, sentences)
        structure_score = self._assess_structure(sentences, words)
        development_score = self._assess_development(sentences)

        raw = (thesis_score + evidence_score + structure_score + development_score) / 4
        score = min(100.0, max(0.0, raw * 100))

        evidence: List[str] = []
        if thesis_score < 0.5:
            evidence.append("Weak or unclear thesis statement")
        if evidence_score < 0.5:
            evidence.append("Limited use of supporting evidence or examples")
        if structure_score < 0.5:
            evidence.append("Weak argument structure or logical flow")
        if development_score < 0.5:
            evidence.append("Insufficient topic development or shallow reasoning")

        if not evidence:
            evidence.append("Content and argument structure adequate")

        sem = _standard_error([thesis_score, evidence_score, structure_score, development_score])
        ci = (max(0, score - sem * 2 * 100), min(100, score + sem * 2 * 100))

        result = LayerResult(
            layer_name="Content & Argument (heuristic)",
            score=score,
            normalized_score=round(score / 100.0, 3),
            raw_details={
                "thesis_score": round(thesis_score, 3),
                "evidence_score": round(evidence_score, 3),
                "structure_score": round(structure_score, 3),
                "development_score": round(development_score, 3),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "transition_count": self._count_transitions(essay.text),
            },
            evidence=evidence,
            confidence_interval=ci,
        )

        _CACHE[key] = result
        if len(_CACHE) > _CACHE_MAX_SIZE:
            _CACHE.pop(next(iter(_CACHE)))
        return result

    def _assess_thesis(self, sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 0.2
        first_three = " ".join(sentences[:3]).lower()
        thesis_signals = {
            "argue that", "this essay", "in this paper", "the main",
            "will discuss", "will examine", "will explore", "will analyze",
            "the purpose", "the goal", "i believe", "i argue", "i think",
            "my position", "the issue", "the problem", "the question",
            "important because", "significant because",
            "has changed", "have changed", "is a big problem", "is an issue",
            "there are", "there is", "can be", "should be",
        }
        hits = sum(1 for s in thesis_signals if s in first_three)
        score = min(1.0, 0.3 + hits * 0.15)
        if len(sentences[0].split()) < 3:
            score *= 0.5
        return score

    def _assess_evidence(self, words: List[str], sentences: List[str]) -> float:
        evidence_signals = {
            "for example", "for instance", "study", "research", "data",
            "evidence", "according to", "report", "survey", "statistic",
            "percent", "figure", "finding", "result", "experiment",
            "demonstrate", "show that", "indicate", "suggest that",
            "prove", "confirm", "support", "cited", "reference",
            "such as", "like", "these", "this", "because", "since",
        }
        text_lower = " ".join(sentences).lower()
        hits = sum(1 for s in evidence_signals if s in text_lower)
        score = min(1.0, 0.2 + hits * 0.10)
        if len(words) < 50:
            score *= 0.5
        return score

    def _assess_structure(self, sentences: List[str], words: List[str]) -> float:
        if len(sentences) < 3:
            return 0.3
        transition_count = self._count_transitions(" ".join(sentences))
        ideal_transitions = max(1, int(len(sentences) * 0.3))
        transition_score = min(1.0, transition_count / ideal_transitions)
        avg_words = len(words) / max(len(sentences), 1)
        length_variety = min(1.0, avg_words / 20) if 5 <= avg_words <= 35 else 0.5
        return (transition_score * 0.6 + length_variety * 0.4)

    def _assess_development(self, sentences: List[str]) -> float:
        if len(sentences) < 3:
            return 0.3
        text_lower = " ".join(sentences).lower()
        conclusion_signals = {
            "in conclusion", "to conclude", "in summary", "to summarize",
            "overall", "ultimately", "in the end", "finally", "therefore",
            "thus", "as a result", "consequently",
        }
        last_two = " ".join(sentences[-2:]).lower()
        has_conclusion = any(s in last_two for s in conclusion_signals)
        middle_development = len(sentences) > 4
        deep_signals = {"because", "since", "as a result", "this means", "this implies",
                        "the reason", "caused by", "due to", "leads to", "resulting in"}
        depth_hits = sum(1 for s in deep_signals if s in text_lower)
        depth_score = min(1.0, depth_hits * 0.12)
        score = 0.3
        if has_conclusion:
            score += 0.25
        if middle_development:
            score += 0.2
        score += depth_score * 0.25
        return min(1.0, score)

    def _count_transitions(self, text: str) -> int:
        text_lower = text.lower()
        return sum(1 for t in self._transition_phrases if t in text_lower)


MockContentJudge = MockContentAnalyzer
MockContentAnalyzer.evaluate = MockContentAnalyzer.analyze


def _standard_error(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.25
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance / n)
