"""
PreFilter: Grammar/Cohesion Disambiguation Layer.

Inserted BEFORE centering analysis to prevent spaCy from producing
spurious cohesion errors on ungrammatical EFL input.

Architecture:
  ESSAY TEXT
      │
      ├─► LanguageTool (optional) — grammar/spelling check
      │     └─ Categorizes errors: GRAMMAR vs POSSIBLE_COHESION
      ├─► Heuristic fallback — article count, pronoun ratio, sentence structure
      └─► PreFilterReport
            ├─ grammar_issues: errors that are clearly grammar, not cohesion
            ├─ cohesion_risks: grammar issues that MAY affect Cb/Cf
            ├─ parse_confidence: estimated spaCy parse reliability
            └─ recommendations: what to tell the teacher

Key insight: If a student writes "he" instead of "she" (L1 transfer),
this is a GRAMMAR error that happens to use a pronoun — NOT a cohesion
break. The prefilter tags it so the cohesion layer can adjust confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .utils import split_sentences, ARTICLE_RATIO_EXPECTED


@dataclass
class PreFilterReport:
    text_length_words: int
    grammar_issues: List[Dict[str, Any]] = field(default_factory=list)
    cohesion_risks: List[Dict[str, Any]] = field(default_factory=list)
    parse_confidence: float = 1.0
    recommendations: List[str] = field(default_factory=list)
    language_tool_available: bool = False

    @property
    def has_critical_grammar_issues(self) -> bool:
        return len(self.grammar_issues) >= 5 or self.parse_confidence < 0.5

    @property
    def cohesion_override_count(self) -> int:
        return len(self.cohesion_risks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_length_words": self.text_length_words,
            "grammar_issues_count": len(self.grammar_issues),
            "cohesion_risks_count": len(self.cohesion_risks),
            "parse_confidence": self.parse_confidence,
            "recommendations": self.recommendations,
            "language_tool_available": self.language_tool_available,
        }


class PreFilter:
    """
    Disambiguates grammar errors from cohesion issues in EFL text.

    Without LanguageTool: uses heuristic patterns for common EFL errors.
    With LanguageTool: full grammar check + categorization.

    Usage:
        pf = PreFilter()
        report = pf.analyze(text)

        if report.has_critical_grammar_issues:
            # Warn: cohesion analysis may be unreliable
            # Report grammar issues separately from cohesion
    """

    PRONOUN_ERROR_PATTERNS = [
        (re.compile(r"\b(she)\s+(is|was|has|will|would|can|could|should)\b", re.I),
         "Possible gender mismatch — if antecedent is male, 'she' should be 'he'"),
        (re.compile(r"\b(he)\s+(is|was|has|will|would|can|could|should)\b", re.I),
         "Possible gender mismatch — if antecedent is female, 'he' should be 'she'"),
    ]

    def __init__(self, use_language_tool: bool = True):
        self._lt = None
        self._lt_attempted = False
        self._use_lt = use_language_tool

    @property
    def language_tool_available(self) -> bool:
        if not self._use_lt:
            return False
        if self._lt_attempted:
            return self._lt is not None
        self._lt_attempted = True
        try:
            import language_tool_python
            self._lt = language_tool_python.LanguageTool("en-US")
            return True
        except (ImportError, Exception):
            return False

    def analyze(self, text: str) -> PreFilterReport:
        if text is None:
            raise ValueError("Text must not be None.")
        words = text.split()
        sentences = split_sentences(text)
        word_count = len(words)

        report = PreFilterReport(
            text_length_words=word_count,
            language_tool_available=self.language_tool_available,
        )

        if word_count < 10:
            report.parse_confidence = 0.3
            report.recommendations.append(
                "Text too short for reliable cohesion analysis. Minimum 50 words recommended."
            )
            return report

        self._detect_subject_drop(text, sentences, report)
        self._detect_article_issues(words, word_count, report)
        self._assess_parse_confidence(sentences, word_count, report)

        if report.language_tool_available and self._lt:
            self._run_language_tool(text, report)

        if report.has_critical_grammar_issues:
            report.recommendations.append(
                "High grammar error rate detected. Cohesion indicators may be less "
                "reliable — focus on grammar feedback first, then review cohesion."
            )

        return report

    def _detect_subject_drop(
        self, text: str, sentences: List[str], report: PreFilterReport
    ) -> None:
        for i, sent in enumerate(sentences):
            words = sent.split()
            if len(words) < 2:
                continue
            first = words[0].lower().rstrip(".,!?;:\"'")
            if first in {"is", "are", "was", "were", "has", "have", "had",
                         "will", "would", "can", "could", "should", "must",
                         "seems", "feels", "looks", "becomes", "gets"}:
                report.grammar_issues.append({
                    "index": i,
                    "type": "missing_subject",
                    "category": "grammar",
                    "sentence": sent[:80],
                    "description": "Sentence starts with verb — possible missing subject",
                    "l1_transfer": "Turkish pro-drop (subject pronoun optional in Turkish)",
                })

    def _detect_article_issues(
        self, words: List[str], word_count: int, report: PreFilterReport
    ) -> None:
        if word_count < 30:
            return
        article_count = sum(
            1 for w in words if w.lower() in {"a", "an", "the"}
        )
        article_ratio = article_count / word_count
        if article_ratio < ARTICLE_RATIO_EXPECTED * 0.5:
            report.grammar_issues.append({
                "index": 0,
                "type": "article_deficiency",
                "category": "grammar",
                "description": f"Very low article use ({article_ratio:.3f}, expected ~{ARTICLE_RATIO_EXPECTED})",
                "l1_transfer": "Turkish has no articles (a/an/the)",
                "cohesion_impact": (
                    "Article omission can affect referent tracking — spaCy may "
                    "misidentify noun phrases as new entities when they are definite."
                ),
            })
            report.cohesion_risks.append({
                "type": "article_affects_coreference",
                "description": "Low article rate may cause spaCy to misidentify NPs",
                "confidence_impact": -0.2,
            })

    def _assess_parse_confidence(
        self, sentences: List[str], word_count: int, report: PreFilterReport
    ) -> None:
        confidence = 1.0

        non_standard = 0
        for sent in sentences:
            words = sent.split()
            if len(words) < 2:
                non_standard += 1
                continue
            first = words[0].lower().rstrip(".,!?;:\"'")
            if first in {"is", "are", "was", "were", "has", "have", "had",
                         "will", "would", "can", "could", "should", "must"}:
                non_standard += 1

        if len(sentences) > 0:
            error_rate = non_standard / len(sentences)
            confidence -= error_rate * 0.5

        if word_count < 50:
            confidence -= 0.2

        if len(report.grammar_issues) >= 3:
            confidence -= 0.15

        report.parse_confidence = max(0.1, min(1.0, confidence))

    def _run_language_tool(self, text: str, report: PreFilterReport) -> None:
        if self._lt is None:
            return
        try:
            matches = self._lt.check(text)
        except Exception:
            return

        pronoun_errors = {
            "PRP", "PRP$", "WP", "WP$", "WDT",
            "MORFOLOGIK_RULE_EN_US", "PRONOUN_AGREEMENT",
        }

        for m in matches:
            cat = str(getattr(m, "category", "")).upper() if hasattr(m, "category") else ""
            rule = str(getattr(m, "ruleId", "")).upper() if hasattr(m, "ruleId") else ""

            is_pronoun = (
                "PRON" in cat or "PRON" in rule
                or "PRP" in cat or "PRP" in rule
                or rule in pronoun_errors
            )

            if is_pronoun:
                report.cohesion_risks.append({
                    "type": "language_tool_pronoun",
                    "rule": rule,
                    "message": str(getattr(m, "message", ""))[:120],
                    "offset": getattr(m, "offset", 0),
                    "description": "Grammar checker flagged a pronoun — may affect Cb/Cf tracking",
                    "confidence_impact": -0.1,
                })
            else:
                report.grammar_issues.append({
                    "type": "language_tool_grammar",
                    "rule": rule,
                    "message": str(getattr(m, "message", ""))[:120],
                    "offset": getattr(m, "offset", 0),
                })

        report.parse_confidence -= 0.05 * len(report.cohesion_risks)
        report.parse_confidence = max(0.1, report.parse_confidence)
