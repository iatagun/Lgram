"""
Layer: Grammar Error Detection (LanguageTool-powered, no heuristics).

Replaces heuristic PreFilter with real NLP-based grammar checking.
Uses LanguageTool (open-source, rule-based, 6000+ English rules).

Requires: pip install language-tool-python
Requires: Java 8+ (LanguageTool runs on JVM)

Architecture:
  ESSAY TEXT
      │
      ├─► LanguageTool.check() → List[Match]
      │     ├─ Category: GRAMMAR, TYPOGRAPHY, STYLE, etc.
      │     ├─ Rule ID + message + replacement suggestions
      │     └─ Position (offset, length)
      │
      ├─► Error categorization:
      │     ├─ GRAMMAR → counted toward rubric Grammar dimension
      │     ├─ PRONOUN-related → flagged for cohesion impact
      │     └─ TYPO/SPELLING → counted toward rubric Mechanics
      │
      └─► GrammarLayerResult: error rate, severity, per-category breakdown
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .models import Essay, LayerResult
from .utils import split_sentences


@dataclass
class GrammarIssue:
    rule_id: str
    category: str
    message: str
    offset: int
    length: int
    replacements: List[str]
    sentence: str
    affects_cohesion: bool = False


@dataclass
class GrammarReport:
    total_errors: int
    error_rate: float
    grammar_errors: int
    spelling_errors: int
    style_issues: int
    pronoun_errors: int
    issues: List[GrammarIssue]
    score: float


class GrammarLayer:
    """
    Real grammar checking via LanguageTool. No heuristics.

    Error rate scoring:
      - < 2 errors/100 words → 90-100
      - 2-5 errors/100 words → 70-90
      - 5-10 errors/100 words → 50-70
      - > 10 errors/100 words → 30-50
      - > 20 errors/100 words → < 30

    Pronoun-related errors are flagged as cohesion-affecting.
    """

    COHESION_RULE_PATTERNS = {
        "PRP", "PRONOUN", "AGREEMENT", "GENDER", "ANAPHORA",
        "REFLEXIVE", "RELATIVE", "WH_WORD", "POSSESSIVE",
        "MORFOLOGIK_RULE_EN_US", "CONFUSION_OF_HE",
        "HE_OR_SHE", "THEY_THEM", "IT_IS",
    }

    def __init__(self, language: str = "en-US"):
        self._language = language
        self._lt = None
        self._load_error: Optional[str] = None
        self._init_attempted = False

    @property
    def available(self) -> bool:
        if self._lt is not None:
            return True
        if self._init_attempted:
            return False
        self._init_attempted = True
        try:
            import language_tool_python
            self._lt = language_tool_python.LanguageTool(self._language)
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    def evaluate(self, essay: Essay) -> LayerResult:
        words = essay.text.split()
        word_count = len(words)
        sent_count = max(len(split_sentences(essay.text)), 1)

        if not self.available:
            return LayerResult(
                layer_name="Grammar",
                score=50.0,
                normalized_score=0.5,
                raw_details={
                    "error": f"LanguageTool not available: {self._load_error}",
                    "word_count": word_count,
                    "note": "Install language-tool-python and Java 8+ for grammar checking",
                },
                evidence=["Grammar checking unavailable — LanguageTool not loaded"],
                confidence_interval=(40.0, 60.0),
            )

        issues = self._check(text=essay.text, word_count=word_count)
        grammar_errors = issues.get("grammar_errors", 0)
        total = issues.get("total_errors", 0)
        error_rate = total / max(word_count, 1) * 100

        score = self._score_from_error_rate(error_rate)
        normalized = round(score / 100.0, 3)

        evidence: List[str] = []
        for iss in issues.get("issues", [])[:3]:
            ctx = iss.get("sentence", "")[:80]
            evidence.append(f"[{iss.get('rule_id', '?')}] {iss.get('message', '')[:100]} — '{ctx}'")

        if not evidence and word_count >= 20:
            evidence.append("No grammar errors detected")

        sem = 3.0 / math.sqrt(max(total, 1)) if total > 0 else 3.0
        ci = (max(0, score - sem * 2), min(100, score + sem * 2))

        return LayerResult(
            layer_name="Grammar",
            score=round(score, 1),
            normalized_score=normalized,
            raw_details={
                "total_errors": total,
                "error_rate_per_100_words": round(error_rate, 1),
                "grammar_errors": grammar_errors,
                "spelling_errors": issues.get("spelling_errors", 0),
                "style_issues": issues.get("style_issues", 0),
                "pronoun_errors": issues.get("pronoun_errors", 0),
                "word_count": word_count,
                "sentence_count": sent_count,
            },
            evidence=evidence[:3],
            confidence_interval=(round(ci[0], 1), round(ci[1], 1)),
        )

    def _check(self, text: str, word_count: int) -> Dict[str, Any]:
        if self._lt is None or word_count < 5:
            return {"total_errors": 0, "grammar_errors": 0, "spelling_errors": 0,
                    "style_issues": 0, "pronoun_errors": 0, "issues": []}

        try:
            matches = self._lt.check(text)
        except Exception:
            return {"total_errors": 0, "grammar_errors": 0, "spelling_errors": 0,
                    "style_issues": 0, "pronoun_errors": 0, "issues": []}

        grammar_count = 0
        spelling_count = 0
        style_count = 0
        pronoun_count = 0
        issue_list: List[Dict[str, Any]] = []

        for m in matches:
            rule = str(getattr(m, "ruleId", ""))
            cat = str(getattr(m, "category", "")).upper()
            msg = str(getattr(m, "message", ""))
            offset = getattr(m, "offset", 0)
            length = getattr(m, "errorLength", 0)
            reps = [str(r) for r in getattr(m, "replacements", [])[:3]]

            sentence = ""
            if offset < len(text):
                sentence = text[max(0, offset - 20):min(len(text), offset + length + 40)]

            is_pronoun_related = any(
                p in rule.upper() or p in cat
                for p in self.COHESION_RULE_PATTERNS
            )

            if "SPELL" in cat or "TYPO" in cat or "CASING" in cat:
                spelling_count += 1
            elif "STYLE" in cat or "REDUNDANCY" in cat:
                style_count += 1
            else:
                grammar_count += 1

            if is_pronoun_related:
                pronoun_count += 1

            issue_list.append({
                "rule_id": rule,
                "category": cat,
                "message": msg,
                "offset": offset,
                "length": length,
                "replacements": reps,
                "sentence": sentence.strip(),
                "affects_cohesion": is_pronoun_related,
            })

        return {
            "total_errors": len(matches),
            "grammar_errors": grammar_count,
            "spelling_errors": spelling_count,
            "style_issues": style_count,
            "pronoun_errors": pronoun_count,
            "issues": issue_list,
        }

    @staticmethod
    def _score_from_error_rate(rate: float) -> float:
        if rate <= 2:
            return 90.0 + (2 - rate) * 5
        elif rate <= 5:
            return 90.0 - (rate - 2) * 6.67
        elif rate <= 10:
            return 70.0 - (rate - 5) * 4.0
        elif rate <= 20:
            return 50.0 - (rate - 10) * 2.0
        else:
            return max(10.0, 30.0 - (rate - 20) * 1.0)
