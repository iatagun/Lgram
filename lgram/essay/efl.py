"""
EFL Writing Assessment Module.

Target: English as a Foreign Language (EFL) writing assignments.
Focus: Turkish L1 learners, CEFR B1-B2-C1 levels.

Standard EFL rubric (5 dimensions):
  1. Grammar — morphosyntactic accuracy
  2. Content — thesis, argument, evidence
  3. Organization — cohesion, paragraphing, transitions
  4. Style/Expression — vocabulary range, register
  5. Mechanics — spelling, punctuation, formatting

Key advantage over general AES: L1-specific transfer analysis
  - Turkish pro-drop → English pronoun errors
  - Turkish gender-neutral pronouns → he/she confusion
  - Turkish article-less grammar → a/an/the errors
  - Turkish SOV → English SVO transfer

Comparison benchmark: Yavuz (2025) — EFL teachers vs ChatGPT/Bard
on the same 5-dimension rubric.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .models import Essay, LayerResult, RubricCriterion
from .utils import split_sentences, ARTICLE_RATIO_EXPECTED


EFL_RUBRIC = [
    RubricCriterion(
        name="Grammar",
        weight=0.20,
        description="Morphosyntactic accuracy, tense consistency, article/preposition use, "
                    "sentence structure correctness.",
    ),
    RubricCriterion(
        name="Content",
        weight=0.25,
        description="Thesis clarity, argument development, evidence use, topic relevance, "
                    "critical thinking depth.",
    ),
    RubricCriterion(
        name="Organization",
        weight=0.20,
        description="Cohesion, paragraph structure, transition use, "
                    "logical flow between ideas — as modeled by Centering Theory "
                    "(surface-level cohesion, not deep coherence).",
    ),
    RubricCriterion(
        name="Style & Expression",
        weight=0.15,
        description="Vocabulary range, register appropriateness, collocation naturalness, "
                    "idiomaticity, avoidance of L1 transfer calques.",
    ),
    RubricCriterion(
        name="Mechanics",
        weight=0.20,
        description="Spelling, punctuation, capitalization, paragraph formatting, "
                    "word count adequacy.",
    ),
]

CEFR_PROFILES: Dict[str, Dict[str, Any]] = {
    "B1": {
        "label": "Intermediate",
        "expected_score_range": (55, 75),
        "typical_issues": [
            "Occasional pronoun reference ambiguity",
            "Simple cohesive devices (and/but/because) overused",
            "Limited vocabulary range with L1 calques",
            "Article errors frequent but meaning preserved",
            "Paragraph structure present but transitions weak",
        ],
        "cohesion_threshold": 0.50,
        "rough_shift_tolerance": 0.35,
        "min_length_words": 150,
        "target_length_words": 250,
    },
    "B2": {
        "label": "Upper Intermediate",
        "expected_score_range": (65, 85),
        "typical_issues": [
            "Pronoun reference mostly clear, occasional ambiguity",
            "Range of cohesive devices used but sometimes inappropriate",
            "Vocabulary adequate, some collocation errors",
            "Article errors reduced but persist in complex NPs",
            "Clear paragraphing with logical transitions",
        ],
        "cohesion_threshold": 0.60,
        "rough_shift_tolerance": 0.25,
        "min_length_words": 250,
        "target_length_words": 350,
    },
    "C1": {
        "label": "Advanced",
        "expected_score_range": (75, 95),
        "typical_issues": [
            "Pronoun reference consistently clear",
            "Wide range of cohesive devices used appropriately",
            "Rich vocabulary with minimal L1 transfer",
            "Occasional article/preposition errors in complex structures",
            "Effective paragraph organization with sophisticated transitions",
        ],
        "cohesion_threshold": 0.70,
        "rough_shift_tolerance": 0.18,
        "min_length_words": 350,
        "target_length_words": 500,
    },
}


@dataclass
class L1TransferReport:
    text: str
    pro_drop_issues: List[Dict[str, Any]] = field(default_factory=list)
    gender_pronoun_issues: List[Dict[str, Any]] = field(default_factory=list)
    article_estimate: Dict[str, Any] = field(default_factory=dict)
    word_order_issues: List[Dict[str, Any]] = field(default_factory=list)
    overall_transfer_score: float = 0.0
    summary: str = ""


class L1TransferAnalyzer:
    """
    Analyzes Turkish L1 → English L2 transfer patterns.

    Turkish-specific phenomena:
      1. Pro-drop: Turkish drops subject pronouns (gidiyorum = I am going).
         L2 result: missing subject pronouns in English.
      2. Gender: Turkish has no grammatical gender (o = he/she/it).
         L2 result: he/she confusion, especially in longer discourse.
      3. Articles: Turkish has no definite/indefinite articles.
         L2 result: a/an/the omission or overuse.
      4. Word order: Turkish is SOV, English is SVO.
         L2 result: verb-final tendencies, especially in subordinate clauses.
      5. Cohesion: Turkish uses more lexical repetition, fewer pronouns.
         L2 result: overuse of full NPs, underuse of pronouns → clunky text.
    """

    GENDER_PRONOUN_PATTERNS = [
        (re.compile(r"\bhe\b", re.I), "male"),
        (re.compile(r"\bshe\b", re.I), "female"),
        (re.compile(r"\bhis\b", re.I), "male"),
        (re.compile(r"\bher\b", re.I), "female"),
        (re.compile(r"\bhim\b", re.I), "male"),
    ]

    def analyze(self, text: str) -> L1TransferReport:
        sentences = split_sentences(text)
        words = text.split()

        report = L1TransferReport(text=text[:200] + "..." if len(text) > 200 else text)

        miss = self._detect_missing_subject(sentences)
        gender = self._detect_gender_confusion(sentences)
        article = self._estimate_article_issues(words, sentences)
        word_order = self._detect_sov_transfer(sentences)

        report.pro_drop_issues = miss
        report.gender_pronoun_issues = gender
        report.article_estimate = article
        report.word_order_issues = word_order

        subscore_miss = max(0.0, 1.0 - len(miss) * 0.3)
        subscore_gender = max(0.0, 1.0 - len(gender) * 0.3)
        subscore_article = article.get("adequacy", 1.0)
        subscore_order = max(0.0, 1.0 - len(word_order) * 0.2)

        report.overall_transfer_score = round(
            subscore_miss * 0.30
            + subscore_gender * 0.20
            + subscore_article * 0.30
            + subscore_order * 0.20,
            3,
        )

        parts = [f"L1 Transfer: {report.overall_transfer_score:.2f} (1.0 = minimal L1 influence)"]
        if miss:
            parts.append(f"  Pro-drop: {len(miss)} potential missing-subject sentences")
        if gender:
            parts.append(f"  Gender: {len(gender)} potential he/she confusion points")
        parts.append(f"  Articles: adequacy={subscore_article:.2f}")
        if word_order:
            parts.append(f"  Word order: {len(word_order)} potential SOV-transfer clauses")
        report.summary = "\n".join(parts)

        return report

    def _detect_missing_subject(self, sentences: List[str]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for i, sent in enumerate(sentences):
            words = sent.split()
            if len(words) < 3:
                continue
            first_word = words[0].lower().rstrip(".,!?;:\"'")
            if first_word in {"is", "are", "was", "were", "has", "have", "had",
                             "will", "would", "can", "could", "should", "must",
                             "seems", "feels", "looks", "becomes", "gets"}:
                issues.append({
                    "index": i,
                    "sentence": sent[:80],
                    "type": "pro-drop",
                    "description": "Sentence starts with verb — possible missing subject (Turkish pro-drop transfer)",
                })
            if i > 0 and len(words) > 2 and words[0].lower() == "and":
                second = words[1].lower().rstrip(".,!?;:\"'") if len(words) > 1 else ""
                if second in {"then", "also", "so", "therefore", "is", "are", "was", "were"}:
                    issues.append({
                        "index": i,
                        "sentence": sent[:80],
                        "type": "pro-drop-conjunction",
                        "description": "And-clause without explicit subject — possible pro-drop transfer",
                    })
        return issues

    def _detect_gender_confusion(self, sentences: List[str]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        male_count = 0
        female_count = 0
        for sent in sentences:
            for pattern, gender in self.GENDER_PRONOUN_PATTERNS:
                count = len(pattern.findall(sent))
                if gender == "male":
                    male_count += count
                else:
                    female_count += count

        if male_count > 0 and female_count == 0:
            issues.append({
                "index": 0,
                "type": "gender-asymmetry",
                "description": f"Only male pronouns used ({male_count} instances) — check for generic 'he' Turkish transfer (Turkish 'o' = he/she/it)",
            })
        if female_count > 0 and male_count == 0:
            issues.append({
                "index": 0,
                "type": "gender-asymmetry",
                "description": f"Only female pronouns used ({female_count} instances) — check for generic 'she' usage pattern",
            })

        text_lower = " ".join(sentences).lower()
        alternating = re.findall(r"\b(he|she)\b.*?\b(she|he)\b", text_lower)
        if len(alternating) > 3:
            issues.append({
                "index": 0,
                "type": "gender-alternation",
                "description": f"Frequent he/she alternation ({len(alternating)} patterns) — possible gender confusion (Turkish 'o' has no gender)",
            })

        return issues

    def _estimate_article_issues(
        self, words: List[str], sentences: List[str]
    ) -> Dict[str, Any]:
        word_count = len(words)
        if word_count < 30:
            return {"adequacy": 1.0, "note": "Text too short for article analysis"}

        a_count = sum(1 for w in words if w.lower() == "a")
        an_count = sum(1 for w in words if w.lower() == "an")
        the_count = sum(1 for w in words if w.lower() == "the")
        total_articles = a_count + an_count + the_count
        article_ratio = total_articles / word_count

        expected_ratio = ARTICLE_RATIO_EXPECTED

        if article_ratio < 0.03:
            adequacy = 0.4
            note = "Very low article use — typical Turkish L1 transfer (Turkish has no articles)"
        elif article_ratio < 0.05:
            adequacy = 0.7
            note = "Below-expected article rate — possible L1 transfer"
        elif article_ratio < 0.09:
            adequacy = 1.0
            note = "Article use within expected range"
        else:
            adequacy = 0.75
            note = "High article rate — check for overuse (overcorrection from L1 awareness)"

        return {
            "a_count": a_count,
            "an_count": an_count,
            "the_count": the_count,
            "total_articles": total_articles,
            "article_ratio": round(article_ratio, 3),
            "expected_ratio": expected_ratio,
            "adequacy": adequacy,
            "note": note,
        }

    def _detect_sov_transfer(self, sentences: List[str]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for i, sent in enumerate(sentences):
            words = sent.split()
            if len(words) < 8:
                continue
            last = words[-1].lower().rstrip(".,!?;:\"'")
            if last in {"is", "are", "was", "were", "has", "have", "had",
                         "will", "would", "can", "could", "should", "must",
                         "went", "came", "gave", "took", "made", "did", "got",
                         "sees", "says", "goes", "comes"}:
                issues.append({
                    "index": i,
                    "sentence": sent[:80],
                    "type": "sov-transfer",
                    "description": "Verb-final pattern — possible Turkish SOV transfer",
                })
        return issues



def get_cefr_profile(level: str) -> Dict[str, Any]:
    """Get CEFR calibration profile for the given level."""
    level_upper = level.upper()
    if level_upper not in CEFR_PROFILES:
        raise ValueError(
            f"Unknown CEFR level: {level}. Available: {list(CEFR_PROFILES.keys())}"
        )
    return CEFR_PROFILES[level_upper]


def estimate_cefr_level(text: str) -> Tuple[str, float]:
    """
    Estimate CEFR level from text features.

    Heuristic (not definitive — for initial placement only):
      - Word count → rough proficiency proxy
      - Sentence complexity → avg length

    Returns: (level_label, confidence)
    """
    words = text.split()
    sentences = split_sentences(text)
    word_count = len(words)
    sent_count = len(sentences)
    avg_sent_len = word_count / max(sent_count, 1)

    if word_count < 100:
        return ("B1", 0.6)
    elif word_count < 250:
        return ("B1", 0.7)
    elif word_count < 400:
        if avg_sent_len > 15:
            return ("B2", 0.6)
        return ("B2", 0.5)
    else:
        if avg_sent_len > 18:
            return ("C1", 0.6)
        return ("B2", 0.7)
