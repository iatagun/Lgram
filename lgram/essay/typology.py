"""
Error Typology Framework.

Systematically categorizes cohesion issues in EFL writing.
Produces structured typology for:
  - Research contribution (publication-ready error patterns)
  - Teacher dashboards (what to focus on)
  - Curriculum feedback (most common L1 transfer errors)

Error categories are L1-aware: tagged with whether likely
caused by Turkish L1 transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCategory(Enum):
    PRONOUN_REFERENCE = "pronoun_reference"
    MISSING_TRANSITION = "missing_transition"
    ABRUPT_TOPIC_SHIFT = "abrupt_topic_shift"
    OVERUSE_REPETITION = "overuse_repetition"
    GENDER_MISMATCH = "gender_mismatch"
    ARTICLE_COHESION = "article_cohesion"
    SUBJECT_DROP = "subject_drop"
    SOV_TRANSFER = "sov_transfer"


@dataclass
class TypologyEntry:
    category: ErrorCategory
    frequency: int
    severity: float
    cefr_levels: Dict[str, int]
    l1_correlation: float
    examples: List[str]
    teacher_guidance: str


@dataclass
class TypologyReport:
    total_essays: int
    total_errors: int
    entries: List[TypologyEntry]
    l1_transfer_ratio: float
    most_common_category: str
    summary: str


class ErrorTypology:
    """
    Builds an error typology from analyzed reports.

    Categories are grounded in:
      - Centering Theory (Cb/Cf tracking)
      - L1 transfer research (Turkish → English)
      - EFL writing pedagogy (CEFR descriptors)

    Usage:
        typology = ErrorTypology()
        typology.feed(report, cefr_level="B2")
        report = typology.build()
        print(report.summary)
    """

    L1_SENSITIVE_CATEGORIES = {
        ErrorCategory.SUBJECT_DROP,
        ErrorCategory.GENDER_MISMATCH,
        ErrorCategory.ARTICLE_COHESION,
        ErrorCategory.SOV_TRANSFER,
        ErrorCategory.PRONOUN_REFERENCE,
    }

    GUIDANCE = {
        ErrorCategory.PRONOUN_REFERENCE: (
            "Check pronoun-antecedent clarity. Turkish learners often "
            "omit subjects or overuse full nouns instead of pronouns."
        ),
        ErrorCategory.MISSING_TRANSITION: (
            "Paragraph/sentence transitions are missing. Turkish learners "
            "sometimes rely on implicit connections common in Turkish discourse."
        ),
        ErrorCategory.ABRUPT_TOPIC_SHIFT: (
            "Topic shifts without signaling. May reflect Turkish paragraph "
            "organization patterns (less explicit topic marking)."
        ),
        ErrorCategory.OVERUSE_REPETITION: (
            "Full noun phrases repeated instead of pronoun substitution. "
            "Turkish uses more lexical repetition; English prefers pronouns."
        ),
        ErrorCategory.GENDER_MISMATCH: (
            "He/she confusion. Turkish 'o' is gender-neutral — this is an "
            "L1 transfer error, not a content/meaning error."
        ),
        ErrorCategory.ARTICLE_COHESION: (
            "Article patterns affect referent tracking. Turkish has no articles, "
            "so definite/indefinite noun phrase tracking may be inconsistent."
        ),
        ErrorCategory.SUBJECT_DROP: (
            "Missing subject pronouns. Turkish pro-drop allows subject omission — "
            "this is a grammar transfer, not a cohesion choice."
        ),
        ErrorCategory.SOV_TRANSFER: (
            "Verb-final word order patterns. Turkish SOV structure may cause "
            "non-standard clause ordering that affects cohesion flow."
        ),
    }

    def __init__(self):
        self._essay_count = 0
        self._errors: Dict[ErrorCategory, List[Dict[str, Any]]] = {
            cat: [] for cat in ErrorCategory
        }
        self._cefr_counts: Dict[str, int] = {}

    def feed(
        self,
        report: "CAEASReport",
        cefr_level: str = "unknown",
        l1_transfer_score: Optional[float] = None,
        triggers: Optional[List[str]] = None,
    ) -> None:
        self._essay_count += 1
        effective_cefr = cefr_level or getattr(report, "cefr_level", "") or "unknown"
        self._cefr_counts[effective_cefr] = self._cefr_counts.get(effective_cefr, 0) + 1

        effective_triggers = triggers or getattr(report, "triggers", []) or []

        for lr in getattr(report, "layer_results", []):
            if "Cohesion" in lr.layer_name:
                self._categorize_cohesion(lr, effective_cefr)
            elif "Content" in lr.layer_name:
                self._categorize_content(lr, effective_cefr)

        if effective_triggers:
            self._categorize_triggers(effective_triggers, effective_cefr)

    def build(self) -> TypologyReport:
        entries: List[TypologyEntry] = []
        total_errors = 0

        for cat in ErrorCategory:
            instances = self._errors[cat]
            if not instances:
                continue

            freq = len(instances)
            total_errors += freq

            level_dist: Dict[str, int] = {}
            for inst in instances:
                lv = inst.get("cefr", "unknown")
                level_dist[lv] = level_dist.get(lv, 0) + 1

            l1_count = sum(
                1 for inst in instances if inst.get("l1_transfer", False)
            )
            l1_corr = l1_count / max(freq, 1)

            examples = [
                inst.get("evidence", "")[:100]
                for inst in instances[:3]
                if inst.get("evidence")
            ]

            entries.append(TypologyEntry(
                category=cat,
                frequency=freq,
                severity=round(min(1.0, freq / max(self._essay_count, 1)), 3),
                cefr_levels=level_dist,
                l1_correlation=round(l1_corr, 3),
                examples=examples,
                teacher_guidance=self.GUIDANCE.get(cat, ""),
            ))

        entries.sort(key=lambda e: e.frequency, reverse=True)
        l1_total = sum(
            e.frequency for e in entries
            if e.category in self.L1_SENSITIVE_CATEGORIES
        )
        l1_ratio = l1_total / max(total_errors, 1)

        most_common = entries[0].category.value if entries else "none"

        parts = [
            f"Error Typology — {self._essay_count} essays, {total_errors} errors",
            f"L1 transfer ratio: {l1_ratio:.1%} of errors traceable to Turkish L1",
            "",
        ]
        for e in entries[:5]:
            parts.append(
                f"  {e.category.value}: {e.frequency}x "
                f"(severity={e.severity:.2f}, L1={e.l1_correlation:.0%})"
            )

        return TypologyReport(
            total_essays=self._essay_count,
            total_errors=total_errors,
            entries=entries,
            l1_transfer_ratio=l1_ratio,
            most_common_category=most_common,
            summary="\n".join(parts),
        )

    def _categorize_cohesion(self, lr: Any, cefr_level: str) -> None:
        details = getattr(lr, "raw_details", {})
        segments = details.get("segments", [])

        for seg in segments:
            rough = seg.get("rough_shift_ratio", 0)
            if rough > 0.2:
                self._errors[ErrorCategory.MISSING_TRANSITION].append({
                    "cefr": cefr_level,
                    "evidence": f"Segment {seg.get('index', 0)} rough_shift={rough:.2f}",
                    "l1_transfer": False,
                    "weight": int(rough * 10),
                })

            if seg.get("cohesion", 0) < 0.55:
                self._errors[ErrorCategory.ABRUPT_TOPIC_SHIFT].append({
                    "cefr": cefr_level,
                    "evidence": f"Segment {seg.get('index', 0)} cohesion={seg.get('cohesion', 0):.2f}",
                    "l1_transfer": False,
                    "weight": 1,
                })

            if seg.get("continue_ratio", 0) > 0.5:
                self._errors[ErrorCategory.OVERUSE_REPETITION].append({
                    "cefr": cefr_level,
                    "evidence": f"Segment {seg.get('index', 0)} high continue ratio",
                    "l1_transfer": True,
                    "weight": 1,
                })

    def _categorize_content(self, lr: Any, cefr_level: str) -> None:
        evidence = getattr(lr, "evidence", [])
        for e in evidence:
            text = e.lower()
            if "pronoun" in text or "gender" in text:
                self._errors[ErrorCategory.PRONOUN_REFERENCE].append({
                    "cefr": cefr_level,
                    "evidence": e,
                    "l1_transfer": True,
                })

    def _categorize_triggers(
        self, triggers: List[str], cefr_level: str
    ) -> None:
        for t in triggers:
            t_lower = t.lower()
            if "gap" in t_lower:
                self._errors[ErrorCategory.ABRUPT_TOPIC_SHIFT].append({
                    "cefr": cefr_level,
                    "evidence": t,
                    "l1_transfer": False,
                })
            elif "cohesion" in t_lower and "low" in t_lower:
                self._errors[ErrorCategory.MISSING_TRANSITION].append({
                    "cefr": cefr_level,
                    "evidence": t,
                    "l1_transfer": False,
                })
            elif "l1" in t_lower or "transfer" in t_lower:
                self._errors[ErrorCategory.SUBJECT_DROP].append({
                    "cefr": cefr_level,
                    "evidence": t,
                    "l1_transfer": True,
                })
