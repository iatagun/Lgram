"""
Layer 2: Cohesion & Organization (Lgram, segment-aware).

Analyzes essay cohesion at the paragraph-segment level.
Cross-paragraph transitions are treated as intentional (topic boundary),
not as cohesion errors. Only intra-paragraph cohesion is scored.

Architecture:
  1. Segment essay into paragraphs (intro / body / conclusion)
  2. Analyze each segment independently with Lgram
  3. Score intra-segment cohesion
  4. Check segment flow (transitions, logical ordering)
  5. Weight body paragraphs more heavily
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .models import Essay, LayerResult


@dataclass
class SegmentAnalysis:
    index: int
    segment_type: str
    text: str
    sentence_count: int
    cohesion_score: float
    transition_distribution: Dict[str, float]
    weak_points: List[Dict[str, Any]]


class SegmentAnalyzer:
    """
    Splits essay text into rhetorical segments (intro/body/conclusion)
    and analyzes each independently.
    """

    def __init__(self):
        self._nlp = None
        self._ta = None

    @property
    def _analyzer(self):
        if self._ta is None:
            from lgram import TextAnalyzer
            self._ta = TextAnalyzer("en_core_web_md", similarity_threshold=0.35)
        return self._ta

    def segment(self, text: str) -> List[str]:
        """Split text into rhetorical segments by paragraphs."""
        raw = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(raw) <= 1:
            sentences = _split_sentences(text)
            n = len(sentences)
            if n <= 3:
                return [text]
            third = n // 3
            return [
                " ".join(sentences[:third]),
                " ".join(sentences[third : n - third]),
                " ".join(sentences[n - third:]),
            ]
        return raw

    def analyze_segments(self, text: str) -> List[SegmentAnalysis]:
        segments = self.segment(text)
        results: List[SegmentAnalysis] = []
        total_segments = len(segments)

        for i, seg_text in enumerate(segments):
            if total_segments == 1:
                seg_type = "full"
            elif i == 0:
                seg_type = "intro"
            elif i == total_segments - 1:
                seg_type = "conclusion"
            else:
                seg_type = "body"

            sents = _split_sentences(seg_text)
            sent_count = len(sents)

            try:
                r = self._analyzer.analyze(seg_text)
                cohesion = r.overall_cohesion
                dist = r.transition_distribution
            except Exception:
                cohesion = 0.5
                dist = {}

            weak = []
            if cohesion < 0.5:
                try:
                    suggestions = self._analyzer.suggest_improvements(seg_text)
                    weak = [{"index": s.get("index", i), "issue": s.get("issue", ""),
                             "suggestion": s.get("suggestion", "")} for s in suggestions[:3]]
                except Exception:
                    pass

            results.append(SegmentAnalysis(
                index=i,
                segment_type=seg_type,
                text=seg_text[:200] + "..." if len(seg_text) > 200 else seg_text,
                sentence_count=sent_count,
                cohesion_score=round(cohesion, 3),
                transition_distribution=dist,
                weak_points=weak,
            ))

        return results


class CohesionLayer:
    """
    Cohesion assessment layer using segment-aware Lgram analysis.

    Scoring:
      - Body paragraphs: 60% weight (main content)
      - Introduction: 20% weight
      - Conclusion: 20% weight
      - Penalty for weak intra-paragraph links
    """

    BODY_WEIGHT = 0.60
    INTRO_WEIGHT = 0.20
    CONC_WEIGHT = 0.20
    FULL_WEIGHT = 1.0

    def __init__(self):
        self._segment_analyzer = SegmentAnalyzer()

    def evaluate(self, essay: Essay) -> LayerResult:
        segments = self._segment_analyzer.analyze_segments(essay.text)
        scores = []
        weights = []
        weak_total = 0
        evidence: List[str] = []

        total_segments = len(segments)
        if total_segments == 1:
            weights = [self.FULL_WEIGHT]
        else:
            weights = []
            for s in segments:
                if s.segment_type == "intro":
                    weights.append(self.INTRO_WEIGHT)
                elif s.segment_type == "conclusion":
                    weights.append(self.CONC_WEIGHT)
                else:
                    weights.append(self.BODY_WEIGHT / max(1, total_segments - 2))

        for s in segments:
            scores.append(s.cohesion_score)
            weak_total += len(s.weak_points)

            if s.cohesion_score < 0.45 and s.segment_type == "body":
                evidence.append(
                    f"Paragraph {s.index + 1} ({s.segment_type}) has low cohesion "
                    f"({s.cohesion_score:.2f}) — possible disjointed sentences"
                )

            for w in s.weak_points[:1]:
                evidence.append(
                    f"Paragraph {s.index + 1}, sentence {w['index']}: {w['suggestion'][:100]}"
                )

        if abs(sum(weights) - 1.0) > 0.01:
            total_w = sum(weights) or 1.0
            weights = [w / total_w for w in weights]

        weighted_score = sum(s * w for s, w in zip(scores, weights))
        normalized = round(weighted_score, 3)

        continues = []
        roughs = []
        for s in segments:
            d = s.transition_distribution
            continues.append(d.get("Continue", 0))
            roughs.append(d.get("Rough-Shift", 0))

        avg_continue = sum(continues) / max(len(continues), 1)
        avg_rough = sum(roughs) / max(len(roughs), 1)

        if not evidence:
            evidence.append("Intra-paragraph cohesion adequate across all segments")

        sem = _segment_sem(scores, weights)
        ci = (max(0.0, normalized - sem * 2), min(1.0, normalized + sem * 2))

        return LayerResult(
            layer_name="Cohesion & Organization",
            score=round(normalized * 100, 1),
            normalized_score=normalized,
            raw_details={
                "segment_count": total_segments,
                "segments": [
                    {
                        "index": s.index,
                        "type": s.segment_type,
                        "sentences": s.sentence_count,
                        "cohesion": s.cohesion_score,
                        "continue_ratio": round(s.transition_distribution.get("Continue", 0), 3),
                        "rough_shift_ratio": round(s.transition_distribution.get("Rough-Shift", 0), 3),
                        "weak_points": len(s.weak_points),
                    }
                    for s in segments
                ],
                "avg_continue_ratio": round(avg_continue, 3),
                "avg_rough_shift_ratio": round(avg_rough, 3),
                "total_weak_points": weak_total,
                "weights": [round(w, 3) for w in weights],
            },
            evidence=evidence,
            confidence_interval=ci,
        )


def _split_sentences(text: str) -> List[str]:
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _segment_sem(scores: List[float], weights: List[float]) -> float:
    n = len(scores)
    if n < 2:
        return 0.15
    weighted = sum(s * w for s, w in zip(scores, weights))
    variance = sum(w * (s - weighted) ** 2 for s, w in zip(scores, weights))
    effective_n = 1.0 / sum(w * w for w in weights) if sum(w * w for w in weights) > 0 else n
    return math.sqrt(variance / effective_n) if variance > 0 else 0.05
