"""
Text Analysis Layer — built on Centering Theory core.

Provides high-level text cohesion analysis with automatic
sentence/paragraph detection, statistical summaries, batch
processing, and export formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import spacy

from .models.centering_theory import EnhancedCenteringTheory, TransitionType


@dataclass
class SentenceAnalysis:
    index: int
    text: str
    transition: str
    preferred_center: Optional[str]
    backward_center: Optional[str]
    forward_centers: List[str]
    entities: Dict[str, Dict[str, Any]]


@dataclass
class ParagraphAnalysis:
    index: int
    sentences: List[SentenceAnalysis]
    cohesion_score: float
    transition_distribution: Dict[str, float]
    segment_count: int
    clause_analyses: List[Dict[str, Any]]


@dataclass
class TextReport:
    text: str
    sentence_count: int
    paragraph_count: int
    word_count: int
    overall_cohesion: float
    transition_distribution: Dict[str, float]
    paragraphs: List[ParagraphAnalysis]
    segments: List[int]
    quality: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextAnalyzer:
    """High-level text cohesion analyzer with batch and export support."""

    def __init__(
        self,
        model: str = "en_core_web_sm",
        similarity_threshold: float = 0.65,
        gender_map: Optional[Dict[str, str]] = None,
        history_limit: int = 20,
    ):
        self.nlp = spacy.load(model)
        self.model_name = model
        self._ct_kwargs = {
            "similarity_threshold": similarity_threshold,
            "gender_map": gender_map,
            "history_limit": history_limit,
        }

    def _make_ct(self) -> EnhancedCenteringTheory:
        return EnhancedCenteringTheory(self.nlp, **self._ct_kwargs)

    # ------------------------------------------------------------------
    # Single text analysis
    # ------------------------------------------------------------------

    def analyze(self, text: str, include_clauses: bool = True) -> TextReport:
        """Full analysis of a single text."""
        doc = self.nlp(text)
        paragraphs = self._split_paragraphs(text, doc)
        ct = self._make_ct()

        paragraph_results: List[ParagraphAnalysis] = []
        all_sentences: List[SentenceAnalysis] = []
        all_transitions: Dict[TransitionType, int] = {}

        for p_idx, para_sents in enumerate(paragraphs):
            sent_analyses: List[SentenceAnalysis] = []
            clause_analyses: List[Dict[str, Any]] = []

            for s_idx, sent_text in enumerate(para_sents):
                state = ct.update_discourse(sent_text)
                t = state.transition
                if t:
                    all_transitions[t] = all_transitions.get(t, 0) + 1

                sa = SentenceAnalysis(
                    index=s_idx,
                    text=sent_text,
                    transition=t.value if t else "?",
                    preferred_center=state.preferred_center,
                    backward_center=state.backward_center,
                    forward_centers=state.forward_centers,
                    entities={
                        k: {"text": v.text, "pos": v.pos, "dep": v.dep,
                            "is_person": v.is_person, "gender": v.gender}
                        for k, v in state._entity_map.items()
                    },
                )
                sent_analyses.append(sa)
                all_sentences.append(sa)

                if include_clauses:
                    clauses = ct.extract_clauses(sent_text)
                    if len(clauses) >= 2:
                        clause_analyses.append({
                            "sentence": sent_text,
                            "clauses": clauses,
                            "analysis": ct.analyze_intra_sentential(sent_text),
                        })

            total = sum(all_transitions.values()) or 1
            para_score = round(
                sum(all_transitions.get(t, 0) * ct._transition_weights.get(t, 0.5)
                    for t in TransitionType) / total, 4
            )
            para_dist = {t.value: all_transitions.get(t, 0) / total
                        for t in TransitionType}

            boundaries = ct.detect_boundaries([s.text for s in sent_analyses])

            paragraph_results.append(ParagraphAnalysis(
                index=p_idx,
                sentences=sent_analyses,
                cohesion_score=para_score,
                transition_distribution=para_dist,
                segment_count=len(boundaries),
                clause_analyses=clause_analyses,
            ))

        all_trans_total = sum(all_transitions.values()) or 1
        overall_score = round(
            sum(all_transitions.get(t, 0) * ct._transition_weights.get(t, 0.5)
                for t in TransitionType) / all_trans_total, 4
        )
        overall_dist = {t.value: all_transitions.get(t, 0) / all_trans_total
                       for t in TransitionType}

        all_texts = [s.text for s in all_sentences]
        all_boundaries = ct.detect_boundaries(all_texts)

        if overall_score >= 0.80:
            quality = "high"
        elif overall_score >= 0.55:
            quality = "medium"
        else:
            quality = "low"

        return TextReport(
            text=text,
            sentence_count=len(all_sentences),
            paragraph_count=len(paragraphs),
            word_count=len(text.split()),
            overall_cohesion=overall_score,
            transition_distribution=overall_dist,
            paragraphs=paragraph_results,
            segments=all_boundaries,
            quality=quality,
            metadata={"model": self.model_name},
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def analyze_batch(
        self, texts: List[str], labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze multiple texts and return comparative statistics."""
        reports = [self.analyze(t) for t in texts]
        scores = [r.overall_cohesion for r in reports]

        rankings = []
        for i, r in enumerate(reports):
            label = labels[i] if labels and i < len(labels) else f"text_{i+1}"
            rankings.append({
                "label": label,
                "cohesion": r.overall_cohesion,
                "quality": r.quality,
                "sentences": r.sentence_count,
                "segments": len(r.segments),
            })

        rankings.sort(key=lambda x: x["cohesion"], reverse=True)

        return {
            "count": len(texts),
            "mean_cohesion": round(sum(scores) / len(scores), 4),
            "min_cohesion": round(min(scores), 4),
            "max_cohesion": round(max(scores), 4),
            "rankings": rankings,
            "reports": reports,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self, report: TextReport) -> Dict[str, Any]:
        """Convert report to JSON-serializable dict."""
        return {
            "sentence_count": report.sentence_count,
            "paragraph_count": report.paragraph_count,
            "word_count": report.word_count,
            "overall_cohesion": report.overall_cohesion,
            "transition_distribution": report.transition_distribution,
            "segments": len(report.segments),
            "quality": report.quality,
            "metadata": report.metadata,
            "paragraphs": [
                {
                    "index": p.index,
                    "sentence_count": len(p.sentences),
                    "cohesion_score": p.cohesion_score,
                    "segment_count": p.segment_count,
                    "sentences": [
                        {
                            "index": s.index,
                            "text": s.text,
                            "transition": s.transition,
                            "preferred_center": s.preferred_center,
                            "backward_center": s.backward_center,
                            "forward_centers": s.forward_centers,
                            "entities": s.entities,
                        }
                        for s in p.sentences
                    ],
                    "clauses": p.clause_analyses,
                }
                for p in report.paragraphs
            ],
        }

    def to_json(self, report: TextReport, indent: int = 2) -> str:
        """Export report as JSON string."""
        return json.dumps(self.to_dict(report), indent=indent, ensure_ascii=False)

    def to_summary(self, report: TextReport) -> str:
        """Generate human-readable summary string."""
        lines = [
            f"Text Analysis Report",
            f"{'='*50}",
            f"Sentences: {report.sentence_count}  |  Paragraphs: {report.paragraph_count}",
            f"Words: {report.word_count}  |  Cohesion: {report.overall_cohesion:.3f}  [{report.quality}]",
            f"Segments: {len(report.segments)}",
            f"",
            f"Transitions:",
        ]
        for t, pct in report.transition_distribution.items():
            bar = "#" * int(pct * 30)
            lines.append(f"  {t:14s} {bar} {pct:.0%}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split_paragraphs(self, text: str, doc) -> List[List[str]]:
        """Split text into paragraphs, each containing sentences."""
        raw_paras = text.split("\n\n")
        paragraphs: List[List[str]] = []
        pos = 0

        for raw in raw_paras:
            if not raw.strip():
                continue
            para_doc = self.nlp(raw.strip())
            sents = [s.text.strip() for s in para_doc.sents if s.text.strip()]
            if sents:
                paragraphs.append(sents)

        if not paragraphs:
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            if sents:
                paragraphs.append(sents)

        return paragraphs
