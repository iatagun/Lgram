"""
Text Analysis Layer — built on Centering Theory core.

Provides high-level text cohesion analysis with automatic
sentence/paragraph detection, statistical summaries, batch
processing, and export formats.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class EntityGrid:
    entities: List[str]
    matrix: List[List[str]]
    transition_probs: Dict[Tuple[str, str], float]
    score: float


@dataclass
class CohesionGraph:
    adjacency: List[List[float]]
    density: float
    avg_similarity: float
    communities: List[List[int]]
    central_sentences: List[int]


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
    # Entity Grid Model (Barzilay & Lapata 2005/2008)
    # ------------------------------------------------------------------

    _subject_deps = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass"})
    _object_deps = frozenset({"dobj", "iobj", "pobj", "obj", "obl"})

    def entity_grid_score(self, text: str) -> EntityGrid:
        """Score cohesion via entity role transitions across sentences."""
        ct = self._make_ct()
        doc = self.nlp(text)
        sentences_str = [s.text.strip() for s in doc.sents if s.text.strip()]

        # resolve pronouns to canonical entities across sentences
        canonical: Dict[str, str] = {}  # pronoun/text -> entity key
        resolved_roles: List[Dict[str, str]] = []

        for sent_text in sentences_str:
            ct.update_discourse(sent_text)

        for sent_idx, sent_text in enumerate(sentences_str):
            state = ct.discourse_history[sent_idx] if sent_idx < len(ct.discourse_history) else None
            roles: Dict[str, str] = {}
            seen: set = set()

            for token in self.nlp(sent_text):
                if token.pos_ not in ("NOUN", "PROPN", "PRON"):
                    continue
                key = token.text.lower()
                if key in seen:
                    continue
                seen.add(key)

                # resolve pronoun to canonical entity
                entity_key = key
                if token.pos_ == "PRON" and state:
                    for ck in state.forward_centers:
                        if key in ck or ck in key:
                            entity_key = ck
                            break
                    if entity_key == key and state.backward_center:
                        entity_key = state.backward_center
                canonical[key] = entity_key

                dep = token.dep_.split(":")[0]
                if dep in self._subject_deps:
                    roles[entity_key] = "S"
                elif dep in self._object_deps:
                    roles[entity_key] = "O"
                else:
                    roles[entity_key] = "X"
            resolved_roles.append(roles)

        # collect all unique entities
        entity_list: List[str] = []
        entity_index: Dict[str, int] = {}
        for roles in resolved_roles:
            for key in roles:
                if key not in entity_index:
                    entity_index[key] = len(entity_list)
                    entity_list.append(key)

        # build matrix
        n_entities = len(entity_list)
        matrix: List[List[str]] = []
        for roles in resolved_roles:
            row = ["-"] * n_entities
            for key, role in roles.items():
                idx = entity_index[key]
                row[idx] = role
            matrix.append(row)

        if not matrix or len(matrix) < 2:
            return EntityGrid(
                entities=entity_list, matrix=matrix,
                transition_probs={}, score=1.0,
            )

        # coherence rule: smooth transitions (S→S, O→O) > rough (S→-, O→-)
        smooth = 0
        rough = 0
        for col in range(n_entities):
            for row in range(1, len(matrix)):
                prev = matrix[row - 1][col]
                curr = matrix[row][col]
                if prev != "-" and curr != "-" and prev == curr:
                    smooth += 1
                elif prev != "-" and curr == "-":
                    rough += 1

        total = smooth + rough
        score = round(smooth / max(total, 1), 4) if total > 0 else 1.0

        return EntityGrid(
            entities=entity_list, matrix=matrix,
            transition_probs={}, score=score,
        )

    # ------------------------------------------------------------------
    # TextTiling Segmentation (Hearst 1994/1997)
    # ------------------------------------------------------------------

    def texttile_segments(
        self, text: str, w: int = 20, k: int = 10, cutoff: float = 0.2,
    ) -> List[int]:
        """
        TextTiling segmentation — returns sentence indices of boundaries.
        w = pseudo-sentence size in tokens, k = smoothing window,
        cutoff = relative depth threshold (0-1).
        """
        doc = self.nlp(text)
        sents = [s for s in doc.sents if s.text.strip()]
        if len(sents) < 3:
            return [0]

        # build sentence vectors via average token vectors
        sent_vecs: List = []
        for sent in sents:
            vecs = [t.vector for t in sent if t.has_vector and not t.is_stop]
            if vecs:
                avg = sum(vecs) / len(vecs)
                sent_vecs.append(avg)
            else:
                sent_vecs.append(None)

        # similarity between adjacent sentences
        sims: List[float] = []
        for i in range(len(sent_vecs) - 1):
            a, b = sent_vecs[i], sent_vecs[i + 1]
            if a is not None and b is not None and (a_norm := math.sqrt(sum(x*x for x in a))) and (b_norm := math.sqrt(sum(x*x for x in b))):
                sims.append(sum(x*y for x, y in zip(a, b)) / (a_norm * b_norm))
            else:
                sims.append(0.5)

        # smooth
        smoothed = []
        half = k // 2
        for i in range(len(sims)):
            start = max(0, i - half)
            end = min(len(sims), i + half + 1)
            smoothed.append(sum(sims[start:end]) / (end - start))

        # find valleys
        boundaries: List[int] = [0]
        for i in range(1, len(smoothed) - 1):
            left_peak = max(smoothed[:i]) if i > 0 else smoothed[i]
            right_peak = max(smoothed[i+1:]) if i < len(smoothed) - 1 else smoothed[i]
            depth = (left_peak - smoothed[i]) + (right_peak - smoothed[i])
            if depth >= cutoff:
                boundaries.append(i + 1)

        boundaries.sort()
        return boundaries

    def hybrid_boundaries(self, text: str) -> Dict[str, Any]:
        """Combine centering + TextTiling boundaries for high-confidence segments."""
        ct = self._make_ct()
        doc = self.nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]

        centering_bounds = ct.detect_boundaries(sents)
        texttile_bounds = self.texttile_segments(text)

        # intersection = high confidence
        c_set = set(centering_bounds)
        t_set = set(texttile_bounds)
        high_conf = sorted(c_set & t_set)
        all_bounds = sorted(c_set | t_set)

        return {
            "centering": centering_bounds,
            "texttile": texttile_bounds,
            "high_confidence": high_conf,
            "union": all_bounds,
            "agreement": round(len(high_conf) / max(len(all_bounds), 1), 3),
        }

    # ------------------------------------------------------------------
    # Cohesion Graph Analysis
    # ------------------------------------------------------------------

    def build_cohesion_graph(self, text: str) -> CohesionGraph:
        """Build sentence-to-sentence cohesion graph with metrics."""
        doc = self.nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        n = len(sents)

        if n < 2:
            return CohesionGraph(
                adjacency=[[1.0]], density=1.0, avg_similarity=1.0,
                communities=[[0]], central_sentences=[0],
            )

        ct = self._make_ct()
        adj = [[0.0] * n for _ in range(n)]

        for i in range(n):
            adj[i][i] = 1.0

        for i in range(n):
            for j in range(i + 1, n):
                # entity overlap
                si = ct.compute_forward_centers(sents[i])[0]
                sj = ct.compute_forward_centers(sents[j])[0]
                overlap = len(set(si) & set(sj)) / max(len(set(si) | set(sj)), 1)

                # vector similarity
                di = self.nlp(sents[i])
                dj = self.nlp(sents[j])
                vec_sim = di.similarity(dj) if di.vector_norm and dj.vector_norm else 0.5

                weight = round(0.6 * overlap + 0.4 * vec_sim, 4)
                adj[i][j] = weight
                adj[j][i] = weight

        # density
        edge_count = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j] > 0)
        max_edges = n * (n - 1) / 2
        density = round(edge_count / max(max_edges, 1), 4)

        # avg similarity
        similarities = [adj[i][j] for i in range(n) for j in range(i+1, n)]
        avg_sim = round(sum(similarities) / max(len(similarities), 1), 4)

        # simple community detection (threshold-based connected components)
        visited = [False] * n
        communities: List[List[int]] = []
        for i in range(n):
            if visited[i]:
                continue
            comp: List[int] = []
            stack = [i]
            while stack:
                v = stack.pop()
                if visited[v]:
                    continue
                visited[v] = True
                comp.append(v)
                for u in range(n):
                    if adj[v][u] > 0.3 and not visited[u]:
                        stack.append(u)
            communities.append(sorted(comp))

        # centrality: degree
        degrees = [sum(1 for j in range(n) if j != i and adj[i][j] > 0.3) for i in range(n)]
        max_deg = max(degrees) if degrees else 0
        central = [i for i, d in enumerate(degrees) if d == max_deg] if max_deg > 0 else [0]

        return CohesionGraph(
            adjacency=adj, density=density, avg_similarity=avg_sim,
            communities=communities, central_sentences=central,
        )

    # ------------------------------------------------------------------
    # Lexical Chain Approximation
    # ------------------------------------------------------------------

    def lexical_chain_score(self, text: str, threshold: float = 0.5) -> float:
        """Score lexical cohesion via noun-noun similarity chains."""
        doc = self.nlp(text)
        sents = [s for s in doc.sents if s.text.strip()]
        if len(sents) < 2:
            return 1.0

        # extract nouns per sentence
        sent_nouns: List[List[str]] = []
        for sent in sents:
            nouns = [t.text.lower() for t in sent
                    if t.pos_ in ("NOUN", "PROPN") and not t.is_stop]
            sent_nouns.append(nouns)

        # build chains: adjacent sentences with shared/similar nouns
        chain_lengths: List[int] = []
        current_chain = 0
        for i in range(1, len(sent_nouns)):
            prev_nouns = set(sent_nouns[i - 1])
            curr_nouns = set(sent_nouns[i])

            # direct overlap
            if prev_nouns & curr_nouns:
                current_chain += 1
                continue

            # vector similarity
            matched = False
            for pn in prev_nouns:
                for cn in curr_nouns:
                    try:
                        sim = self.nlp(pn).similarity(self.nlp(cn))
                        if sim >= threshold:
                            matched = True
                            break
                    except Exception:
                        pass
                if matched:
                    break

            if matched:
                current_chain += 1
            else:
                if current_chain > 0:
                    chain_lengths.append(current_chain)
                current_chain = 0

        if current_chain > 0:
            chain_lengths.append(current_chain)

        if not chain_lengths:
            return 0.0

        avg_chain = sum(chain_lengths) / len(chain_lengths)
        max_possible = len(sents) - 1
        return round(min(avg_chain / max(max_possible, 1), 1.0), 4)

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
