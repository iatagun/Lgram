"""
Text Analysis Layer — built on Centering Theory core.

Provides high-level text cohesion analysis with automatic
sentence/paragraph detection, statistical summaries, batch
processing, and export formats.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import spacy

from .models.centering_theory import EnhancedCenteringTheory, TransitionType

# cohesion quality thresholds
QUALITY_HIGH = 0.80
QUALITY_MEDIUM = 0.55
HEATMAP_WEAK = 0.3
GRAPH_EDGE_THRESHOLD = 0.3
TREND_SLOPE_THRESHOLD = 0.05
DIFF_DELTA_THRESHOLD = 0.05


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
        similarity_threshold: Optional[float] = None,
        gender_map: Optional[Dict[str, str]] = None,
        history_limit: int = 20,
        use_sentence_transformers: bool = False,
    ):
        self.nlp = spacy.load(model)
        self.model_name = model
        self._st_model = None
        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                pass

        if similarity_threshold is None:
            similarity_threshold = 0.35 if self._st_model else 0.65

        self._ct_kwargs = {
            "similarity_threshold": similarity_threshold,
            "gender_map": gender_map,
            "history_limit": history_limit,
        }

    def _make_ct(self) -> EnhancedCenteringTheory:
        kwargs = dict(self._ct_kwargs)
        if self._st_model:
            def _sim(a: str, b: str) -> float:
                try:
                    emb = self._st_model.encode([a, b])
                    a_vec, b_vec = emb[0], emb[1]
                    dot = float(sum(x*y for x, y in zip(a_vec, b_vec)))
                    na = math.sqrt(sum(x*x for x in a_vec))
                    nb = math.sqrt(sum(x*x for x in b_vec))
                    return dot / (na * nb) if na and nb else 0.0
                except Exception:
                    return 0.0
            kwargs["custom_similarity"] = _sim
        return EnhancedCenteringTheory(self.nlp, **kwargs)

    def _sentence_similarity(self, text_a: str, text_b: str) -> float:
        """Sentence-to-sentence similarity, using transformers if available."""
        if self._st_model:
            try:
                emb = self._st_model.encode([text_a, text_b])
                a, b = emb[0], emb[1]
                return float(sum(x*y for x, y in zip(a, b)) / (
                    math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b))
                ))
            except Exception:
                pass  # fall through to spaCy similarity
        da = self.nlp(text_a)
        db = self.nlp(text_b)
        if da.vector_norm and db.vector_norm:
            return float(da.similarity(db))
        return 0.5

    # ------------------------------------------------------------------
    # Single text analysis
    # ------------------------------------------------------------------

    def analyze(self, text: str, include_clauses: bool = True) -> TextReport:
        """Full analysis of a single text."""
        doc = self.nlp(text)
        paragraphs = self._split_paragraphs(text, doc)
        ct = self._make_ct()

        all_sentences_flat = [s for p in paragraphs for s in p]
        if len(all_sentences_flat) < 2:
            return TextReport(
                text=text,
                sentence_count=len(all_sentences_flat),
                paragraph_count=len(paragraphs),
                word_count=len(text.split()),
                overall_cohesion=0.0,
                transition_distribution={},
                paragraphs=[],
                segments=[0] if all_sentences_flat else [],
                quality="insufficient_data",
                metadata={"model": self.model_name, "warning": "Need at least 2 sentences"},
            )

        paragraph_results: List[ParagraphAnalysis] = []
        all_sentences: List[SentenceAnalysis] = []
        all_transitions: Dict[TransitionType, int] = {}

        for p_idx, para_sents in enumerate(paragraphs):
            sent_analyses: List[SentenceAnalysis] = []
            clause_analyses: List[Dict[str, Any]] = []
            para_transitions: Dict[TransitionType, int] = {}

            for s_idx, sent_text in enumerate(para_sents):
                state = ct.update_discourse(sent_text)
                t = state.transition
                if t:
                    all_transitions[t] = all_transitions.get(t, 0) + 1
                    para_transitions[t] = para_transitions.get(t, 0) + 1

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

            para_total = sum(para_transitions.values()) or 1
            para_score = round(
                sum(para_transitions.get(t, 0) * ct._transition_weights.get(t, 0.5)
                    for t in TransitionType) / para_total, 4
            )
            para_dist = {t.value: para_transitions.get(t, 0) / para_total
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

        if overall_score >= QUALITY_HIGH:
            quality = "high"
        elif overall_score >= QUALITY_MEDIUM:
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
    # LLM Output Evaluation
    # ------------------------------------------------------------------

    def analyze_llm(
        self, response: str, prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze LLM-generated text cohesion. Optionally compare with prompt."""
        ct = self._make_ct()
        doc = self.nlp(response)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

        result: Dict[str, Any] = {
            "response_sentences": len(sentences),
            "response_words": len(response.split()),
        }

        transition_counts: Dict[TransitionType, int] = {}
        for sent in sentences:
            state = ct.update_discourse(sent)
            t = state.transition
            if t:
                transition_counts[t] = transition_counts.get(t, 0) + 1

        score, dist = ct._score_transitions(transition_counts)

        boundaries = list(ct.discourse_history)
        _b = [0]
        rough = 0
        for _i, _s in enumerate(boundaries):
            if _s.transition == TransitionType.ROUGH_SHIFT:
                rough += 1
                if _s.backward_center is None and rough >= 2:
                    _b.append(_i)
                    rough = 0
            else:
                rough = max(0, rough - 1)

        result["response_cohesion"] = round(score, 4)
        result["response_transitions"] = dist
        result["response_segments"] = len(_b)

        if score >= QUALITY_HIGH:
            result["quality"] = "high"
        elif score >= QUALITY_MEDIUM:
            result["quality"] = "medium"
        else:
            result["quality"] = "low"

        if prompt:
            prompt_doc = self.nlp(prompt)
            prompt_sents = [s.text.strip() for s in prompt_doc.sents if s.text.strip()]
            ct2 = self._make_ct()
            prompt_counts: Dict[TransitionType, int] = {}
            for sent in prompt_sents:
                state = ct2.update_discourse(sent)
                t = state.transition
                if t:
                    prompt_counts[t] = prompt_counts.get(t, 0) + 1
            prompt_score, _ = ct2._score_transitions(prompt_counts)
            result["prompt_cohesion"] = round(prompt_score, 4)
            result["prompt_sentences"] = len(prompt_sents)

            ct3 = self._make_ct()
            combined = prompt_sents + sentences
            result["cross_boundary_penalty"] = 0
            for sent in combined:
                ct3.update_discourse(sent)
            for i in range(len(prompt_sents), len(combined)):
                if (ct3.discourse_history[i].transition == TransitionType.ROUGH_SHIFT
                        and ct3.discourse_history[i].backward_center is None):
                    result["cross_boundary_penalty"] += 1

        return result

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

        resolved_roles: List[Dict[str, str]] = []

        for sent_text in sentences_str:
            ct.update_discourse(sent_text)

        for sent_idx, sent_span in enumerate(doc.sents):
            sent_text = sent_span.text.strip()
            if not sent_text:
                continue
            state = ct.discourse_history[sent_idx] if sent_idx < len(ct.discourse_history) else None
            roles: Dict[str, str] = {}
            seen: set = set()

            for token in sent_span:
                if token.pos_ not in ("NOUN", "PROPN", "PRON"):
                    continue
                key = token.text.lower()
                if key in seen:
                    continue
                seen.add(key)

                entity_key = key
                if token.pos_ == "PRON" and state and state.backward_center:
                    entity_key = state.backward_center

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
            return EntityGrid(entities=entity_list, matrix=matrix, score=1.0)

        # cohesion: count entity persistence vs disappearance across adjacent sentences
        persist = 0
        disrupt = 0
        for col in range(n_entities):
            for row in range(1, len(matrix)):
                prev = matrix[row - 1][col]
                curr = matrix[row][col]
                if prev != "-" and curr != "-":
                    persist += 1
                elif prev != "-" and curr == "-":
                    disrupt += 1

        total = persist + disrupt
        score = round(persist / max(total, 1), 4) if total > 0 else 1.0

        return EntityGrid(entities=entity_list, matrix=matrix, score=score)

    # ------------------------------------------------------------------
    # TextTiling Segmentation (Hearst 1994/1997)
    # ------------------------------------------------------------------

    def texttile_segments(
        self, text: str, k: int = 10, cutoff: float = 0.2,
    ) -> List[int]:
        """
        TextTiling segmentation — returns sentence indices of boundaries.
        k = smoothing window, cutoff = relative depth threshold (0-1).
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
        for i in range(len(sents) - 1):
            sims.append(self._sentence_similarity(sents[i].text, sents[i + 1].text))

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
                vec_sim = self._sentence_similarity(sents[i], sents[j])

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
                    if adj[v][u] > GRAPH_EDGE_THRESHOLD and not visited[u]:
                        stack.append(u)
            communities.append(sorted(comp))

        # centrality: degree
        degrees = [sum(1 for j in range(n) if j != i and adj[i][j] > GRAPH_EDGE_THRESHOLD) for i in range(n)]
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
        """Score lexical cohesion via noun overlap across adjacent sentences."""
        doc = self.nlp(text)
        sents = [s for s in doc.sents if s.text.strip()]
        if len(sents) < 2:
            return 1.0

        # extract nouns per sentence (use doc directly, no re-parse)
        sent_nouns: List[List[str]] = []
        for sent in sents:
            nouns = [t.text.lower() for t in sent
                    if t.pos_ in ("NOUN", "PROPN") and not t.is_stop]
            sent_nouns.append(nouns)

        chain_lengths: List[int] = []
        current_chain = 0
        for i in range(1, len(sent_nouns)):
            prev_set = set(sent_nouns[i - 1])
            curr_set = set(sent_nouns[i])
            if prev_set & curr_set:
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
    # P3: Cohesion Trend (sliding window)
    # ------------------------------------------------------------------

    def cohesion_trend(
        self, text: str, window: int = 3,
    ) -> Dict[str, Any]:
        """Track cohesion across text using sliding window."""
        doc = self.nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        n = len(sents)

        if n < window:
            return {"windows": [], "mean": 1.0, "min": 1.0, "trend": "flat"}

        ct = self._make_ct()
        scores: List[float] = []
        for i in range(n - window + 1):
            chunk = sents[i:i + window]
            ct.discourse_history = []
            for sent in chunk:
                ct.update_discourse(sent)
            # compute score from accumulated transitions (avoid double-parse)
            counts: Dict = {}
            for st in ct.discourse_history:
                t = st.transition
                if t:
                    counts[t] = counts.get(t, 0) + 1
            s, _ = ct._score_transitions(counts)
            scores.append(s)

        mean_score = round(sum(scores) / len(scores), 4)
        min_score = round(min(scores), 4)
        min_idx = scores.index(min_score)

        if len(scores) >= 2:
            slope = (scores[-1] - scores[0]) / len(scores)
            if slope > TREND_SLOPE_THRESHOLD:
                trend = "improving"
            elif slope < -TREND_SLOPE_THRESHOLD:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "windows": [
                {"start": i, "end": i + window - 1, "score": round(s, 4)}
                for i, s in enumerate(scores)
            ],
            "mean": mean_score,
            "min": min_score,
            "min_window_start": min_idx,
            "trend": trend,
        }

    # ------------------------------------------------------------------
    # P4: Cohesion Heatmap
    # ------------------------------------------------------------------

    def cohesion_heatmap(
        self, text: str, ascii_render: bool = True,
    ) -> Dict[str, Any]:
        """NxN sentence similarity matrix. Weak pairs flagged."""
        doc = self.nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        n = len(sents)

        if n < 2:
            return {"matrix": [[1.0]], "weak_pairs": [], "ascii": ""}

        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._sentence_similarity(sents[i], sents[j])
                matrix[i][j] = round(sim, 4)
                matrix[j][i] = round(sim, 4)

        weak_pairs = []
        for i in range(n - 1):
            if matrix[i][i + 1] < HEATMAP_WEAK:
                weak_pairs.append({
                    "sentence_a": i,
                    "sentence_b": i + 1,
                    "similarity": matrix[i][i + 1],
                })

        ascii_out = ""
        if ascii_render:
            chars = [" ", ".", ":", "#"]
            lines: List[str] = []
            for i in range(n):
                row_chars = "".join(
                    chars[min(3, int(matrix[i][j] * 4))] for j in range(n)
                )
                lines.append(f"  [{i:02d}] {row_chars}")
            ascii_out = "\n".join(lines)

        return {
            "matrix": matrix,
            "weak_pairs": weak_pairs,
            "weak_count": len(weak_pairs),
            "ascii": ascii_out,
        }

    # ------------------------------------------------------------------
    # P5: Readability Integration
    # ------------------------------------------------------------------

    def readability_score(self, text: str) -> Dict[str, float]:
        """Flesch Reading Ease + basic text statistics (pure Python)."""
        words = text.split()
        sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".")
                     if s.strip()]
        n_words = len(words) or 1
        n_sents = len(sentences) or 1

        # syllable count (simple heuristic: count vowel groups)
        syllables = 0
        for word in words:
            word = word.lower().strip(".,;:!?\"'()[]{}")
            if not word:
                continue
            count = 0
            prev_vowel = False
            for ch in word:
                is_vowel = ch in "aeiouy"
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            syllables += max(count, 1)

        # Flesch Reading Ease
        flesch = 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (syllables / n_words)
        flesch = round(max(0, min(120, flesch)), 1)

        avg_sent_len = round(n_words / n_sents, 1)
        avg_word_len = round(sum(len(w) for w in words) / n_words, 1)

        return {
            "flesch_reading_ease": flesch,
            "avg_sentence_length": avg_sent_len,
            "avg_word_length": avg_word_len,
            "words": n_words,
            "sentences": n_sents,
        }

    def combined_score(self, text: str) -> float:
        """Cohesion + readability combined quality score."""
        r = self.analyze(text)
        readability = self.readability_score(text)
        read_norm = min(readability["flesch_reading_ease"] / 100.0, 1.0)
        return round(r.overall_cohesion * 0.6 + read_norm * 0.4, 4)

    # ------------------------------------------------------------------
    # P6: Improvement Suggestions
    # ------------------------------------------------------------------

    def suggest_improvements(self, text: str) -> List[Dict[str, Any]]:
        """Detect weak cohesion points and suggest fixes."""
        doc = self.nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        ct = self._make_ct()

        suggestions: List[Dict[str, Any]] = []
        rough_count = 0

        for i, sent in enumerate(sents):
            state = ct.update_discourse(sent)
            t = state.transition
            if t is None:
                continue

            if t.value == "Rough-Shift":
                rough_count += 1
                if state.backward_center is None:
                    suggestions.append({
                        "index": i,
                        "issue": "no_backward_center",
                        "severity": "high",
                    "suggestion": (
                        "No link to previous topic. Add a pronoun, repeat a key word, "
                        "or use a transition phrase (however, therefore, in addition)."
                    ),
                    })
                else:
                    suggestions.append({
                        "index": i,
                        "issue": "rough_shift",
                        "severity": "medium",
                        "suggestion": "Topic shift detected. Consider adding a transition phrase.",
                    })

        # consecutive rough shifts warning
        if rough_count >= 3:
            suggestions.append({
                "index": -1,
                "issue": "consecutive_rough_shifts",
                "severity": "high",
                "suggestion": f"{rough_count} consecutive topic shifts. This section may need restructuring.",
            })

        return suggestions

    def annotate_weak_points(self, text: str) -> str:
        """Insert <<<WEAK>>> markers at low-cohesion boundaries."""
        suggestions = self.suggest_improvements(text)
        doc = self.nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        weak_indices = {s["index"] for s in suggestions if s["index"] >= 0}

        result = []
        for i, sent in enumerate(sents):
            marker = " <<<WEAK>>>" if i in weak_indices else ""
            result.append(f"[{i}] {sent}{marker}")

        return "\n".join(result)

    # ------------------------------------------------------------------
    # P7: Diff Analysis
    # ------------------------------------------------------------------

    def diff_cohesion(
        self, original: str, revised: str,
    ) -> Dict[str, Any]:
        """Compare cohesion of two text versions."""
        r1 = self.analyze(original)
        r2 = self.analyze(revised)

        delta = round(r2.overall_cohesion - r1.overall_cohesion, 4)

        def _count(t_dist, name):
            return t_dist.get(name, 0)

        continue_delta = round(
            _count(r2.transition_distribution, "Continue")
            - _count(r1.transition_distribution, "Continue"), 3
        )
        rough_delta = round(
            _count(r2.transition_distribution, "Rough-Shift")
            - _count(r1.transition_distribution, "Rough-Shift"), 3
        )

        if delta > DIFF_DELTA_THRESHOLD:
            verdict = "improved"
        elif delta < -DIFF_DELTA_THRESHOLD:
            verdict = "declined"
        else:
            verdict = "similar"

        return {
            "original_score": r1.overall_cohesion,
            "revised_score": r2.overall_cohesion,
            "delta": delta,
            "verdict": verdict,
            "continue_delta": continue_delta,
            "rough_shift_delta": rough_delta,
            "original_segments": len(r1.segments),
            "revised_segments": len(r2.segments),
            "original_quality": r1.quality,
            "revised_quality": r2.quality,
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
