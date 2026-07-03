"""
Discourse Relation Analysis — PDTB-lite approach.

Detects explicit discourse connectives and their argument spans using
spaCy dependency parsing. Computes discourse-level cohesion based on
connective usage patterns.

Connectives are organized by PDTB relation type:
- COMPARISON: but, however, although, whereas, while, yet, in contrast, ...
- CONTINGENCY: because, therefore, thus, so, if, since, unless, ...
- EXPANSION: and, also, moreover, furthermore, in addition, for example, ...
- TEMPORAL: then, after, before, when, while, until, meanwhile, ...
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


_CONNECTIVE_MAP: Dict[str, str] = {
    "but": "COMPARISON",
    "however": "COMPARISON",
    "although": "COMPARISON",
    "though": "COMPARISON",
    "whereas": "COMPARISON",
    "while": "COMPARISON",
    "yet": "COMPARISON",
    "nevertheless": "COMPARISON",
    "nonetheless": "COMPARISON",
    "instead": "COMPARISON",
    "rather": "COMPARISON",
    "conversely": "COMPARISON",
    "otherwise": "COMPARISON",
    "because": "CONTINGENCY",
    "therefore": "CONTINGENCY",
    "thus": "CONTINGENCY",
    "hence": "CONTINGENCY",
    "so": "CONTINGENCY",
    "consequently": "CONTINGENCY",
    "accordingly": "CONTINGENCY",
    "if": "CONTINGENCY",
    "since": "CONTINGENCY",
    "unless": "CONTINGENCY",
    "provided": "CONTINGENCY",
    "as a result": "CONTINGENCY",
    "for this reason": "CONTINGENCY",
    "and": "EXPANSION",
    "also": "EXPANSION",
    "moreover": "EXPANSION",
    "furthermore": "EXPANSION",
    "additionally": "EXPANSION",
    "besides": "EXPANSION",
    "indeed": "EXPANSION",
    "in addition": "EXPANSION",
    "for example": "EXPANSION",
    "for instance": "EXPANSION",
    "specifically": "EXPANSION",
    "in particular": "EXPANSION",
    "in fact": "EXPANSION",
    "then": "TEMPORAL",
    "after": "TEMPORAL",
    "before": "TEMPORAL",
    "when": "TEMPORAL",
    "until": "TEMPORAL",
    "meanwhile": "TEMPORAL",
    "subsequently": "TEMPORAL",
    "previously": "TEMPORAL",
    "afterward": "TEMPORAL",
    "eventually": "TEMPORAL",
    "finally": "TEMPORAL",
    "initially": "TEMPORAL",
    "later": "TEMPORAL",
    "earlier": "TEMPORAL",
}


@dataclass
class DiscourseRelation:
    connective: str
    relation_type: str
    source_sentence: int
    target_sentence: int
    confidence: float = 1.0


@dataclass
class DiscourseReport:
    sentence_count: int
    relations: List[DiscourseRelation]
    relation_distribution: Dict[str, int]
    relation_density: float
    dominant_relation: str
    cohesion_score: float
    connective_count: int


class DiscourseAnalyzer:
    """Analyze discourse relations via explicit connectives."""

    def __init__(self, nlp: Optional[Any] = None):
        self._nlp = nlp

    def _ensure_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")

    def extract_relations(self, text: str) -> List[DiscourseRelation]:
        """Extract explicit discourse relations between sentences."""
        self._ensure_nlp()
        doc = self._nlp(text)
        sents = list(doc.sents)
        relations: List[DiscourseRelation] = []

        if len(sents) < 2:
            return relations

        for i, sent in enumerate(sents):
            sent_start = sent.start
            for token in sent:
                token_lower = token.text.lower()
                if token_lower in _CONNECTIVE_MAP:
                    rel_type = _CONNECTIVE_MAP[token_lower]
                    source = i
                    target = i
                    if token.dep_ in ("cc", "mark"):
                        target = i
                    relations.append(DiscourseRelation(
                        connective=token_lower,
                        relation_type=rel_type,
                        source_sentence=source,
                        target_sentence=target,
                    ))

        for i in range(len(sents)):
            text_lower = sents[i].text.lower()
            for phrase in ("as a result", "for this reason", "in addition",
                          "for example", "for instance", "in particular",
                          "in fact", "in contrast", "on the other hand"):
                if phrase in text_lower:
                    rel_type = _CONNECTIVE_MAP.get(phrase, "EXPANSION")
                    relations.append(DiscourseRelation(
                        connective=phrase,
                        relation_type=rel_type,
                        source_sentence=i,
                        target_sentence=i,
                    ))

        return relations

    def analyze(self, text: str) -> DiscourseReport:
        """Full discourse analysis of a text."""
        self._ensure_nlp()
        doc = self._nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]

        relations = self.extract_relations(text)
        rel_counts: Dict[str, int] = {}
        for r in relations:
            rel_counts[r.relation_type] = rel_counts.get(r.relation_type, 0) + 1

        density = len(relations) / max(len(sents) - 1, 1)
        connective_count = len(set(r.connective for r in relations))

        if rel_counts:
            dominant = max(rel_counts, key=lambda k: rel_counts[k])
        else:
            dominant = "none"

        relation_score = min(density / 1.5, 0.4)
        variety_bonus = min(connective_count / 5.0, 0.3)
        distribution_penalty = 0.0
        if rel_counts:
            total = sum(rel_counts.values())
            max_pct = max(rel_counts.values()) / total
            if max_pct > 0.7:
                distribution_penalty = (max_pct - 0.7) * 0.3

        cohesion_score = round(min(relation_score + variety_bonus - distribution_penalty, 1.0), 4)

        return DiscourseReport(
            sentence_count=len(sents),
            relations=relations,
            relation_distribution=rel_counts,
            relation_density=round(density, 4),
            dominant_relation=dominant,
            cohesion_score=max(cohesion_score, 0.0),
            connective_count=connective_count,
        )

    def compare_discourse(
        self, text_a: str, text_b: str,
    ) -> Dict[str, Any]:
        """Compare discourse structure of two texts."""
        r1 = self.analyze(text_a)
        r2 = self.analyze(text_b)

        delta_density = round(r2.relation_density - r1.relation_density, 4)
        delta_score = round(r2.cohesion_score - r1.cohesion_score, 4)

        if delta_score > 0.05:
            verdict = "improved"
        elif delta_score < -0.05:
            verdict = "declined"
        else:
            verdict = "similar"

        return {
            "text_a": {
                "density": r1.relation_density,
                "score": r1.cohesion_score,
                "dominant": r1.dominant_relation,
                "connectives": r1.connective_count,
            },
            "text_b": {
                "density": r2.relation_density,
                "score": r2.cohesion_score,
                "dominant": r2.dominant_relation,
                "connectives": r2.connective_count,
            },
            "delta_density": delta_density,
            "delta_score": delta_score,
            "verdict": verdict,
        }
