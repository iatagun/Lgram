"""
Centering Theory Implementation (Grosz, Joshi, Weinstein 1983/1995)

Provides discourse cohesion analysis through center computation,
transition classification, and cohesion scoring.

Note: save()/load() use pickle for serialization. Only load files
from trusted sources, as pickle can execute arbitrary code.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import spacy

logger = logging.getLogger(__name__)


class TransitionType(Enum):
    ESTABLISH = "Establish"        # No previous utterance (first in discourse)
    CONTINUE = "Continue"          # Cb(Ui) = Cb(Ui-1) = Cp(Ui)
    RETAIN = "Retain"              # Cb(Ui) = Cb(Ui-1) != Cp(Ui)
    SMOOTH_SHIFT = "Smooth-Shift"  # Cb(Ui) != Cb(Ui-1) = Cp(Ui)
    ROUGH_SHIFT = "Rough-Shift"    # Cb(Ui) != Cb(Ui-1) != Cp(Ui)


@dataclass
class DiscourseEntity:
    text: str
    dep: str
    pos: str
    ent_type: str = ""
    tag: str = ""
    is_plural: bool = False
    is_person: bool = False


@dataclass
class CenteringState:
    utterance: str
    forward_centers: List[str] = field(default_factory=list)
    backward_center: Optional[str] = None
    preferred_center: Optional[str] = None
    transition: Optional[TransitionType] = None
    _entity_map: Dict[str, DiscourseEntity] = field(default_factory=dict, repr=False)


class EnhancedCenteringTheory:

    def __init__(
        self,
        nlp_model: spacy.language.Language,
        history_limit: int = 10,
        salience_weights: Optional[Dict[str, float]] = None,
        pos_weights: Optional[Dict[str, float]] = None,
    ):
        self.nlp = nlp_model
        self.discourse_history: List[CenteringState] = []
        self.history_limit = history_limit

        self.salience_weights = salience_weights or {
            "nsubj": 4.0,
            "nsubjpass": 4.0,
            "dobj": 3.0,
            "iobj": 2.0,
            "pobj": 2.0,
            "attr": 2.0,
            "oprd": 2.0,
            "poss": 1.0,
            "appos": 1.0,
        }

        self.pos_weights = pos_weights or {
            "PRON": 3.0,
            "PROPN": 2.0,
            "NOUN": 1.0,
        }

    # ------------------------------------------------------------------
    # Core – center computation
    # ------------------------------------------------------------------

    def compute_forward_centers(self, utterance: str) -> Tuple[List[str], Dict[str, DiscourseEntity]]:
        doc = self.nlp(utterance)
        token_count = len(doc)
        centers: List[Tuple[str, float, int]] = []
        entity_map: Dict[str, DiscourseEntity] = {}

        for token in doc:
            if not self._is_center_candidate(token):
                continue

            key = token.text.lower()
            if key not in entity_map:
                entity_map[key] = DiscourseEntity(
                    text=token.text,
                    dep=token.dep_,
                    pos=token.pos_,
                    ent_type=token.ent_type_,
                    tag=token.tag_,
                    is_plural=token.tag_ in ("NNS", "NNPS"),
                    is_person=token.ent_type_ == "PERSON",
                )

            salience = self._calculate_salience(token, token_count)
            centers.append((key, salience, token.i))

        centers.sort(key=lambda x: (-x[1], x[2]))

        seen: set = set()
        unique: List[str] = []
        for c, _, _ in centers:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique[:5], entity_map

    def _is_center_candidate(self, token) -> bool:
        if token.pos_ not in ("NOUN", "PROPN", "PRON"):
            return False
        if token.is_punct:
            return False
        if token.dep_ in ("det", "aux", "auxpass", "cop"):
            return False
        if token.is_stop and token.pos_ != "PRON":
            return False

        if token.pos_ == "PRON":
            _allowed = {"he", "she", "it", "i", "we", "they", "you",
                        "him", "her", "them", "us", "me"}
            if token.text.lower() in _allowed:
                return True
            return False
        else:
            if len(token.text) <= 1:
                return False

        return True

    def _calculate_salience(self, token, token_count: int) -> float:
        score = self.salience_weights.get(token.dep_, 0.0)
        score += self.pos_weights.get(token.pos_, 0.0)
        score += 1.0 - (token.i / max(token_count, 1))
        if token.ent_type_ in ("PERSON", "ORG", "GPE"):
            score += 1.5
        if token.pos_ == "PRON" and self._has_clear_antecedent(token):
            score += 1.0
        return score

    def _has_clear_antecedent(self, pronoun_token) -> bool:
        if not self.discourse_history:
            return False
        pronoun = pronoun_token.text.lower()
        for state in self.discourse_history[-3:]:
            for center in state.forward_centers:
                ent = state._entity_map.get(center)
                if ent is None:
                    continue
                if pronoun in self._person_pronouns and ent.is_person:
                    return True
                if pronoun in self._object_pronouns and not ent.is_person:
                    return True
                if pronoun in self._plural_pronouns and ent.is_plural:
                    return True
        return False

    # ------------------------------------------------------------------
    # Backward center (Cb)
    # ------------------------------------------------------------------

    _person_pronouns = frozenset({"he", "him", "his", "she", "her", "hers"})
    _object_pronouns = frozenset({"it", "its"})
    _plural_pronouns = frozenset({"they", "them", "their", "theirs"})
    _all_pronouns: frozenset = _person_pronouns | _object_pronouns | _plural_pronouns

    def compute_backward_center(
        self, current_cf: List[str], current_entity_map: Dict[str, DiscourseEntity],
        current_utterance: Optional[str] = None,
    ) -> Optional[str]:
        if not self.discourse_history or not current_cf:
            return None

        prev_state = self.discourse_history[-1]
        prev_cf = prev_state.forward_centers
        prev_map = prev_state._entity_map

        # possessive pronouns referring to person entities take priority
        if current_utterance:
            doc = self.nlp(current_utterance)
            for token in doc:
                if token.pos_ == "PRON":
                    token_lower = token.text.lower()
                    if token_lower in self._person_pronouns | self._object_pronouns:
                        for pc in prev_cf:
                            if self._pronoun_matches_entity(token_lower, pc, prev_map):
                                ent = prev_map.get(pc)
                                if ent and ent.is_person:
                                    return pc

        for pc in prev_cf:
            if pc in current_cf:
                return pc

        for cc in current_cf:
            cc_lower = cc.lower()
            if cc_lower in self._all_pronouns:
                for pc in prev_cf:
                    if self._pronoun_matches_entity(cc_lower, pc, prev_map):
                        return pc

        for pc in prev_cf:
            for cc in current_cf:
                if self._are_coreferent_cached(pc, cc, prev_map, current_entity_map):
                    return pc

        for cc in current_cf:
            cc_lower = cc.lower()
            if cc_lower in self._plural_pronouns:
                persons = [
                    pc for pc in prev_cf
                    if prev_map.get(pc) and prev_map[pc].is_person
                ]
                if len(persons) >= 2:
                    return persons[0]

        return None

    def _pronoun_matches_entity(
        self, pronoun: str, entity_key: str, prev_map: Dict[str, DiscourseEntity]
    ) -> bool:
        ent = prev_map.get(entity_key)
        if ent is None:
            return False

        entity_text_lower = ent.text.lower()

        if pronoun in self._person_pronouns:
            if ent.is_person:
                return True
            if entity_text_lower in self._person_pronouns:
                return True

        if pronoun in self._object_pronouns:
            if not ent.is_person and entity_text_lower not in self._person_pronouns:
                return True

        if pronoun in self._plural_pronouns:
            if ent.is_plural:
                return True
            if entity_text_lower in self._plural_pronouns:
                return True

        return False

    def _find_in_history(self, key: str) -> Optional[DiscourseEntity]:
        for state in reversed(self.discourse_history):
            ent = state._entity_map.get(key)
            if ent is not None:
                return ent
        return None

    def _resolve_entity(
        self, key: str, primary: Dict[str, DiscourseEntity]
    ) -> Optional[DiscourseEntity]:
        ent = primary.get(key)
        if ent is not None:
            return ent
        return self._find_in_history(key)

    def _are_coreferent_cached(
        self,
        e1: str, e2: str,
        map1: Dict[str, DiscourseEntity],
        map2: Dict[str, DiscourseEntity],
    ) -> bool:
        if e1.lower() == e2.lower():
            return True
        ent1 = self._resolve_entity(e1, map1)
        ent2 = self._resolve_entity(e2, map2)
        if ent1 is None or ent2 is None:
            return False
        e2_lower = e2.lower()
        e1_lower = e1.lower()
        if ent1.is_person and e2_lower in self._person_pronouns:
            return True
        if ent2.is_person and e1_lower in self._person_pronouns:
            return True
        if not ent1.is_person and e2_lower in self._object_pronouns:
            return True
        if not ent2.is_person and e1_lower in self._object_pronouns:
            return True
        if ent1.is_plural and e2_lower in self._plural_pronouns:
            return True
        if ent2.is_plural and e1_lower in self._plural_pronouns:
            return True
        return False

    # ------------------------------------------------------------------
    # Transition classification
    # ------------------------------------------------------------------

    def determine_transition(self, current_state: CenteringState) -> TransitionType:
        # determine_transition is called before current_state is in history.
        # discourse_history[-1] is the previous utterance.
        if not self.discourse_history:
            return TransitionType.ESTABLISH

        prev_state = self.discourse_history[-1]
        cb = current_state.backward_center
        prev_cb = prev_state.backward_center
        cp = current_state.preferred_center

        if cb is None and prev_cb is None:
            if cp and prev_state.preferred_center:
                if self._are_coreferent_cached(
                    cp, prev_state.preferred_center,
                    current_state._entity_map, prev_state._entity_map,
                ):
                    return TransitionType.CONTINUE
                return TransitionType.ROUGH_SHIFT
            return TransitionType.RETAIN

        if cb is None and prev_cb is not None:
            return TransitionType.ROUGH_SHIFT

        if cb is not None and prev_cb is None:
            # First transition: previous Cb undefined.
            # Cb comes from prev Cf, cp from current Cf.
            # CONTINUE if they refer to the same entity.
            if cp and self._are_coreferent_cached(
                cb, cp, prev_state._entity_map, current_state._entity_map,
            ):
                return TransitionType.CONTINUE
            return TransitionType.RETAIN

        cb_same = self._are_coreferent_cached(
            cb, prev_cb, prev_state._entity_map, prev_state._entity_map
        )
        cp_same_as_cb = self._are_coreferent_cached(
            cb, cp, prev_state._entity_map, current_state._entity_map
        )

        if cb_same:
            return TransitionType.CONTINUE if cp_same_as_cb else TransitionType.RETAIN
        else:
            if prev_cb and cp:
                if self._are_coreferent_cached(
                    prev_cb, cp, prev_state._entity_map, current_state._entity_map
                ):
                    return TransitionType.SMOOTH_SHIFT
            return TransitionType.ROUGH_SHIFT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_state(self, utterance: str) -> CenteringState:
        cf, entity_map = self.compute_forward_centers(utterance)
        cb = self.compute_backward_center(cf, entity_map, utterance)
        cp = cf[0] if cf else None
        return CenteringState(
            utterance=utterance,
            forward_centers=cf,
            backward_center=cb,
            preferred_center=cp,
            _entity_map=entity_map,
        )

    def analyze_utterance(self, utterance: str) -> CenteringState:
        state = self._build_state(utterance)
        state.transition = self.determine_transition(state)
        return state

    def update_discourse(self, utterance: str) -> CenteringState:
        state = self._build_state(utterance)
        state.transition = self.determine_transition(state)
        self.discourse_history.append(state)
        if len(self.discourse_history) > self.history_limit:
            self.discourse_history = self.discourse_history[-self.history_limit:]
        return state

    def get_coherent_next_center(self) -> Optional[str]:
        if not self.discourse_history:
            return None
        last = self.discourse_history[-1]
        if last.backward_center:
            return last.backward_center
        if last.forward_centers:
            return last.forward_centers[0]
        return None

    def evaluate_cohesion(self, utterance_sequence: List[str]) -> Dict[str, Any]:
        saved = list(self.discourse_history)
        self.discourse_history = []

        transition_counts: Dict[TransitionType, int] = {}
        for u in utterance_sequence:
            state = self.update_discourse(u)
            if state.transition:
                transition_counts[state.transition] = \
                    transition_counts.get(state.transition, 0) + 1

        self.discourse_history = saved

        score, dist = self._score_transitions(transition_counts)
        return {
            "cohesion_score": score,
            "transition_distribution": dist,
            "total_transitions": sum(transition_counts.values()),
        }

    def evaluate_coherence(self, utterance_sequence: List[str]) -> Dict[str, Any]:
        """Deprecated: use evaluate_cohesion instead."""
        return self.evaluate_cohesion(utterance_sequence)

    def get_discourse_summary(self) -> Dict[str, Any]:
        if not self.discourse_history:
            return {"message": "No discourse history"}

        recent_centers: List[str] = []
        recent_transitions: List[str] = []
        for s in self.discourse_history[-5:]:
            recent_centers.extend(s.forward_centers[:2])
            if s.transition:
                recent_transitions.append(s.transition.value)

        return {
            "recent_centers": list(dict.fromkeys(recent_centers)),
            "recent_transitions": recent_transitions,
            "current_cb": self.discourse_history[-1].backward_center,
            "current_cp": self.discourse_history[-1].preferred_center,
            "discourse_length": len(self.discourse_history),
        }

    # transit weights: CONTINUE=1.0, RETAIN=0.8, SMOOTH=0.6, ROUGH=0.3, ESTABLISH=1.0
    _transition_weights = {
        TransitionType.CONTINUE: 1.0,
        TransitionType.RETAIN: 0.8,
        TransitionType.SMOOTH_SHIFT: 0.6,
        TransitionType.ROUGH_SHIFT: 0.3,
        TransitionType.ESTABLISH: 1.0,
    }

    @classmethod
    def _score_transitions(
        cls, transition_counts: Dict[TransitionType, int]
    ) -> Tuple[float, Dict[str, float]]:
        total = sum(transition_counts.values())
        if total == 0:
            return 1.0, {}
        weighted = sum(
            transition_counts.get(t, 0) * cls._transition_weights.get(t, 0.5)
            for t in TransitionType
        )
        dist = {t.value: transition_counts.get(t, 0) / total for t in TransitionType}
        return round(weighted / total, 4), dist

    # ------------------------------------------------------------------
    # Intra-sentential (clause-level) analysis
    # ------------------------------------------------------------------

    _clause_verb_deps = frozenset({"ROOT", "conj", "advcl", "ccomp", "acl", "relcl"})
    _clause_separator_tags = frozenset({"CCONJ", "PUNCT"})

    def extract_clauses(self, sentence: str) -> List[Tuple[str, str]]:
        """
        Split a complex sentence into clauses using dependency parse.
        Returns list of (clause_text, clause_type) in original text order.
        """
        doc = self.nlp(sentence)
        claimed: set = set()
        sub_clauses: List[Tuple[int, int, str]] = []

        for token in doc:
            if token.pos_ not in ("VERB", "AUX"):
                continue
            dep = token.dep_
            if dep not in self._clause_verb_deps or dep == "ROOT":
                continue

            start = token.left_edge.i
            end = token.right_edge.i + 1

            # extend left to include preceding separator tokens
            while start > 0:
                prev = doc[start - 1]
                if prev.pos_ in self._clause_separator_tags:
                    start -= 1
                else:
                    break

            claimed.update(range(start, end))
            sub_clauses.append((start, end, dep))

        main_tokens = [t.i for t in doc if t.i not in claimed]

        all_ranges: List[Tuple[int, int, str]] = []
        if main_tokens:
            all_ranges.append((min(main_tokens), max(main_tokens) + 1, "main"))
        all_ranges.extend(sub_clauses)
        all_ranges.sort(key=lambda x: x[0])

        result: List[Tuple[str, str]] = []
        for start, end, ctype in all_ranges:
            if start >= end:
                continue
            text = doc[start:end].text.strip().rstrip(",;").strip()
            if text:
                result.append((text, ctype))

        if not result:
            result.append((sentence.strip().rstrip(".").strip(), "main"))
        return result

    def analyze_intra_sentential(self, sentence: str) -> Dict[str, Any]:
        """Analyze clause-level cohesion within a single complex sentence."""
        clauses = self.extract_clauses(sentence)
        if len(clauses) < 2:
            return {
                "sentence": sentence,
                "clause_count": len(clauses),
                "transitions": [],
                "cohesion_score": 1.0,
            }

        saved = list(self.discourse_history)
        self.discourse_history = []

        transition_counts: Dict[TransitionType, int] = {}
        transitions: List[Dict[str, Any]] = []
        for clause_text, clause_type in clauses:
            state = self.update_discourse(clause_text)
            t = state.transition
            if t:
                transition_counts[t] = transition_counts.get(t, 0) + 1
                transitions.append({
                    "clause": clause_text,
                    "type": clause_type,
                    "transition": t.value,
                    "cp": state.preferred_center,
                    "cb": state.backward_center,
                })

        score, dist = self._score_transitions(transition_counts)

        self.discourse_history = saved

        return {
            "sentence": sentence,
            "clause_count": len(clauses),
            "transitions": transitions,
            "cohesion_score": round(score, 4),
            "transition_distribution": dist,
        }

    def analyze_full(self, text: str) -> Dict[str, Any]:
        """Analyze both inter-sentential and intra-sentential cohesion."""
        doc = self.nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

        intra_results: List[Dict[str, Any]] = []
        for sent in sentences:
            intra = self.analyze_intra_sentential(sent)
            intra_results.append(intra)

        saved = list(self.discourse_history)
        self.discourse_history = []
        for sent in sentences:
            self.update_discourse(sent)
        inter_result = self.evaluate_cohesion(sentences)
        self.discourse_history = saved

        return {
            "sentence_count": len(sentences),
            "inter_sentential": inter_result,
            "intra_sentential": intra_results,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "history": self.discourse_history,
                "salience_weights": self.salience_weights,
                "pos_weights": self.pos_weights,
                "history_limit": self.history_limit,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.discourse_history = data["history"]
        self.salience_weights = data.get("salience_weights", self.salience_weights)
        self.pos_weights = data.get("pos_weights", self.pos_weights)
        self.history_limit = data.get("history_limit", self.history_limit)
