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
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    gender: str = ""  # "" | "male" | "female"
    is_female: bool = False


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
        similarity_threshold: float = 0.65,
        gender_map: Optional[Dict[str, str]] = None,
        custom_similarity: Optional[Callable[[str, str], float]] = None,
    ):
        self.nlp = nlp_model
        self.discourse_history: List[CenteringState] = []
        self.history_limit = history_limit
        self.similarity_threshold = similarity_threshold
        self._custom_similarity = custom_similarity

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

        self._gender_lookup = gender_map if gender_map is not None else dict(self._gender_map)

    def _resolve_gender(self, token) -> str:
        text = token.text.lower()
        gender = self._gender_lookup.get(text, "")

        # title-based detection
        if not gender:
            if text in self._titles_male:
                return "male"
            if text in self._titles_female:
                return "female"

        # suffix-based heuristics for unknown names (only for PROPN tokens)
        if not gender and token.pos_ == "PROPN" and len(text) > 3:
            for suffix in self._female_suffixes:
                if text.endswith(suffix) and text[-len(suffix)-1] not in "aeiou":
                    gender = "female"
                    break
            if not gender:
                for suffix in self._male_suffixes:
                    if text.endswith(suffix):
                        gender = "male"
                        break

        return gender

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
                g = self._resolve_gender(token)
                is_p = token.ent_type_ == "PERSON" or g in ("male", "female")
                entity_map[key] = DiscourseEntity(
                    text=token.text,
                    dep=token.dep_,
                    pos=token.pos_,
                    ent_type=token.ent_type_,
                    tag=token.tag_,
                    is_plural=token.tag_ in ("NNS", "NNPS"),
                    is_person=is_p,
                    gender=g,
                    is_female=g == "female",
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
        is_male = pronoun in self._male_pronouns
        is_female = pronoun in self._female_pronouns
        for state in self.discourse_history[-3:]:
            for center in state.forward_centers:
                ent = state._entity_map.get(center)
                if ent is None:
                    continue
                if is_male and (ent.gender == "male" or ent.is_person):
                    return True
                if is_female and (ent.is_female or ent.is_person):
                    return True
                if pronoun in self._object_pronouns and not ent.is_person:
                    return True
                if pronoun in self._plural_pronouns and ent.is_plural:
                    return True
        return False

    # ------------------------------------------------------------------
    # Backward center (Cb)
    # ------------------------------------------------------------------

    # pronoun sets with gender distinction
    _male_pronouns = frozenset({"he", "him", "his"})
    _female_pronouns = frozenset({"she", "her", "hers"})
    _person_pronouns = frozenset({"he", "him", "his", "she", "her", "hers"})
    _object_pronouns = frozenset({"it", "its"})
    _plural_pronouns = frozenset({"they", "them", "their", "theirs"})
    _all_pronouns: frozenset = _person_pronouns | _object_pronouns | _plural_pronouns

    _gender_map: Dict[str, str] = {
        # male
        "john": "male", "bob": "male", "dave": "male", "frank": "male",
        "mike": "male", "tom": "male", "jim": "male", "bill": "male",
        "steve": "male", "peter": "male", "paul": "male", "mark": "male",
        "james": "male", "robert": "male", "david": "male", "richard": "male",
        "charles": "male", "daniel": "male", "matthew": "male", "andrew": "male",
        "charlie": "male", "henry": "male", "george": "male", "jack": "male",
        "alex": "male", "sam": "male", "harry": "male", "ben": "male",
        "william": "male", "joseph": "male", "thomas": "male", "christopher": "male",
        "donald": "male", "ronald": "male", "steven": "male", "edward": "male",
        "brian": "male", "kevin": "male", "jason": "male", "jeff": "male",
        "ryan": "male", "eric": "male", "scott": "male", "gary": "male",
        "larry": "male", "patrick": "male", "adam": "male", "nathan": "male",
        "justin": "male", "brandon": "male", "aaron": "male", "nick": "male",
        "ethan": "male", "jacob": "male", "noah": "male", "liam": "male",
        "mason": "male", "lucas": "male", "logan": "male", "oliver": "male",
        "musa": "male", "mustafa": "male", "murat": "male", "ahmet": "male",
        "mehmet": "male", "ali": "male", "hasan": "male", "hüseyin": "male",
        "osman": "male", "orhan": "male", "kemal": "male", "can": "male",
        "emir": "male", "eren": "male", "barış": "male", "burak": "male",
        "ilker": "male",
        # female
        "alice": "female", "mary": "female", "sue": "female", "jane": "female",
        "sarah": "female", "emma": "female", "lisa": "female", "anna": "female",
        "laura": "female", "kate": "female", "sophie": "female", "lucy": "female",
        "emily": "female", "olivia": "female", "grace": "female", "diana": "female",
        "helen": "female", "martha": "female", "nancy": "female", "linda": "female",
        "barbara": "female", "elizabeth": "female", "margaret": "female",
        "susan": "female", "dorothy": "female", "betty": "female", "carol": "female",
        "jennifer": "female", "patricia": "female", "jessica": "female", "amanda": "female",
        "melissa": "female", "michelle": "female", "stephanie": "female", "rebecca": "female",
        "karen": "female", "amy": "female", "angela": "female", "sandra": "female",
        "rachel": "female", "katherine": "female", "nicole": "female", "christine": "female",
        "samantha": "female", "isabella": "female", "mia": "female", "charlotte": "female",
        "amelia": "female", "ava": "female", "harper": "female", "evelyn": "female",
        "fatma": "female", "ayşe": "female", "zeynep": "female", "elif": "female",
        "merve": "female", "büşra": "female", "esra": "female", "tuğba": "female",
        "gizem": "female", "eda": "female", "cansu": "female", "burcu": "female",
        "sibel": "female", "ayla": "female", "deniz": "female",
    }

    # name suffixes that strongly indicate gender
    _female_suffixes = frozenset({
        "a", "ia", "na", "ica", "ina", "ela", "ella", "ette", "ine", "lyn",
    })
    _male_suffixes = frozenset({
        "us", "er", "or", "an", "en", "on", "os",
    })

    _titles_male = frozenset({"mr", "sir", "lord", "king", "duke", "prince"})
    _titles_female = frozenset({"mrs", "ms", "miss", "lady", "queen", "duchess", "princess"})

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

        if pronoun in self._male_pronouns:
            if ent.is_female:
                return False
            if ent.gender == "male" or ent.is_person:
                return True

        elif pronoun in self._female_pronouns:
            if ent.gender == "male":
                return False
            if ent.is_female or ent.is_person:
                return True

        elif pronoun in self._object_pronouns:
            return not ent.is_person and entity_text_lower not in self._person_pronouns

        elif pronoun in self._plural_pronouns:
            if ent.is_plural:
                return True
            if entity_text_lower in self._plural_pronouns:
                return True

        # pronoun chain: match any person pronoun to any person pronoun
        if pronoun in self._person_pronouns:
            if entity_text_lower in self._person_pronouns:
                return True
            if ent.is_person:
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

    def _vector_similarity(self, e1: str, e2: str) -> float:
        if self._custom_similarity:
            try:
                return self._custom_similarity(e1, e2)
            except Exception:
                pass
        if not getattr(self.nlp.vocab, "vectors_length", 0):
            return 0.0
        try:
            d1 = self.nlp(e1)
            d2 = self.nlp(e2)
            if not d1.vector_norm or not d2.vector_norm:
                return 0.0
            return float(d1.similarity(d2))
        except Exception:
            return 0.0

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

        # gender-aware person pronoun matching
        if ent1.is_female and e2_lower in self._male_pronouns:
            return False
        if ent2.is_female and e1_lower in self._male_pronouns:
            return False
        if ent1.gender == "male" and e2_lower in self._female_pronouns:
            return False
        if ent2.gender == "male" and e1_lower in self._female_pronouns:
            return False

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

        # vector-based semantic similarity fallback
        if self.similarity_threshold > 0.0:
            sim = self._vector_similarity(e1, e2)
            if sim >= self.similarity_threshold:
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
        doc = self.nlp(utterance)
        cf, entity_map = self._compute_centers_from_doc(doc)
        cb = self._compute_backward_from_doc(cf, entity_map, doc)
        cp = cf[0] if cf else None
        return CenteringState(
            utterance=utterance,
            forward_centers=cf,
            backward_center=cb,
            preferred_center=cp,
            _entity_map=entity_map,
        )

    def _compute_centers_from_doc(
        self, doc
    ) -> Tuple[List[str], Dict[str, DiscourseEntity]]:
        token_count = len(doc)
        centers: List[Tuple[str, float, int]] = []
        entity_map: Dict[str, DiscourseEntity] = {}
        for token in doc:
            if not self._is_center_candidate(token):
                continue
            key = token.text.lower()
            if key not in entity_map:
                g = self._resolve_gender(token)
                is_p = token.ent_type_ == "PERSON" or g in ("male", "female")
                entity_map[key] = DiscourseEntity(
                    text=token.text, dep=token.dep_, pos=token.pos_,
                    ent_type=token.ent_type_, tag=token.tag_,
                    is_plural=token.tag_ in ("NNS", "NNPS"),
                    is_person=is_p, gender=g, is_female=g == "female",
                )
            salience = self._calculate_salience(token, token_count)
            centers.append((key, salience, token.i))
        centers.sort(key=lambda x: (-x[1], x[2]))
        seen, unique = set(), []
        for c, _, _ in centers:
            if c not in seen:
                seen.add(c); unique.append(c)
        return unique[:5], entity_map

    def _compute_backward_from_doc(
        self, current_cf: List[str], current_entity_map: Dict[str, DiscourseEntity], doc,
    ) -> Optional[str]:
        if not self.discourse_history or not current_cf:
            return None
        prev_state = self.discourse_history[-1]
        prev_cf = prev_state.forward_centers
        prev_map = prev_state._entity_map

        # possessive pronouns in the doc
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
                persons = [pc for pc in prev_cf if prev_map.get(pc) and prev_map[pc].is_person]
                if len(persons) >= 2:
                    return persons[0]
        return None

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

        try:
            transition_counts: Dict[TransitionType, int] = {}
            for u in utterance_sequence:
                state = self.update_discourse(u)
                if state.transition:
                    transition_counts[state.transition] = \
                        transition_counts.get(state.transition, 0) + 1

            score, dist = self._score_transitions(transition_counts)
            return {
                "cohesion_score": score,
                "transition_distribution": dist,
                "total_transitions": sum(transition_counts.values()),
            }
        finally:
            self.discourse_history = saved

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

    def analyze_llm(
        self, response: str, prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze LLM-generated text cohesion. Optionally compare with prompt."""
        doc = self.nlp(response)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

        result: Dict[str, Any] = {
            "response_sentences": len(sentences),
            "response_words": len(response.split()),
        }

        # response analysis
        saved = list(self.discourse_history)
        self.discourse_history = []

        try:
            transition_counts: Dict[TransitionType, int] = {}
            for sent in sentences:
                state = self.update_discourse(sent)
                t = state.transition
                if t:
                    transition_counts[t] = transition_counts.get(t, 0) + 1

            score, dist = self._score_transitions(transition_counts)
            boundaries = list(self.discourse_history)
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

            # quality rating
            if score >= 0.80:
                result["quality"] = "high"
            elif score >= 0.55:
                result["quality"] = "medium"
            else:
                result["quality"] = "low"
        finally:
            self.discourse_history = saved

        # prompt analysis (separate, no cross-contamination)
        if prompt:
            prompt_doc = self.nlp(prompt)
            prompt_sents = [s.text.strip() for s in prompt_doc.sents if s.text.strip()]
            self.discourse_history = []
            prompt_counts: Dict[TransitionType, int] = {}
            for sent in prompt_sents:
                state = self.update_discourse(sent)
                t = state.transition
                if t:
                    prompt_counts[t] = prompt_counts.get(t, 0) + 1
            prompt_score, _ = self._score_transitions(prompt_counts)
            result["prompt_cohesion"] = round(prompt_score, 4)
            result["prompt_sentences"] = len(prompt_sents)

            # gap analysis: does response maintain prompt's cohesion?
            combined = prompt_sents + sentences
            self.discourse_history = []
            for sent in combined:
                self.update_discourse(sent)
            result["cross_boundary_penalty"] = 0
            for i in range(len(prompt_sents), len(combined)):
                if (self.discourse_history[i].transition == TransitionType.ROUGH_SHIFT
                        and self.discourse_history[i].backward_center is None):
                    result["cross_boundary_penalty"] += 1

        return result

    def detect_boundaries(self, utterance_sequence: List[str]) -> List[int]:
        saved = list(self.discourse_history)
        self.discourse_history = []

        try:
            boundaries: List[int] = [0]
            recent_rough: int = 0

            for i, u in enumerate(utterance_sequence):
                state = self.update_discourse(u)
                t = state.transition
                if t == TransitionType.ROUGH_SHIFT:
                    recent_rough += 1
                    if state.backward_center is None and recent_rough >= 2:
                        boundaries.append(i)
                        recent_rough = 0
                elif t == TransitionType.CONTINUE:
                    recent_rough = max(0, recent_rough - 1)
                else:
                    recent_rough = max(0, recent_rough - 1)

            return boundaries
        finally:
            self.discourse_history = saved

    # ------------------------------------------------------------------
    # Annotated text output
    # ------------------------------------------------------------------

    def annotate_text(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

        saved = list(self.discourse_history)
        self.discourse_history = []

        try:
            annotations: List[Dict[str, Any]] = []
            for i, sent in enumerate(sentences):
                state = self.update_discourse(sent)
                annotations.append({
                    "index": i,
                    "utterance": sent,
                    "transition": state.transition.value if state.transition else None,
                    "preferred_center": state.preferred_center,
                    "backward_center": state.backward_center,
                    "forward_centers": state.forward_centers,
                    "entities": {
                        k: {"text": v.text, "pos": v.pos, "dep": v.dep,
                            "is_person": v.is_person, "gender": v.gender}
                        for k, v in state._entity_map.items()
                    },
                })
            return annotations
        finally:
            self.discourse_history = saved

    # ------------------------------------------------------------------
    # Validity checking
    # ------------------------------------------------------------------

    def validate_sequence(self, utterance_sequence: List[str]) -> Dict[str, Any]:
        """
        Validate discourse sequence against Centering Theory constraints.

        Rule 1: Cb(Ui) is the highest-ranked element of Cf(Ui-1) realized in Ui.
        Rule 2: If any Cf(Ui-1) element is pronominalized in Ui, Cb(Ui) = that element.
        """
        saved = list(self.discourse_history)
        self.discourse_history = []

        violations: List[Dict[str, Any]] = []
        try:
            states: List[CenteringState] = []
            for u in utterance_sequence:
                states.append(self.update_discourse(u))

            # Rule 1: Cb(Ui) must come from highest-ranked Cf(Ui-1)
            for i in range(1, len(states)):
                prev = states[i - 1]
                curr = states[i]
                if curr.backward_center is None:
                    continue

                # verify Cb(Ui) matches a Cf(Ui-1) element
                cb_in_prev = False
                for pc in prev.forward_centers:
                    if pc == curr.backward_center:
                        cb_in_prev = True
                        break

                if not cb_in_prev:
                    violations.append({
                        "utterance_index": i,
                        "rule": "Rule 1",
                        "desc": f"Cb '{curr.backward_center}' not found in Cf(Ui-1) {prev.forward_centers}",
                    })

            # Rule 2: pronoun realization constraint
            for i in range(1, len(states)):
                prev = states[i - 1]
                curr = states[i]
                for pc in prev.forward_centers:
                    if pc not in curr.forward_centers:
                        for cc in curr.forward_centers:
                            if cc.lower() in self._all_pronouns:
                                if self._pronoun_matches_entity(cc.lower(), pc, prev._entity_map):
                                    if curr.backward_center != pc:
                                        violations.append({
                                            "utterance_index": i,
                                            "rule": "Rule 2",
                                            "desc": f"'{pc}' pronominalized as '{cc}' but Cb is '{curr.backward_center}'",
                                        })

            score, dist = self._score_transitions(
                {t: sum(1 for s in states if s.transition == t) for t in TransitionType}
            )
        finally:
            self.discourse_history = saved

        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "violation_count": len(violations),
            "cohesion_score": score,
            "transition_distribution": dist,
            "utterance_count": len(utterance_sequence),
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    _viz_symbols = {
        TransitionType.ESTABLISH: "[+]",
        TransitionType.CONTINUE: "-->",
        TransitionType.RETAIN: "~> ",
        TransitionType.SMOOTH_SHIFT: "~~>",
        TransitionType.ROUGH_SHIFT: "//>",
    }

    def visualize(self, text: str) -> str:
        """Generate ASCII cohesion graph showing centers and transitions."""
        doc = self.nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

        saved = list(self.discourse_history)
        self.discourse_history = []

        try:
            lines: List[str] = []
            max_len = max((len(s) for s in sentences), default=40)
            transition_counts: Dict[TransitionType, int] = {}
            rough_streak: int = 0
            boundary_count: int = 1

            for i, sent in enumerate(sentences):
                state = self.update_discourse(sent)
                symbol = self._viz_symbols.get(state.transition, "?")
                t_name = state.transition.value if state.transition else "?"
                t = state.transition

                if t:
                    transition_counts[t] = transition_counts.get(t, 0) + 1

                # boundary detection inline
                if t == TransitionType.ROUGH_SHIFT:
                    rough_streak += 1
                    if state.backward_center is None and rough_streak >= 2:
                        boundary_count += 1
                        rough_streak = 0
                elif t == TransitionType.CONTINUE:
                    rough_streak = max(0, rough_streak - 1)
                else:
                    rough_streak = max(0, rough_streak - 1)

                # utterance line
                display = sent if len(sent) <= max_len else sent[:max_len - 3] + "..."
                lines.append(f"{symbol} [{i:02d}] {t_name:12s} | {display}")

                # center line
                cp = state.preferred_center or "."
                cb = state.backward_center or "."
                cf = ", ".join(state.forward_centers[:3]) if state.forward_centers else "."
                lines.append(f"  {'':12s} | Cp: {cp:12s} Cb: {cb:12s} Cf: [{cf}]")

                if i < len(sentences) - 1:
                    lines.append(f"  {'':12s} |")

            score, _ = self._score_transitions(transition_counts)
            bar_len = int(score * 40)
            bar = "#" * bar_len + "-" * (40 - bar_len)

            lines.append(f"\n  Cohesion: {bar} {score:.3f}")
            lines.append(f"  Segments: {boundary_count}")
            lines.append(f"  Transitions: {len(sentences)} sentences")

            return "\n".join(lines)
        finally:
            self.discourse_history = saved

    # ------------------------------------------------------------------
    # Comparative analysis
    # ------------------------------------------------------------------

    def compare_texts(
        self, *texts: str, labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare cohesion metrics across multiple texts."""
        results: List[Dict[str, Any]] = []

        for i, text in enumerate(texts):
            r = self.analyze_llm(text)
            results.append({
                "label": labels[i] if labels and i < len(labels) else f"text_{i+1}",
                "cohesion": r["response_cohesion"],
                "segments": r["response_segments"],
                "sentences": r["response_sentences"],
                "transitions": r["response_transitions"],
                "quality": r["quality"],
            })

        best = max(results, key=lambda x: x["cohesion"])
        worst = min(results, key=lambda x: x["cohesion"])

        return {
            "count": len(texts),
            "best": best,
            "worst": worst,
            "mean_cohesion": round(sum(r["cohesion"] for r in results) / len(results), 4),
            "rankings": sorted(results, key=lambda x: x["cohesion"], reverse=True),
        }

    # ------------------------------------------------------------------
    # Streaming / incremental analysis
    # ------------------------------------------------------------------

    def stream_start(self) -> None:
        """Begin a streaming session. Saves existing history, resets for stream."""
        self._stream_saved = list(self.discourse_history)
        self.discourse_history = []
        self._stream_stats: Dict[str, Any] = {
            "sentences_processed": 0,
            "transition_counts": {t: 0 for t in TransitionType},
            "running_cohesion": 1.0,
        }

    def stream_feed(self, utterance: str) -> Dict[str, Any]:
        """Feed one utterance into the stream. Returns incremental stats."""
        if not hasattr(self, "_stream_stats"):
            self.stream_start()

        state = self.update_discourse(utterance)
        t = state.transition
        stats = self._stream_stats
        stats["sentences_processed"] += 1

        if t:
            stats["transition_counts"][t] = stats["transition_counts"].get(t, 0) + 1

        score, dist = self._score_transitions(stats["transition_counts"])
        stats["running_cohesion"] = score

        return {
            "utterance": utterance[:60],
            "transition": t.value if t else None,
            "cp": state.preferred_center,
            "cb": state.backward_center,
            "running_cohesion": score,
            "sentences": stats["sentences_processed"],
        }

    def stream_flush(self) -> Dict[str, Any]:
        """End the streaming session. Restores previous history. Returns final summary."""
        stats = getattr(self, "_stream_stats", None)
        if stats is None:
            return {"error": "no active stream"}

        score, dist = self._score_transitions(stats["transition_counts"])

        if hasattr(self, "_stream_saved"):
            self.discourse_history = self._stream_saved
            del self._stream_saved

        return {
            "total_sentences": stats["sentences_processed"],
            "final_cohesion": score,
            "transition_distribution": dist,
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
                "similarity_threshold": self.similarity_threshold,
                "gender_lookup": self._gender_lookup,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.discourse_history = data["history"]
        self.salience_weights = data.get("salience_weights", self.salience_weights)
        self.pos_weights = data.get("pos_weights", self.pos_weights)
        self.history_limit = data.get("history_limit", self.history_limit)
        self.similarity_threshold = data.get("similarity_threshold", 0.65)
        self._gender_lookup = data.get("gender_lookup", self._gender_lookup)

    def reset(self) -> None:
        """Clear discourse history."""
        self.discourse_history.clear()
