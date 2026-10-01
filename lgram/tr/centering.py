"""
Centering Theory for Turkish, on UD parses (see `lgram.tr.parser`).

What differs from the English engine:

* **Zero pronouns.** Turkish drops subjects; the missing subject is recovered
  from the finite verb's person/number agreement and linked to the previous
  utterance's Cf (or to the speaker/addressee for 1st/2nd person).
* **Possessive suffixes** (`annesi`, `kitabım`) realise a possessor entity even
  without a genitive noun phrase.
* **No gender.** `o` is genderless, so 3rd-person anaphors are resolved by
  number and salience only, never by name/gender lookup.

This is a *diagnostic*: it reports transitions and a rough-shift ratio. It does not
produce a quality score — the English scalar failed external validation (see the
README's Validation Status) and this module has not earned one yet.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import snowballstemmer

from ..models.centering_theory import TransitionType
from .parser import JointParser, Token, split_sentences

_STEM = snowballstemmer.stemmer("turkish")

# Cf ranking by grammatical role. ponytail: Turkish word order is free, so role
# alone is a rough salience proxy; upgrade to information-structure ranking
# (verb-adjacent focus, case marking) if the benchmark shows it matters.
ROLE_RANK = {
    "nsubj": 4.0,
    "nsubj:outer": 4.0,
    "obj": 3.0,
    "iobj": 2.5,
    "obl": 2.0,
    "nmod:poss": 1.0,
    "appos": 1.0,
}
_SUBJECTS = frozenset({"nsubj", "nsubj:outer", "csubj"})
_THIRD = frozenset(
    "o onu ona onda ondan onun onunla onlar onları onlara onlarda onlardan "
    "onların onlarla".split()
)
_SPEECH_PRONOUNS = (  # prefix -> speaker/addressee key; PRON upos guards "benzer"
    ("ben", "@1sg"),
    ("ban", "@1sg"),
    ("sen", "@2sg"),
    ("san", "@2sg"),
    ("biz", "@1pl"),
    ("siz", "@2pl"),
)


@dataclass
class Mention:
    key: str
    rank: float
    pos: int
    plural: bool = False
    kind: str = "noun"  # noun | pronoun | zero | possessor


@dataclass
class UtteranceState:
    text: str
    cf: List[Mention]
    cb: Optional[str]
    cp: Optional[str]
    transition: TransitionType
    # anaphor kinds ("zero" | "pronoun" | "possessor") with no compatible antecedent
    unresolved: List[str] = field(default_factory=list)

    @property
    def zero_subject(self) -> bool:
        """A dropped subject was found *and* linked to an antecedent."""
        return any(m.kind == "zero" for m in self.cf)

    @property
    def zero_detected(self) -> bool:
        """A dropped subject was found, whether or not it could be linked."""
        return self.zero_subject or "zero" in self.unresolved


@dataclass
class TurkishReport:
    utterances: List[UtteranceState]

    @property
    def transition_distribution(self) -> Dict[str, float]:
        counts = Counter(
            u.transition.value
            for u in self.utterances
            if u.transition != TransitionType.ESTABLISH
        )
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()} if total else {}

    @property
    def rough_shift_ratio(self) -> float:
        """Share of Rough-Shift + NOCB transitions. Lower = smoother flow.

        Keeps its pre-split meaning (NOCB used to be labelled Rough-Shift) so earlier
        order-shuffling results stay comparable. A test statistic, not a quality score.
        """
        dist = self.transition_distribution
        return dist.get(TransitionType.ROUGH_SHIFT.value, 0.0) + dist.get(
            TransitionType.NOCB.value, 0.0
        )


def _lower(s: str) -> str:
    # Python's str.lower() turns "İ" into "i" + combining dot; fix Turkish I/İ first
    return s.replace("İ", "i").replace("I", "ı").lower()


def _base(form: str) -> str:
    return _lower(form).replace("’", "'").split("'")[0]


def entity_key(tok: Token) -> str:
    """Lexical identity of a noun mention. Proper names are not stemmed: the
    Turkish snowball stemmer mangles them ("Çadır" -> "ça")."""
    base = _base(tok.form)
    return base if tok.upos == "PROPN" else _STEM.stemWord(base)


def _speech_key(base: str) -> Optional[str]:
    for prefix, key in _SPEECH_PRONOUNS:
        if base.startswith(prefix):
            return key
    return None


def _role(tok: Token, by_id: Dict[int, Token]) -> Tuple[str, float]:
    """Grammatical role, following `conj` chains (conjuncts rank a bit lower)."""
    penalty = 0.0
    for _ in range(10):  # bounded: a malformed parse must not loop forever
        if tok.deprel != "conj" or tok.head not in by_id:
            break
        tok, penalty = by_id[tok.head], penalty + 0.5
    return tok.deprel, penalty


def _finite_carrier(root: Token, children: Dict[int, List[Token]]) -> Optional[Token]:
    """The token carrying tense/person agreement: the root, or its copula/aux."""
    for cand in [root] + [
        c for c in children.get(root.id, []) if c.deprel in ("cop", "aux")
    ]:
        if cand.feats.get("VerbForm") == "Fin":
            return cand
    return None


def extract_mentions(
    tokens: Sequence[Token],
    prev_cf: Sequence[Mention],
    resolve_zero: bool = True,
    prev_cb: Optional[str] = None,
) -> Tuple[List[Mention], List[str]]:
    # anaphors look at the previous Cb first, then down the previous Cf
    # (+0.013 transition accuracy on Turkish-ITCC train, +0.006 dev)
    prev_cf = sorted(prev_cf, key=lambda m: m.key != prev_cb) if prev_cb else prev_cf
    by_id = {t.id: t for t in tokens}
    children: Dict[int, List[Token]] = {}
    for t in tokens:
        children.setdefault(t.head, []).append(t)

    mentions: List[Mention] = []
    anaphors: List[Mention] = []  # key unresolved yet: kind + plural + rank set
    subject_of: Dict[int, Mention] = {}  # predicate token id -> its subject mention

    def anaphor(kind: str, rank: float, pos: int, plural: bool) -> None:
        anaphors.append(Mention("", rank, pos, plural, kind))

    def speech(person: str, number: str) -> str:
        return f"@{person}{'pl' if number == 'Plur' else 'sg'}"

    for t in tokens:
        rel, penalty = _role(t, by_id)
        rank = ROLE_RANK.get(rel, 0.0) - penalty
        plural = t.feats.get("Number") == "Plur"

        if t.upos in ("NOUN", "PROPN") and rel in ROLE_RANK:
            mentions.append(Mention(entity_key(t), rank, t.id, plural, "noun"))
            if t.deprel in _SUBJECTS:
                subject_of[t.head] = mentions[-1]
        elif t.upos == "PRON" and rel in ROLE_RANK:
            base = _base(t.form)
            if base in _THIRD:
                anaphor("pronoun", rank, t.id, base.startswith("onlar"))
            elif (key := _speech_key(base)) is not None:
                mentions.append(Mention(key, rank, t.id, key.endswith("pl"), "pronoun"))

        # possessive suffix without an explicit genitive NP -> implicit possessor
        psor = t.feats.get("Person[psor]")
        has_genitive = any(c.deprel == "nmod:poss" for c in children.get(t.id, []))
        if t.upos in ("NOUN", "PROPN") and psor and not has_genitive:
            if psor in ("1", "2"):
                key = speech(psor, t.feats.get("Number[psor]", ""))
                mentions.append(
                    Mention(key, ROLE_RANK["nmod:poss"], t.id, False, "possessor")
                )
            else:
                anaphor(
                    "possessor",
                    ROLE_RANK["nmod:poss"],
                    t.id,
                    t.feats.get("Number[psor]") == "Plur",
                )

    # zero subject: every finite predicate (root, subordinate, parataxis...) with no
    # overt subject. The root's zero keeps pos 0 so it heads the Cf among subjects.
    predicates = {
        (t.head if t.deprel in ("cop", "aux") and t.head in by_id else t.id)
        for t in tokens
        if t.feats.get("VerbForm") == "Fin"
    }
    for pred in [by_id[i] for i in sorted(predicates)] if resolve_zero else []:
        if any(c.deprel in _SUBJECTS for c in children.get(pred.id, [])):
            continue
        # a conjunct shares its subject with the clause it is conjoined to
        if pred.deprel == "conj" and pred.head in predicates:
            continue
        carrier = _finite_carrier(pred, children)
        if carrier is None or carrier.feats.get("Mood") == "Imp":
            continue
        pos = 0 if pred.head == 0 else pred.id
        person = carrier.feats.get("Person", "3")
        number = carrier.feats.get("Number", "")
        if person in ("1", "2"):
            mentions.append(Mention(speech(person, number), 4.0, pos, False, "zero"))
        else:
            anaphor("zero", 4.0, pos, number == "Plur")
            subject_of[pred.id] = anaphors[-1]

    def clause_subject(tok: Token) -> Optional[Mention]:
        """Subject of the clause `tok` sits in, unless `tok` is inside that subject."""
        cur, in_subject = tok, False
        for _ in range(20):  # bounded walk up to the governing predicate
            if cur.id in predicates or cur.head not in by_id:
                break
            in_subject = in_subject or cur.deprel in _SUBJECTS
            cur = by_id[cur.head]
        while cur.id not in subject_of and cur.deprel == "conj" and cur.head in by_id:
            cur = by_id[cur.head]  # conjuncts share the subject
        return None if in_subject else subject_of.get(cur.id)

    # Resolve 3rd-person anaphors against the previous Cf. Skip entities named
    # overtly in this utterance (disjoint reference: "Ali onu gördü").
    overt = {m.key for m in mentions}
    unresolved: List[str] = []
    # possessors last: they may bind to a subject resolved in this loop
    for a in sorted(anaphors, key=lambda a: a.kind == "possessor"):
        if a.kind == "possessor":
            # "Ali annesini aradı": a possessor is usually its own clause's subject
            subj = clause_subject(by_id[a.pos])
            if subj is not None and subj.key and not subj.key.startswith("@"):
                if subj.plural == a.plural:
                    a.key = subj.key
                    mentions.append(a)
                    continue
        for cand in prev_cf:
            if (
                cand.plural == a.plural
                and not cand.key.startswith("@")
                and cand.key not in overt
            ):
                a.key = cand.key
                mentions.append(a)
                break
        else:
            unresolved.append(a.kind)
    return mentions, unresolved


def _merge_cf(mentions: Sequence[Mention]) -> List[Mention]:
    """One Mention per entity (best rank, earliest position), salience-ordered."""
    best: Dict[str, Mention] = {}
    for m in mentions:
        cur = best.get(m.key)
        if cur is None or (m.rank, -m.pos) > (cur.rank, -cur.pos):
            best[m.key] = m
    return sorted(best.values(), key=lambda m: (-m.rank, m.pos))


def _transition(
    prev: Optional[UtteranceState], cf: List[Mention]
) -> Tuple[Optional[str], Optional[str], TransitionType]:
    cp = cf[0].key if cf else None
    if prev is None:
        return None, cp, TransitionType.ESTABLISH
    keys = {m.key for m in cf}
    cb = next((m.key for m in prev.cf if m.key in keys), None)
    if cb is None:  # no shared entity: Cb undefined
        return None, cp, TransitionType.NOCB
    if prev.cb is None or cb == prev.cb:
        return cb, cp, (TransitionType.CONTINUE if cb == cp else TransitionType.RETAIN)
    return (
        cb,
        cp,
        (TransitionType.SMOOTH_SHIFT if cb == cp else TransitionType.ROUGH_SHIFT),
    )


def analyze_parsed(
    sentences: Sequence[str],
    parses: Sequence[Sequence[Token]],
    resolve_zero: bool = True,
) -> TurkishReport:
    """Centering over already-parsed sentences (no model needed).

    `resolve_zero=False` ablates zero-pronoun recovery (for benchmarks).
    """
    states: List[UtteranceState] = []
    for text, tokens in zip(sentences, parses):
        if not tokens:
            continue
        prev = states[-1] if states else None
        mentions, unresolved = extract_mentions(
            tokens,
            prev.cf if prev else [],
            resolve_zero,
            prev.cb if prev else None,
        )
        cf = _merge_cf(mentions)
        cb, cp, transition = _transition(prev, cf)
        states.append(UtteranceState(text, cf, cb, cp, transition, unresolved))
    return TurkishReport(states)


class TurkishCenteringAnalyzer:
    """Text -> `TurkishReport`. The neural parser loads lazily on first use."""

    def __init__(self, parser: Optional[JointParser] = None):
        self._parser = parser

    @property
    def parser(self) -> JointParser:
        if self._parser is None:
            self._parser = JointParser()
        return self._parser

    def analyze(self, text: str) -> TurkishReport:
        sentences = split_sentences(text)
        return analyze_parsed(sentences, [self.parser.parse(s) for s in sentences])
