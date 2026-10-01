"""
Experimental: Centering transitions on top of fastcoref entity IDs.

lgram's built-in centering decides entity identity with surface strings, word-vector
similarity and "it matches anything" rules; this prototype replaces all of that with a
neural coreference resolver and ranks Cf purely by grammatical role. Transitions come
from lgram.transition_eval.derive, so both are scored by the same BFP rules.

Results vs benchmark_data/en_transitions_claude.csv (n=96, single annotator, the rules
below were tuned while looking at these rows -> optimistic):
    lgram built-in 0.323 | FCoref 0.823 | LingMessCoref 0.844 (kappa 0.79)

Requires `pip install fastcoref` (not a package dependency) and en_core_web_md.

Usage:
    python experiments/coref_centering.py GOLD.csv [--lingmess] [--errors]
"""

from __future__ import annotations

import csv
import logging
import re
import sys
import unicodedata
import warnings
from collections import Counter
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lgram.transition_eval import _derived, derive, kappa  # noqa: E402

SUBJ = {"nsubj", "nsubjpass", "csubj", "expl"}
IOBJ = {"dative", "iobj"}
OBJ = {"dobj", "obj"}
SUBORD = {"advcl", "relcl", "acl", "ccomp", "xcomp", "csubj", "pcomp"}
SAY = {"say", "cry", "ask", "reply", "answer", "add", "exclaim", "call", "shout",
       "whisper", "think", "continue"}  # fmt: skip
# a quote opened in this sentence, or one carried over from the previous sentence;
# ’ followed by a letter is an apostrophe (king’s, can’t), not a closing quote
QUOTES = re.compile(r"[‘“\"].*?(?:[’”\"](?![A-Za-z])|$)|^[^‘“]*?[’”](?![A-Za-z])")


def fold(w: str) -> str:
    """Lowercase and strip diacritics: Türk -> turk."""
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", w.lower())
        if not unicodedata.combining(c)
    )


def key(t) -> str:
    """Lexical identity of an unclustered noun; full name for proper nouns."""
    if t.pos_ == "PROPN":  # Murad II != Mehmed II although both heads are "II"
        compound = " ".join(c.text for c in t.children if c.dep_ == "compound")
        return fold(f"{compound} {t.text}").strip()
    return fold(t.lemma_)


def role(t) -> int:
    return 0 if t.dep_ in SUBJ else 1 if t.dep_ in IOBJ else 2 if t.dep_ in OBJ else 3


def subordinate(t) -> bool:
    return any(a.dep_ in SUBORD for a in [t] + list(t.ancestors))


class CorefCentering:
    def __init__(self, lingmess: bool = False):
        import spacy

        warnings.filterwarnings("ignore")
        logging.disable(logging.WARNING)
        if lingmess:
            import transformers.modeling_utils as mu

            # ponytail: Longformer has no sdpa in transformers>=4.5x; force eager.
            # Drop once fastcoref passes attn_implementation itself.
            mu.PreTrainedModel._check_and_adjust_attn_implementation = (
                lambda self, *a, **k: "eager"
            )
            from fastcoref import LingMessCoref as Model
        else:
            from fastcoref import FCoref as Model
        self.nlp = spacy.load("en_core_web_md")
        self.coref = Model(device="cpu")

    def paragraph(self, sents: list[str]) -> list[tuple | None]:
        """Per sentence: None for the first, else (cp, cb, transition, cf, prev_cf)."""
        text, offs = "", []
        for s in sents:
            offs.append(len(text))
            text += s + " "
        clusters = self.coref.predict(texts=[text])[0].get_clusters(as_strings=False)
        span2id = {(a, b): f"E{i}" for i, cl in enumerate(clusters) for a, b in cl}
        docs = [self.nlp(s) for s in sents]

        def mentions(j):
            off, s = offs[j], sents[j]
            for (a, b), eid in span2id.items():
                if off <= a and b <= off + len(s) + 1:
                    sp = docs[j].char_span(a - off, b - off, alignment_mode="expand")
                    if sp is not None:
                        yield eid, sp

        # an unclustered noun with the same head as a cluster mention joins that cluster
        key2id: dict[str, str] = {}
        for j in range(len(sents)):
            for eid, sp in mentions(j):
                if sp.root.pos_ in ("NOUN", "PROPN"):
                    key2id.setdefault(key(sp.root), eid)

        states = []
        for j, (s, doc) in enumerate(zip(sents, docs)):
            quoted = {i for m in QUOTES.finditer(s) for i in range(m.start(), m.end())}
            said_subj = set()  # "‘…’ said the ass" -> the ass is the subject
            for v in doc:
                if v.lemma_.lower() in SAY and v.idx not in quoted:
                    if not any(
                        c.dep_ in SUBJ and c.idx not in quoted for c in v.children
                    ):
                        nxt = next(
                            (t for t in doc[v.i + 1 :]
                             if t.pos_ in ("NOUN", "PROPN", "PRON") and t.idx not in quoted),
                            None,
                        )  # fmt: skip
                        if nxt is not None:
                            said_subj.add(nxt.i)

            best: dict[str, tuple] = {}  # best-ranked mention per entity

            def offer(eid, tok, span_len):
                r = 0 if tok.i in said_subj else role(tok)
                # narration > quoted speech; role; main clause; position; longer span
                # (the group "the peasant and his wife" beats its member "peasant")
                k = (tok.idx in quoted, r, subordinate(tok), tok.i, -span_len)
                if eid not in best or k < best[eid][0]:
                    best[eid] = (k, tok)

            used = set()
            for eid, sp in mentions(j):
                offer(eid, sp.root, len(sp))
                used.add(sp.root.i)
            for t in doc:
                if (
                    t.pos_ in ("NOUN", "PROPN")
                    and t.dep_ not in ("compound", "intj")
                    and t.tag_ != "UH"
                    and t.i not in used
                ):
                    k = key(t)
                    possessed = any(c.dep_ == "poss" for c in t.children)
                    # Kepler's laws is not "his laws": possessed nouns keep their own ID
                    offer("L:" + k if possessed else key2id.get(k, "L:" + k), t, 1)
            cf = sorted(best, key=lambda e: best[e][0])
            states.append([(e, best[e][1].text, best[e][1].dep_) for e in cf])

        out, prev, prev_cb = [], None, None
        for cfx in states:
            cf = [e for e, _, _ in cfx]
            if prev is None:
                out.append(None)
            else:
                cp = cf[0] if cf else None
                cb = next((e for e, _, _ in prev if e in cf), None)
                out.append((cp, cb, derive(cp, cb, prev_cb), cfx, prev))
                prev_cb = cb
            prev = cfx
        return out


def main(argv: list[str]) -> int:
    path = Path(argv[0])
    cc = CorefCentering(lingmess="--lingmess" in argv)
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    byid = {r["id"]: r for r in rows}
    gold = _derived(rows)
    pred = []
    for _, g in groupby(rows, key=lambda r: r["para"]):
        pred += [x for x in cc.paragraph([r["sentence"] for r in g]) if x]

    gt, st = [x[2] for x in gold], [x[2] for x in pred]
    n = len(gt)
    nocb = sum((g[1] is None) == (p[1] is None) for g, p in zip(gold, pred)) / n
    acc = sum(a == b for a, b in zip(gt, st)) / n
    print(
        f"n={n}  accuracy={acc:.3f}  kappa={kappa(gt, st):.3f}  NOCB agreement={nocb:.3f}"
    )
    labels = ["Continue", "Retain", "Smooth-Shift", "Rough-Shift", "NOCB"]
    c = Counter(zip(gt, st))
    print("gold \\ system  " + " ".join(f"{x[:8]:>8s}" for x in labels))
    for a in labels:
        print(f"{a:14s} " + " ".join(f"{c[(a, b)]:8d}" for b in labels))

    if "--errors" in argv:
        fmt = lambda cfx: " ".join(f"{t}/{d}[{e}]" for e, t, d in cfx)  # noqa: E731
        for (i, _, g), (cp, cb, s, cfx, pcfx) in zip(gold, pred):
            if g != s:
                r = byid[i]
                print(f"\n--- id {i}  gold {g} (cp={r['gold_cp']} cb={r['gold_cb']})"
                      f"  system {s} (cp={cp} cb={cb})  {r['note']}")  # fmt: skip
                print(f"  S     : {r['sentence'][:160]}")
                print(f"  Cf    : {fmt(cfx)}")
                print(f"  prevCf: {fmt(pcfx)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
