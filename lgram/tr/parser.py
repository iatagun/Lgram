"""
Turkish parsing front-end: raw text -> UD tokens (form, UPOS, FEATS, HEAD, DEPREL).

The neural parser is DizgeBERT-Joint (morphology + dependencies in one pass).
Everything downstream (`lgram.tr.centering`) only sees `Token` objects, so it can
be tested without the model.

Install the optional stack with:  pip install centering-lgram[tr]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

MODEL_ID = "iatagun/DizgeBERT-Joint"
# Pinned: the model ships custom code (trust_remote_code), so never track a
# moving branch. Bump deliberately after re-running the tr benchmark.
MODEL_REVISION = "891eee1a6a2c63338b57e83fba4cd649fb7d2111"

_WORD = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]")
# Candidate boundary: sentence punctuation, optional closing quotes/brackets, space.
_BOUNDARY = re.compile(r"([.!?…]+)['\"’”)\]]*\s+")
# ponytail: a closed list, and no split after ":" or before a lower-case start.
# On Turkish-ITCC raw text: boundary precision 0.99, recall 0.93. Swap in a trained
# splitter if that starts to matter.
_ABBREVIATIONS = frozenset(
    "prof dr doç yrd av sn bkz vb vs vd örn yy no nr st mr mrs ms alb yzb tğm "
    "gen org uzm op müh öğr gör".split()
)


@dataclass
class Token:
    """One UD token. `id` and `head` are 1-based; head == 0 means root."""

    id: int
    form: str
    upos: str
    feats: Dict[str, str] = field(default_factory=dict)
    head: int = 0
    deprel: str = "dep"


def parse_feats(feats: str) -> Dict[str, str]:
    if not feats or feats == "_":
        return {}
    return dict(kv.split("=", 1) for kv in feats.split("|") if "=" in kv)


def _is_boundary(text: str, m: "re.Match[str]") -> bool:
    nxt = text[m.end() : m.end() + 1]
    # a sentence starts with a capital, a digit, a quote, a dash or a bracket;
    # "21. yüzyıl", "vb. gibi" and "Nerdeydin? dedi" continue in lower case
    if not (nxt.isupper() or nxt.isdigit() or nxt in "'\"‘“—–-(["):
        return False
    punct = m.group(1)
    if punct == ".":
        words = text[: m.start()].split()
        prev = words[-1].strip("'\"‘“(") if words else ""
        if prev.lower() in _ABBREVIATIONS or (len(prev) == 1 and prev.isupper()):
            return False  # "Prof. Dr. Ahmet", "A. Kadir"
    return True


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    out, start = [], 0
    for m in _BOUNDARY.finditer(text):
        if _is_boundary(text, m):
            out.append(text[start : m.end()].strip())
            start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


def tokenize(sentence: str) -> List[str]:
    return _WORD.findall(sentence)


class JointParser:
    """Thin wrapper over DizgeBERT-Joint. Loads the model once, on construction."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        scheme: str = "kenet",
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Turkish support needs the optional stack: "
                "pip install centering-lgram[tr]"
            ) from e
        self.scheme = scheme
        self._model = AutoModel.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        ).eval()
        self._tok = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def parse(self, sentence: str) -> List[Token]:
        words = tokenize(sentence)
        if not words:
            return []
        rows = self._model.predict(words, scheme=self.scheme, tokenizer=self._tok)
        return [
            Token(i, form, upos, parse_feats(feats), int(head), deprel)
            for i, (form, upos, _xpos, feats, head, deprel) in enumerate(rows, 1)
        ]
