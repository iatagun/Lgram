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
# ponytail: punctuation-only sentence splitter — abbreviations ("Dr.", "vb.")
# will over-split; swap in a proper splitter if it shows up in the benchmark.
_SENT = re.compile(r"(?<=[.!?…])\s+")


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


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


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
