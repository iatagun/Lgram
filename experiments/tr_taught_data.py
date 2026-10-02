"""
Hand-written teaching examples -> fastcoref training lines.

tr_taught_examples.txt holds short texts written to show the model what a reader does
with Turkish reference: who a dropped subject is, whose "annesi" it is, that two names
are two people, that a plural verb needs a plural antecedent. One text per line; a
referring word carries tags, an integer naming the entity:

    Ali/1 dün Ayşe'yi/2 aradı. Ona/2 kitabını/p2 geri verecekti/s1. Ama evde yoktu/s2.

    word/N    the word is a mention of entity N
    word/sN   the dropped subject of this predicate is entity N
    word/pN   the unexpressed possessor of this noun is entity N   (tags can be chained)

The text is parsed and turned into the same token stream the model sees at inference
(tr_coref_centering.build_stream: dropped subjects and possessors as "o" / "onun"
placeholders). A tag whose slot the parser did not produce is dropped and counted.

Usage:
    python experiments/tr_taught_data.py EXAMPLES.txt OUT.jsonl [--repeat 1]
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tr_coref_centering import build_stream  # noqa: E402

from lgram.tr.parser import JointParser, split_sentences, tokenize  # noqa: E402

TAGGED = re.compile(r"([^\s/]+)((?:/[sp]?\d+)+)")


def read(line):
    """(clean text, [(word, [(kind, entity), ...]), ...] in text order)."""
    tags = [
        (m.group(1), [(t[:-len(n)] or "m", n) for t in m.group(2).split("/")[1:]
                      for n in [re.search(r"\d+$", t).group()]])
        for m in TAGGED.finditer(line)
    ]  # fmt: skip
    return TAGGED.sub(lambda m: m.group(1), line), tags


def convert(line, parser, stats):
    text, tags = read(line)
    sents = split_sentences(text)
    parses = [parser.parse(s) for s in sents]
    stream, where = build_stream(parses)
    # tagged words are found in order among the tokens of the whole text
    tokens = [(i, t) for i, toks in enumerate(parses) for t in toks]
    clusters, k = {}, 0
    for word, marks in tags:
        head = tokenize(word)[-1]
        while k < len(tokens) and tokens[k][1].form != head:
            k += 1
        if k == len(tokens):
            stats["tagged word not found"] += len(marks)
            break
        i, t = tokens[k]
        k += 1
        for kind, ent in marks:
            if kind == "m":
                idx = where.get((i, "tok", t.id))
            elif kind == "s":  # the root's dropped subject is stored at position 0
                idx = where.get((i, "zero", 0 if t.head == 0 else t.id))
                if idx is None and t.deprel in ("cop", "aux"):  # "okulda değildi"
                    h = parses[i][t.head - 1]
                    idx = where.get((i, "zero", 0 if h.head == 0 else h.id))
            else:
                idx = where.get((i, "possessor", t.id))
            if idx is None:
                stats[f"no slot for /{kind}"] += 1
            else:
                clusters.setdefault(ent, []).append([idx, idx])
                stats["mentions"] += 1
    return {"tokens": stream, "clusters": [sorted(c) for c in clusters.values()]}


def main(argv):
    src, dst = argv[0], argv[1]
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv else 1
    parser, stats, docs = JointParser(), Counter(), []
    for n, line in enumerate(Path(src).read_text(encoding="utf-8").splitlines()):
        if line.strip() and not line.startswith("#"):
            docs.append(
                {"doc_key": f"taught{n}", **convert(line.strip(), parser, stats)}
            )
    with open(dst, "w", encoding="utf-8") as f:
        for r in range(repeat):
            for d in docs:
                f.write(json.dumps({**d, "doc_key": f"{d['doc_key']}_{r}"}, ensure_ascii=False) + "\n")  # fmt: skip
    print(f"{len(docs)} texts x{repeat}, {sum(len(d['tokens']) for d in docs)} tokens:", dict(stats))  # fmt: skip
    return 0


if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore")
    sys.exit(main(sys.argv[1:]))
