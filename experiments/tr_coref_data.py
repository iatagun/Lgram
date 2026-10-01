"""
Turkish-ITCC (CorefUD) -> fastcoref training jsonlines.

Zero mentions (empty nodes: dropped subjects / possessors) are inserted into the token
stream as the pronoun forms ITCC gives them ("o", "ben", "onun"...), right after the
surface token they hang on. At inference lgram.tr's rule-based zero detector inserts the
same placeholders, so the model only has to resolve them.

Each line: {"doc_key", "tokens", "clusters": [[[start, end], ...], ...]} (inclusive ends).

Documents are cut at sentence boundaries into chunks of at most --max-tokens tokens
(default 400): a 4 GB GPU cannot backprop a 3k-token document, and centering only
needs links between adjacent sentences.

Usage:
    python experiments/tr_coref_data.py IN.conllu OUT.jsonl [--max-tokens 400]
"""

import json
import sys

import udapi


def convert(path):
    doc = udapi.Document(path)
    docs, cur = [], None
    for bundle in doc.bundles:
        root = bundle.trees[0]
        if root.newdoc or cur is None:
            cur = {"doc_key": root.newdoc if isinstance(root.newdoc, str) else root.bundle.bundle_id,
                   "tokens": [], "node2idx": {}, "sent_starts": []}  # fmt: skip
            docs.append(cur)
        cur["sent_starts"].append(len(cur["tokens"]))
        empties = sorted(root.empty_nodes, key=lambda n: float(n.ord))
        for tok in root.token_descendants:
            words = tok.words if hasattr(tok, "words") else [tok]
            idx = len(cur["tokens"])
            cur["tokens"].append(tok.form)
            for w in words:
                cur["node2idx"][id(w)] = idx
            last = max(int(w.ord) for w in words)
            first = min(int(w.ord) for w in words)
            for e in empties:
                if first <= int(float(e.ord)) <= last:
                    cur["node2idx"][id(e)] = len(cur["tokens"])
                    cur["tokens"].append(e.form)

    clusters_by_doc = {id(d): {} for d in docs}
    node_doc = {k: d for d in docs for k in d["node2idx"]}
    for e in doc.coref_entities:
        for m in e.mentions:
            idxs = [
                node_doc[id(w)]["node2idx"][id(w)] for w in m.words if id(w) in node_doc
            ]
            if not idxs:
                continue
            d = node_doc[id(m.words[0])]
            clusters_by_doc[id(d)].setdefault(e.eid, []).append([min(idxs), max(idxs)])
    out = []
    for d in docs:
        clusters = [sorted(c) for c in clusters_by_doc[id(d)].values()]
        out.append({"doc_key": d["doc_key"], "tokens": d["tokens"], "clusters": clusters,
                    "sent_starts": d["sent_starts"]})  # fmt: skip
    return out


def chunk(doc, max_tokens):
    """Split at sentence boundaries; keep mentions that fall inside each chunk."""
    starts = doc["sent_starts"] + [len(doc["tokens"])]
    bounds, a = [], 0
    for s, e in zip(starts, starts[1:]):
        if e - a > max_tokens and s > a:
            bounds.append((a, s))
            a = s
    bounds.append((a, len(doc["tokens"])))
    for k, (a, b) in enumerate(bounds):
        clusters = [
            [[s - a, e - a] for s, e in c if a <= s and e < b] for c in doc["clusters"]
        ]
        yield {"doc_key": f"{doc['doc_key']}_{k}", "tokens": doc["tokens"][a:b],
               "clusters": [c for c in clusters if c]}  # fmt: skip


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    max_tokens = (
        int(sys.argv[sys.argv.index("--max-tokens") + 1])
        if "--max-tokens" in sys.argv
        else 400
    )
    docs = [c for d in convert(src) for c in chunk(d, max_tokens)]
    with open(dst, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n_tok = sum(len(d["tokens"]) for d in docs)
    n_m = sum(len(c) for d in docs for c in d["clusters"])
    print(f"{len(docs)} docs, {n_tok} tokens, {n_m} mentions, "
          f"{sum(len(d['clusters']) for d in docs)} clusters -> {dst}")  # fmt: skip
