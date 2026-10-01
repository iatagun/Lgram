"""
Turkish centering derived from Turkish-ITCC gold coreference (CorefUD 1.3), including
zero subjects (nSubj) and dropped possessors (dPsor) annotated as empty nodes.

Compares Cf rankings by how often Cp(U_i-1) is the Cb of U_i, with and without zeros.
Data: CorefUD-1.3-public (LINDAT hdl 11234/1-5896). Turkish-ITCC is CC BY-NC-SA 4.0
and is not redistributed here. Requires `pip install udapi`.

Usage:
    python experiments/itcc_centering.py tr_itcc-corefud-train.conllu [more.conllu ...]
"""

import sys
from collections import Counter
from pathlib import Path

import udapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lgram.transition_eval import derive  # noqa: E402


def read(path):
    """Yield paragraphs: list of sentences; sentence = list of (eid, mentionType, head)."""
    doc = udapi.Document(path)
    by_root = {}
    for e in doc.coref_entities:
        for m in e.mentions:
            mt = (
                m.other.get("mentionType", "overt")
                if hasattr(m.other, "get")
                else "overt"
            )
            h = m.head
            tok = {
                "form": h.form,
                "deprel": h.deprel or "",
                "feats": str(h.feats),
                "order": float(h.ord),
                "empty": h.is_empty(),
            }
            by_root.setdefault(id(h.root), []).append((e.eid, mt, tok))
    para = []
    for bundle in doc.bundles:
        root = bundle.trees[0]
        if (root.newdoc or root.newpar) and para:
            yield para
            para = []
        para.append(by_root.get(id(root), []))
    if para:
        yield para


def role(mt, tok):
    if mt == "nSubj":
        return 0
    if mt == "dPsor":
        return 3
    d = tok["deprel"].split(":")[0]
    if d in ("nsubj", "csubj"):
        return 0
    if d == "obl" and "Case=Dat" in tok["feats"]:
        return 1
    if d in ("obj", "iobj"):
        return 2
    return 3


RANKINGS = {
    "role": lambda mt, t: (role(mt, t), t["order"]),
    "role,zero-first": lambda mt, t: (role(mt, t), mt == "overt", t["order"]),
    "zero-first,role": lambda mt, t: (mt == "overt", role(mt, t), t["order"]),
    "linear": lambda mt, t: (t["order"],),
    "linear,zero-first": lambda mt, t: (mt == "overt", t["order"]),
}


def cf(sent, rank):
    best = {}
    for eid, mt, t in sent:
        k = rank(mt, t)
        if eid not in best or k < best[eid]:
            best[eid] = k
    return sorted(best, key=best.get)


def run(paths, zeros=True):
    paras = [p for path in paths for p in read(path)]
    if not zeros:
        paras = [[[m for m in s if m[1] == "overt"] for s in p] for p in paras]
    for name, rank in RANKINGS.items():
        pred = tot = 0
        trans = Counter()
        for p in paras:
            cfs = [cf(s, rank) for s in p]
            prev_cb = None
            for i in range(1, len(cfs)):
                cb = next((e for e in cfs[i - 1] if e in cfs[i]), None)
                cp = cfs[i][0] if cfs[i] else None
                trans[derive(cp, cb, prev_cb)] += 1
                prev_cb = cb
                # Cp(U_{i-1}) predicts Cb(U_i)
                if cb is not None and cfs[i - 1]:
                    tot += 1
                    pred += cfs[i - 1][0] == cb
        n = sum(trans.values())
        dist = "  ".join(f"{k[:6]}={v/n:.0%}" for k, v in sorted(trans.items()))
        print(f"  {name:18s} Cp->next Cb {pred/tot:.3f} (n={tot})   {dist}")
    return paras


if __name__ == "__main__":
    paths = sys.argv[1:]
    paras = list(p for path in paths for p in read(path))
    print(
        f"paragraphs={len(paras)} sentences={sum(map(len,paras))} transitions={sum(len(p)-1 for p in paras)}"
    )
    print("with zero mentions:")
    run(paths, True)
    print("overt mentions only:")
    run(paths, False)
