"""
Transition accuracy against hand-annotated Cp/Cb (benchmark_data/en_transitions_gold.csv).

The annotator marks Cp and Cb per utterance; the gold transition is derived from those
with the same BFP rules the system uses (Cb undefined -> NOCB; Cb(Ui-1) undefined ->
treated as equal). The system is run live on the gold sentences, paragraph by paragraph.

Usage:
    python -m lgram.transition_eval [benchmark_data/en_transitions_gold.csv]
    python -m lgram.transition_eval --agree a.csv b.csv   # inter-annotator kappa
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import List, Optional

NONE = {"", "-", "none", "yok"}


def _norm(x: str) -> Optional[str]:
    x = x.strip().lower()
    return None if x in NONE else x


def derive(cp: Optional[str], cb: Optional[str], prev_cb: Optional[str]) -> str:
    if cb is None:
        return "NOCB"
    if prev_cb is None or cb == prev_cb:
        return "Continue" if cb == cp else "Retain"
    return "Smooth-Shift" if cb == cp else "Rough-Shift"


def _derived(rows: List[dict]) -> List[tuple]:
    """(id, gold Cb, gold transition) for every non-initial utterance."""
    out = []
    for _, group in groupby(rows, key=lambda r: r["para"]):
        prev_cb = None
        for r in group:
            if r["idx"] == "0":
                continue
            cp, cb = _norm(r["gold_cp"]), _norm(r["gold_cb"])
            out.append((r["id"], cb, derive(cp, cb, prev_cb)))
            prev_cb = cb
    return out


def kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / n**2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agree(path_a: Path, path_b: Path) -> int:
    da, db = (
        _derived(list(csv.DictReader(p.open(encoding="utf-8-sig"))))
        for p in (path_a, path_b)
    )
    assert [x[0] for x in da] == [
        x[0] for x in db
    ], "files do not cover the same utterances"
    ta, tb = [x[2] for x in da], [x[2] for x in db]
    na, nb = [x[1] is None for x in da], [x[1] is None for x in db]
    n = len(ta)
    print(
        f"n={n}  transition agreement={sum(x == y for x, y in zip(ta, tb)) / n:.3f}  kappa={kappa(ta, tb):.3f}"
    )
    print(
        f"NOCB agreement={sum(x == y for x, y in zip(na, nb)) / n:.3f}  kappa={kappa(na, nb):.3f}"
    )
    for (i, _, x), (_, _, y) in zip(da, db):
        if x != y:
            print(f"  id {i}: {x} vs {y}")
    return 0


def main(argv: List[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--agree"]:
        return agree(Path(argv[1]), Path(argv[2]))
    path = Path(argv[0] if argv else "benchmark_data/en_transitions_gold.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    todo = [r for r in rows if r["idx"] != "0" and not r["gold_cb"].strip()]
    if todo:
        print(f"{len(todo)} utterances still unannotated (first id {todo[0]['id']}).")
        return 1

    from .analyzer import TextAnalyzer

    ta = TextAnalyzer(model="en_core_web_md")
    confusion: Counter = Counter()
    nocb_hit = cb_exact = cb_both = 0
    n = 0
    for _, group in groupby(rows, key=lambda r: r["para"]):
        ct = ta._make_ct()
        prev_cb = None
        for r in group:
            st = ct.update_discourse(r["sentence"])
            if r["idx"] == "0":
                continue
            cp, cb = _norm(r["gold_cp"]), _norm(r["gold_cb"])
            gold = derive(cp, cb, prev_cb)
            prev_cb = cb
            confusion[(gold, st.transition.value)] += 1
            n += 1
            nocb_hit += (cb is None) == (st.backward_center is None)
            if cb and st.backward_center:
                cb_both += 1
                cb_exact += cb == st.backward_center.lower()

    labels = ["Continue", "Retain", "Smooth-Shift", "Rough-Shift", "NOCB"]
    acc = sum(v for (g, s), v in confusion.items() if g == s) / n
    print(f"n={n}  transition accuracy={acc:.3f}")
    print(f"Cb present/absent agreement={nocb_hit / n:.3f}")
    print(f"Cb exact string match (both present)={cb_exact}/{cb_both}")
    print("\ngold \\ system  " + " ".join(f"{s[:8]:>8s}" for s in labels))
    for g in labels:
        print(f"{g:14s} " + " ".join(f"{confusion[(g, s)]:8d}" for s in labels))
    return 0


if __name__ == "__main__":
    assert derive("king", "king", None) == "Continue"
    assert derive("queen", "king", "king") == "Retain"
    assert derive("queen", "queen", "king") == "Smooth-Shift"
    assert derive("frog", "queen", "king") == "Rough-Shift"
    assert derive("frog", None, "king") == "NOCB"
    assert kappa(["a", "b"], ["a", "b"]) == 1.0 and kappa(["a", "b"], ["b", "a"]) < 0
    sys.exit(main())
