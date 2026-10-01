"""
Baseline: how well does lgram.tr (DizgeBERT-Joint + rule-based zeros) recover the
centering structure that Turkish-ITCC gold coreference implies?

ITCC's own tokenization is fed to the parser, so every system mention can be mapped to
a gold entity: overt nouns/pronouns by token, zero subjects via the empty nSubj node on
the root verb, implicit possessors via the empty dPsor node on the possessed noun.
Gold Cf ranking: grammatical role as lgram.tr ranks it, zero before overt.

Usage:
    python experiments/tr_baseline_itcc.py CACHE.pkl FILE.conllu [FILE.conllu ...]
"""

import pickle
import sys
from collections import Counter
from pathlib import Path

import udapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lgram.models.centering_theory import TransitionType as TT  # noqa: E402
from lgram.tr.centering import analyze_parsed  # noqa: E402
from lgram.tr.parser import Token, parse_feats  # noqa: E402
from lgram.transition_eval import derive, kappa  # noqa: E402

SUBJ = ("nsubj", "csubj")


def gold_role(mt, node):
    """subj > obj > iobj > obl > possessor/other, as lgram.tr ranks. On ITCC every
    role-based order scores the same Cp->next-Cb (0.678-0.679, itcc_centering.py), so
    the gold uses the system's order rather than penalise an arbitrary choice."""
    if mt == "nSubj":
        return 0
    if mt == "dPsor":
        return 4
    d = (node.deprel or "").split(":")[0]
    if d in SUBJ:
        return 0
    return {"obj": 1, "iobj": 2, "obl": 3}.get(d, 4)


def sentences(path):
    """Per sentence: surface forms, surface->word ords, gold mentions, paragraph start."""
    doc = udapi.Document(path)
    by_root = {}
    for e in doc.coref_entities:
        for m in e.mentions:
            mt = (
                m.other.get("mentionType", "overt")
                if hasattr(m.other, "get")
                else "overt"
            )
            by_root.setdefault(id(m.head.root), []).append((e.eid, mt, m.head))
    for b in doc.bundles:
        root = b.trees[0]
        forms, ords = [], []
        for tok in root.token_descendants:
            forms.append(tok.form)
            ws = tok.words if hasattr(tok, "words") else [tok]
            ords.append({int(w.ord) for w in ws})
        yield {
            "newpar": bool(root.newdoc or root.newpar),
            "forms": forms,
            "ords": ords,
            "mentions": by_root.get(id(root), []),
        }


def anchor(node):
    """Word ord an empty node hangs on (from its enhanced deps)."""
    return int(node.deps[0]["parent"].ord) if node.deps else None


def evaluate(sents, parses, make_identity=None):
    """`make_identity(para)` -> identity(i, mention) lets a coref model set entity
    keys (see tr_coref_centering.py); None = lgram.tr's rule-based resolution."""
    paras, cur = [], []
    for s, p in zip(sents, parses):
        if s["newpar"] and cur:
            paras.append(cur)
            cur = []
        cur.append((s, p))
    if cur:
        paras.append(cur)

    gold_t, sys_t = [], []
    cb_pres = cb_same = cb_both = 0
    zero = Counter()  # detection / resolution
    psor = Counter()
    miss = Counter()  # gold Cb not recovered, by how it is realized in U_i
    for para in paras:
        report = analyze_parsed(
            [" ".join(s["forms"]) for s, _ in para],
            [p for _, p in para],
            identity=make_identity(para) if make_identity else None,
        )
        if len(report.utterances) != len(para):  # empty parse dropped a sentence
            continue
        prev_gold_cf, prev_gold_cb, prev_sys_map = None, None, None
        for (s, toks), u in zip(para, report.utterances):
            overt = {}
            n_subj, d_psor = {}, {}
            for eid, mt, node in s["mentions"]:
                if mt == "overt":
                    overt.setdefault(int(node.ord), eid)
                elif mt == "nSubj":
                    n_subj.setdefault(anchor(node), eid)
                else:
                    d_psor.setdefault(anchor(node), eid)
            root = next((t for t in toks if t.head == 0), None)
            root_ords = s["ords"][root.id - 1] if root else set()

            prev_gold_ids = set(prev_gold_cf or [])
            # system mention key -> gold eid, via token position
            sys_map = {}
            for m in u.cf:
                if m.kind == "zero":  # pos 0 = the root's zero, else its predicate
                    ords = root_ords if m.pos == 0 else s["ords"][m.pos - 1]
                    eid = next((n_subj[o] for o in ords if o in n_subj), None)
                elif m.kind == "possessor":
                    ords = s["ords"][m.pos - 1]
                    eid = next((d_psor[o] for o in ords if o in d_psor), None)
                else:
                    ords = s["ords"][m.pos - 1]
                    eid = next((overt[o] for o in ords if o in overt), None)
                sys_map[m.key] = eid

            # zero subject on the root: detection and resolution. ITCC also adds an
            # nSubj node when the subject is overt (agreement doubling, "Osman geldi");
            # that is not a dropped subject.
            gold_zero = next((n_subj[o] for o in root_ords if o in n_subj), None)
            overt_subj = {
                eid
                for eid, mt, node in s["mentions"]
                if mt == "overt"
                and (node.deprel or "").split(":")[0] in SUBJ
                and node.parent is not None
                and int(node.parent.ord) in root_ords
            }
            if gold_zero in overt_subj:
                gold_zero = None
            zero[("gold", gold_zero is not None, "sys", u.zero_detected)] += 1
            for zm in (m for m in u.cf if m.kind == "zero"):
                g = sys_map.get(zm.key)
                if g is not None and prev_sys_map is not None:
                    who = "1/2" if zm.key.startswith("@") else "3"
                    zero["resolved"] += 1
                    zero["resolved_ok"] += prev_sys_map.get(zm.key) == g
                    zero[who] += 1
                    zero[who + "_ok"] += prev_sys_map.get(zm.key) == g
                    zero[who + "_ante_absent"] += g not in prev_gold_ids
            for m in u.cf:
                if m.kind == "possessor" and prev_sys_map is not None:
                    g = sys_map.get(m.key)
                    if g is not None:
                        psor["resolved"] += 1
                        psor["resolved_ok"] += prev_sys_map.get(m.key) == g

            best = {}
            for eid, mt, node in s["mentions"]:
                k = (gold_role(mt, node), mt == "overt", float(node.ord))
                if eid not in best or k < best[eid]:
                    best[eid] = k
            gold_cf = sorted(best, key=best.get)
            if prev_gold_cf is not None:
                g_cb = next((e for e in prev_gold_cf if e in gold_cf), None)
                g_cp = gold_cf[0] if gold_cf else None
                gold_t.append(derive(g_cp, g_cb, prev_gold_cb))
                prev_gold_cb = g_cb
                st = "NOCB" if u.cb is None else u.transition.value
                sys_t.append(st)
                cb_pres += (g_cb is None) == (u.cb is None)
                if g_cb is not None and prev_sys_map.get(u.cb) != g_cb:
                    types = {mt for e, mt, _ in s["mentions"] if e == g_cb}
                    miss["+".join(sorted(types))] += 1
                if g_cb is not None and u.cb is not None:
                    cb_both += 1
                    cb_same += prev_sys_map.get(u.cb) == g_cb
            prev_gold_cf, prev_sys_map = gold_cf, sys_map
        assert all(t != TT.ESTABLISH.value for t in sys_t)

    n = len(gold_t)
    acc = sum(a == b for a, b in zip(gold_t, sys_t)) / n
    print(f"transitions n={n}  accuracy={acc:.3f}  kappa={kappa(gold_t, sys_t):.3f}")
    print(f"Cb present/absent agreement={cb_pres / n:.3f}")
    print(f"Cb is the right gold entity (both present)={cb_same}/{cb_both}"
          f" = {cb_same / max(cb_both, 1):.3f}")  # fmt: skip
    tp = zero[("gold", True, "sys", True)]
    fp = zero[("gold", False, "sys", True)]
    fn = zero[("gold", True, "sys", False)]
    print(f"root zero subject detection: P={tp / max(tp + fp, 1):.3f}"
          f" R={tp / max(tp + fn, 1):.3f} (gold={tp + fn})")  # fmt: skip
    print(f"zero subject resolution: {zero['resolved_ok']}/{zero['resolved']}")
    for who in ("1/2", "3"):
        print(f"  person {who}: {zero[who + '_ok']}/{zero[who]}"
              f"  (gold referent absent from U_i-1: {zero[who + '_ante_absent']})")  # fmt: skip
    print(
        f"possessor -> previous-sentence resolution: {psor['resolved_ok']}/{psor['resolved']}"
    )
    labels = ["Continue", "Retain", "Smooth-Shift", "Rough-Shift", "NOCB"]
    c = Counter(zip(gold_t, sys_t))
    print("gold \\ system  " + " ".join(f"{x[:8]:>8s}" for x in labels))
    for a in labels:
        print(f"{a:14s} " + " ".join(f"{c[(a, b)]:8d}" for b in labels))
    print("gold dist:", Counter(gold_t))
    print("gold Cb missed/wrong, realized in U_i as:", dict(miss.most_common()))


def main(argv):
    cache, files = Path(argv[0]), argv[1:]
    sents = [s for f in files for s in sentences(f)]
    if cache.exists():
        parses = pickle.loads(cache.read_bytes())
    else:
        from lgram.tr.parser import JointParser

        jp = JointParser()
        parses = []
        for i, s in enumerate(sents):
            rows = jp._model.predict(s["forms"], scheme=jp.scheme, tokenizer=jp._tok)
            parses.append(
                [Token(j, f, u, parse_feats(ft), int(h), d)
                 for j, (f, u, _x, ft, h, d) in enumerate(rows, 1)]  # fmt: skip
            )
            if i % 500 == 0:
                print(f"parsed {i}/{len(sents)}", flush=True)
        cache.write_bytes(pickle.dumps(parses))
    evaluate(sents, parses)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
