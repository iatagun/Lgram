"""
Score lgram.tr against the hand labels of tr_label.py (finished texts only).

For every dropped subject, implicit possessor and pronoun the annotator answered: did
the system put it in one entity with a mention of its gold referent? An answer "not in
the text" is right when the system links the slot to nothing. With MODEL_DIR the entity
ids come from the coreference model, lgram.tr's own key where the model has none.

Not measured: slots the system misses (the tool only asks about the ones it finds), and
the centre of attention, until texts carry "focus" answers.

    python experiments/tr_label_score.py BATCH.json [MODEL_DIR] [-v]
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lgram.tr.centering as C  # noqa: E402
from lgram.tr.parser import Token  # noqa: E402

ASKED = ("zero", "possessor", "pronoun")


def node(sid):
    """Label id -> (sentence, kind, pos); a noun or pronoun is the token it sits on."""
    i, kind, pos = sid.split(":")
    return int(i), "tok" if kind in ("noun", "pronoun") else kind, int(pos)


def rule_keys(parses):
    """node -> lgram.tr's entity key (None: left unresolved, or not a mention)."""
    got, real = [], C.extract_mentions

    def spy(*a, **k):  # analyze_parsed keeps only the merged Cf; we need every mention
        got.append(real(*a, **k))
        return got[-1]

    with mock.patch.object(C, "extract_mentions", spy):
        C.analyze_parsed([""] * len(parses), parses)
    keys = {}
    for i, toks in enumerate(parses):
        for t in toks:
            if t.upos in ("NOUN", "PROPN"):  # also the nouns lgram.tr does not rank
                keys[(i, "tok", t.id)] = C.entity_key(t)
        for m in got[i][0]:
            keys[node(f"{i}:{m.kind}:{m.pos}")] = m.key
    return keys


def model_keys(model, parses, rules):
    from tr_coref_centering import build_stream, cluster_words

    stream, where = build_stream(parses)
    cluster = cluster_words(model, stream)
    keys = dict(rules)
    keys.update({n: cluster[k] for n, k in where.items() if k in cluster})
    return keys


def gold(label):
    """(find, members of each chain, the several referents of a plural slot)."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    several = {}
    for sid, a in label["answers"].items():
        to = [node(t) for t in a.get("to", [])]
        if len(to) == 1:
            parent[find(node(sid))] = find(to[0])
        elif to:  # "ayı + tilki": its own chain, the members stay out of it
            several[node(sid)] = to
    for a, b in label["links"]:
        parent[find(node(a))] = find(node(b))
    chains = defaultdict(set)
    for x in list(parent):
        chains[find(x)].add(x)
    return find, chains, several


def score(label, keys):
    """One (slot id, kind, verdict) per answered question."""
    find, chains, several = gold(label)
    used = Counter(k for k in keys.values() if k)
    for sid, a in label["answers"].items():
        kind, n = sid.split(":")[1], node(sid)
        if kind not in ASKED:
            continue
        k = keys.get(n)
        alone = not k or used[k] == 1
        if "bogus" in a:
            yield sid, kind, "bogus"
        elif "none" in a:
            yield sid, kind, "none ok" if alone else "none linked"
        else:
            partners = set(chains[find(n)]) - {n}
            for t in several.get(n, []):  # any of the several referents will do
                partners |= chains[find(t)] | {t}
            ok = k and any(keys.get(p) == k for p in partners)
            yield sid, kind, "ok" if ok else "unlinked" if alone else "wrong"


def main(argv):
    batch = Path(argv[0])
    model_dir = next((a for a in argv[1:] if not a.startswith("-")), None)
    windows = json.loads(batch.read_text(encoding="utf-8"))
    labels = json.loads(batch.with_suffix(".labels.json").read_text(encoding="utf-8"))
    done = [w for w in windows if labels.get(w["id"], {}).get("done")]
    systems = ["rules"]
    if model_dir:
        import datasets
        from fastcoref import FCoref

        datasets.disable_progress_bar()
        model = FCoref(model_name_or_path=model_dir, nlp=None, enable_progress_bar=False)
        systems.append("model")

    tally = {s: Counter() for s in systems}
    for w in done:
        label = labels[w["id"]]
        parses = [[Token(**t) for t in s["tokens"]] for s in w["sentences"]]
        find = gold(label)[0]
        # the scorer's own check: the gold chains, scored as a system, make no mistake
        oracle = {node(s): find(node(s)) for s in label["answers"]}
        oracle.update({(i, "tok", t.id): find((i, "tok", t.id)) for i, p in enumerate(parses) for t in p})  # fmt: skip
        assert all(v in ("ok", "none ok", "bogus") or len(label["answers"][s].get("to", [])) > 1
                   for s, _, v in score(label, oracle)), w["id"]  # fmt: skip
        rules = rule_keys(parses)
        for name in systems:
            keys = rules if name == "rules" else model_keys(model, parses, rules)
            for sid, kind, verdict in score(label, keys):
                for row in (kind, "all", "genre " + w["genre"]):
                    tally[name][row, verdict] += 1
                if "-v" in argv and verdict not in ("ok", "none ok"):
                    i, _, pos = sid.split(":")
                    toks = w["sentences"][int(i)]["tokens"]
                    at = toks[int(pos) - 1]["form"] if int(pos) else "(root)"
                    print(f"{name:5} {w['id']} {int(i) + 1}. {kind} @{at}: {verdict}")

    print(f"\n{len(done)} finished texts of {len(windows)}")
    for name in systems:
        print(f"\n{name}\n{'':18}{'referent in text':>22}{'not in text':>16}{'bogus':>7}")
        print(f"{'':18}{'right':>7}{'unlinked':>9}{'wrong':>6}{'left alone':>16}")
        for row in sorted({r for r, _ in tally[name]}, key=lambda r: (r.startswith("genre"), r == "all", r)):  # fmt: skip
            t = lambda v: tally[name][row, v]  # noqa: E731
            n, m = t("ok") + t("unlinked") + t("wrong"), t("none ok") + t("none linked")
            pct = f"{t('ok') / n:.0%}" if n else "-"
            print(f"{row:14}{pct:>5}{t('ok'):>4}/{n:<3}{t('unlinked'):>8}{t('wrong'):>6}"
                  f"{t('none ok'):>12}/{m:<3}{t('bogus'):>7}")  # fmt: skip
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
