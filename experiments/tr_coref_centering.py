"""
Turkish centering with entity identity from a trained coreference model.

lgram.tr finds the mention slots (nouns, pronouns, dropped subjects, implicit
possessors); dropped subjects / possessors are inserted into the token stream as
pronoun placeholders ("o", "onun"), exactly as tr_coref_data.py did for training; the
model's clusters then give every slot its entity id. Scored with tr_baseline_itcc's
evaluator against Turkish-ITCC gold.

--fallback lexical: slots the model leaves unclustered keep lgram.tr's lexical /
speaker key (hybrid). --fallback none: they become singletons (model only).

Usage:
    python experiments/tr_coref_centering.py MODEL_DIR CACHE.pkl FILE.conllu
        [--fallback lexical|none]
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_baseline_itcc as B  # noqa: E402

from lgram.tr.centering import _base, analyze_parsed  # noqa: E402


def build_stream(parses):
    """Token stream the model reads, and where each mention slot sits in it.

    lgram.tr finds the slots; dropped subjects / implicit possessors are inserted
    as pronoun placeholders right after the token they hang on.
    Returns (stream, where): where[(i, "tok", id)] / where[(i, kind, pos)] -> index.
    """
    slots = []  # (sentence i, kind, pos, form)

    def record(i, m):
        slots.append((i, m.kind, m.pos, m.form))
        return None

    analyze_parsed([str(i) for i in range(len(parses))], parses, identity=record)
    stream, where = [], {}
    for i, toks in enumerate(parses):
        root = next((t.id for t in toks if t.head == 0), None)
        after = {}
        for j, kind, pos, form in slots:
            if j == i and kind in ("zero", "possessor"):
                anchor = root if pos == 0 else pos
                after.setdefault(anchor, []).append((kind, pos, form))
        for t in toks:
            where[(i, "tok", t.id)] = len(stream)
            stream.append(t.form)
            for kind, pos, form in sorted(
                after.get(t.id, []), key=lambda x: x[0] != "possessor"
            ):
                where[(i, kind, pos)] = len(stream)
                stream.append(form)
    return stream, where


def proper_names(parses, where):
    """Stream index -> name, for every proper-noun token ("Ayşe'yi" -> "ayşe")."""
    return {
        where[(i, "tok", t.id)]: _base(t.form)
        for i, toks in enumerate(parses)
        for t in toks
        if t.upos == "PROPN"
    }


def split_by_name(res, spans, names):
    """Group id per span, so that no group holds two different proper names.

    The model links mention to mention, so a pronoun that looks like both "Ali" and
    "Ayşe" welds their chains together. Name mentions found their group by name; every
    other mention joins the group of the mention it is most strongly linked to.
    """
    group = {s: names[s[1] - 1] for s in spans if s[1] - 1 in names}
    if len(set(group.values())) < 2:
        return {s: "" for s in spans}
    for s in sorted(spans):
        if s not in group:
            placed = [o for o in spans if o in group]
            group[s] = group[max(placed, key=lambda o: float(res.get_logit(s, o)))]
    return group


def cluster_words(model, stream, names=None):
    """Stream index -> cluster id, for the last word of every clustered span.

    `names` (from proper_names): split clusters that mix different proper names.
    """
    res = model.predict(texts=stream, is_split_into_words=True)
    # for pretokenized input fastcoref returns word spans [start, end)
    word_cluster = {}
    for ci, cluster in enumerate(res.get_clusters(as_strings=False)):
        spans = [s for s in cluster if s is not None]  # None: on a special token
        group = split_by_name(res, spans, names) if names else {}
        for span in spans:
            # Turkish NPs are head-final: the span's last word
            word_cluster[span[1] - 1] = f"C{ci}{group.get(span, '')}"
    return word_cluster


def slot_index(where, i, m):
    if m.kind in ("zero", "possessor"):
        return where.get((i, m.kind, m.pos))
    return where.get((i, "tok", m.pos))


def make_identity_factory(model, fallback, split_names=False):
    def make_identity(para):
        parses = [p for _, p in para]
        stream, where = build_stream(parses)
        names = proper_names(parses, where) if split_names else None
        word_cluster = cluster_words(model, stream, names)

        def identity(i, m):
            cid = word_cluster.get(slot_index(where, i, m))
            if cid or fallback == "lexical":
                # None keeps lgram.tr's own key (or leaves an anaphor unresolved)
                return cid
            return f"S{i}:{m.kind}:{m.pos}"  # model only: unclustered = singleton

        return identity

    return make_identity


def main(argv):
    model_dir, cache, conllu = argv[0], Path(argv[1]), argv[2]
    fallback = argv[argv.index("--fallback") + 1] if "--fallback" in argv else "lexical"
    import datasets
    from fastcoref import FCoref

    datasets.disable_progress_bar()

    model = FCoref(model_name_or_path=model_dir, nlp=None, enable_progress_bar=False)
    sents = list(B.sentences(conllu))
    parses = pickle.loads(cache.read_bytes())
    B.evaluate(sents, parses, make_identity=make_identity_factory(model, fallback))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
