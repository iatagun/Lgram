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

from lgram.tr.centering import analyze_parsed  # noqa: E402


def make_identity_factory(model, fallback):
    def make_identity(para):
        slots = []  # (sentence i, kind, pos, form)

        def record(i, m):
            slots.append((i, m.kind, m.pos, m.form))
            return None

        analyze_parsed(
            [" ".join(s["forms"]) for s, _ in para],
            [p for _, p in para],
            identity=record,
        )
        # token stream with placeholders right after the token they hang on
        stream, where = [], {}
        for i, (s, toks) in enumerate(para):
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
        res = model.predict(texts=stream, is_split_into_words=True)
        # for pretokenized input fastcoref returns word spans [start, end)
        word_cluster = {}
        for ci, cluster in enumerate(res.get_clusters(as_strings=False)):
            for span in cluster:
                if span is not None:  # None: mention on a special token
                    # Turkish NPs are head-final: the span's last word
                    word_cluster[span[1] - 1] = f"C{ci}"

        def identity(i, m):
            idx = (
                where.get((i, m.kind, m.pos))
                if m.kind in ("zero", "possessor")
                else where.get((i, "tok", m.pos))
            )
            cid = word_cluster.get(idx) if idx is not None else None
            if cid or fallback == "lexical":
                return (
                    cid  # None keeps lgram.tr's own key (or leaves anaphor unresolved)
                )
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
