"""
k-fold cross-validation: rule-based lgram.tr vs the trained coreference model, by
centering transition accuracy against Turkish-ITCC gold, over all public documents.

Per fold: test = every k-th document, validation (checkpoint selection) = 2 other
documents, train = the rest. Each fold's result is written to OUT/fold_k.json, so a
crashed run resumes where it stopped. The total is a paired McNemar test on
per-transition correctness.

Usage:
    python experiments/tr_coref_cv.py OUT_DIR PARSES.pkl FILE.conllu [FILE.conllu ...]
        [--folds 5] [--epochs 30] [--only FOLD]
PARSES.pkl = the DizgeBERT parses of FILE(s) in order (tr_baseline_itcc.py caches).
"""

import json
import os
import pickle
import subprocess
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tr_baseline_itcc as B  # noqa: E402
import tr_coref_data as D  # noqa: E402


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def split_docs(paths):
    """CoNLL-U text of each document, in file order (file-level comments dropped)."""
    docs = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines(keepends=True):
            if line.startswith("# newdoc"):
                docs.append([line])
            elif docs and not line.startswith("# global."):
                docs[-1].append(line)
    return ["".join(d) for d in docs]


def write_jsonl(conllu_text, path, tmp):
    tmp.write_text(conllu_text, encoding="utf-8")
    with open(path, "w", encoding="utf-8") as f:
        for d in D.convert(str(tmp)):
            for c in D.chunk(d, 400):
                f.write(json.dumps(c, ensure_ascii=False) + "\n")


def mcnemar_p(b, c):
    """Exact two-sided McNemar: b = only rules right, c = only model right."""
    n = b + c
    return (
        1.0
        if n == 0
        else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)
    )


def main():
    out, cache, files = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    files = [f for f in files if f.endswith(".conllu")]
    k, epochs = int(arg("--folds", 5)), arg("--epochs", "30")
    out.mkdir(parents=True, exist_ok=True)
    docs = split_docs(files)
    parses = pickle.loads(cache.read_bytes())
    header = "# global.Entity = eid-etype-head-other\n"
    n_sents = [d.count("# sent_id") for d in docs]
    assert sum(n_sents) == len(parses), (sum(n_sents), len(parses))
    offsets = [sum(n_sents[:i]) for i in range(len(docs))]
    print(f"{len(docs)} documents, {len(parses)} sentences, {k} folds", flush=True)

    only = arg("--only", None)
    for f in range(k):
        res_path = out / f"fold_{f}.json"
        if res_path.exists() or (only is not None and f != int(only)):
            continue
        test = list(range(f, len(docs), k))
        rest = [i for i in range(len(docs)) if i not in test]
        val, fit = rest[:2], rest[2:]
        fd = out / f"fold_{f}"
        fd.mkdir(exist_ok=True)
        text = lambda ids: header + "".join(docs[i] for i in ids)  # noqa: E731
        write_jsonl(text(fit), fd / "fit.jsonl", fd / "tmp.conllu")
        write_jsonl(text(val), fd / "val.jsonl", fd / "tmp.conllu")
        (fd / "test.conllu").write_text(text(test), encoding="utf-8")
        test_parses = [
            p for i in test for p in parses[offsets[i] : offsets[i] + n_sents[i]]
        ]
        print(f"fold {f}: test docs {test}, training {epochs} epochs...", flush=True)
        # the trainer saves a checkpoint at the first evaluation, so a killed run
        # leaves a half-trained model behind: only a finished run writes "trained"
        if not (fd / "trained").exists():
            subprocess.run(
                [sys.executable, str(HERE / "tr_coref_train.py"), str(fd / "fit.jsonl"),
                 str(fd / "val.jsonl"), str(fd / "model"), "--epochs", epochs,
                 "--eval-steps", "78", "--lr", "5e-5", "--head-lr", "3e-4",
                 "--cache", str(out / "cache")],
                check=True, env={**os.environ, "PYTHONUTF8": "1"},
                stdout=open(fd / "train.log", "w"), stderr=subprocess.STDOUT,
            )  # fmt: skip
            (fd / "trained").touch()

        import datasets
        from fastcoref import FCoref

        import tr_coref_centering as C

        datasets.disable_progress_bar()
        sents = list(B.sentences(str(fd / "test.conllu")))
        assert len(sents) == len(test_parses)
        rule_gold, rule_sys = B.evaluate(sents, test_parses)
        model = FCoref(model_name_or_path=str(fd / "model" / "model"), nlp=None,
                       enable_progress_bar=False)  # fmt: skip
        model_gold, model_sys = B.evaluate(
            sents, test_parses, make_identity=C.make_identity_factory(model, "lexical")
        )
        assert rule_gold == model_gold
        res = {
            "test_docs": test,
            "gold": rule_gold,
            "rule": rule_sys,
            "model": model_sys,
        }
        res_path.write_text(json.dumps(res), encoding="utf-8")
        del model

    if only is not None:
        return
    gold, rule, model = [], [], []
    print("\nfold  n     rule   model")
    for f in range(k):
        r = json.loads((out / f"fold_{f}.json").read_text(encoding="utf-8"))
        n = len(r["gold"])
        ra = sum(a == b for a, b in zip(r["gold"], r["rule"])) / n
        ma = sum(a == b for a, b in zip(r["gold"], r["model"])) / n
        print(f"{f:4d} {n:5d}  {ra:.3f}  {ma:.3f}")
        gold += r["gold"]
        rule += r["rule"]
        model += r["model"]
    rc = [g == s for g, s in zip(gold, rule)]
    mc = [g == s for g, s in zip(gold, model)]
    b = sum(r and not m for r, m in zip(rc, mc))
    c = sum(m and not r for r, m in zip(rc, mc))
    print(f" all {len(gold):5d}  {sum(rc) / len(gold):.3f}  {sum(mc) / len(gold):.3f}")
    print(
        f"McNemar: rules-only right {b}, model-only right {c}, p = {mcnemar_p(b, c):.2g}"
    )


if __name__ == "__main__":
    main()
