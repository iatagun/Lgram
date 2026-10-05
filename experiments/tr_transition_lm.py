"""
A language model over centering transitions, next to a language model over words.

The idea under test: a statistical language model writes a good sentence but loses the
thread across sentences; let an n-gram model of one author's transition types (devam,
yumuşak dönüş, ...) say which sentence should come next. Before any text is generated,
two things must hold, and this script measures them on one author's stories:

  label   transition of every sentence (lgram.tr slots, coreference model identity)
  ngram   is the next transition predictable from the previous ones? (perplexity of a
          1-, 2-, 3-gram model, stories held out)
  select  given four sentences, pick the real next one among ten (nine others from the
          same story): word model alone vs word model + transition model

    python experiments/tr_transition_lm.py label  STORIES.conllu COREF_MODEL LABELS.json
    python experiments/tr_transition_lm.py ngram  LABELS.json [STORIES.conllu]
        (with the stories: also a log-linear model that sees the previous sentence)
    python experiments/tr_transition_lm.py select STORIES.conllu COREF_MODEL LABELS.json OUT.json
        [--items 600] [--lm ytu-ce-cosmos/turkish-gpt2]
    python experiments/tr_transition_lm.py report OUT.json

STORIES.conllu: tr_silver.py output (DizgeBERT parses; its empty nodes are ignored).
"""

import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

KINDS = ("transition", "move")  # Brennan et al.'s five | the lecture notes' four
FOLDS, CONTEXT = 5, 4


def read_docs(path):
    """doc id -> [(text, [Token])], in order."""
    from lgram.tr.parser import Token, parse_feats

    docs, doc, text, toks = {}, None, "", []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if line.startswith("# newdoc id = "):
            doc = docs.setdefault(line[14:], [])
        elif line.startswith("# text = "):
            text = line[9:]
        elif line and line[0].isdigit():
            c = line.split("\t")
            if c[0].isdigit():  # not an empty node "5.1"
                toks.append(Token(int(c[0]), c[1], c[3], parse_feats(c[5]), int(c[6]), c[7]))  # fmt: skip
        elif not line and toks:
            doc.append((text, toks))
            toks = []
    return docs


def chunks(sents, max_tokens=300):
    """Runs of sentences the coreference model reads at once (it was trained on such)."""
    out, size = [[]], 0
    for s in sents:
        if size + len(s[1]) > max_tokens and out[-1]:
            out.append([])
            size = 0
        out[-1].append(s)
        size += len(s[1])
    return out


def load_coref(model_dir):
    import datasets
    from fastcoref import FCoref

    from tr_coref_centering import make_identity_factory

    datasets.disable_progress_bar()
    logging.disable(logging.INFO)  # fastcoref logs two lines per call
    model = FCoref(model_name_or_path=model_dir, nlp=None, enable_progress_bar=False)
    return make_identity_factory(model, "lexical")


def labels_of(sents, make_identity):
    """[transition, move] per sentence; the first has no previous sentence."""
    from lgram.tr.centering import analyze_parsed

    rep = analyze_parsed([t for t, _ in sents], [p for _, p in sents],
                         identity=make_identity(sents))  # fmt: skip
    return [[u.transition.value, u.topic_move or "-"] for u in rep.utterances]


def label(conllu, model_dir, out):
    make_identity = load_coref(model_dir)
    res = {}
    for n, (doc, sents) in enumerate(read_docs(conllu).items()):
        res[doc] = [labels_of(c, make_identity) for c in chunks(sents)]
        print(f"{n + 1} {doc} {len(sents)} sentences", flush=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


class NGram:
    """Add-alpha n-gram over a handful of symbols (5 transition types: counts are dense)."""

    def __init__(self, seqs, n, alpha=0.5):
        self.n, self.alpha, self.c = n, alpha, defaultdict(Counter)
        self.vocab = sorted({s for q in seqs for s in q})
        for q in seqs:
            for i, s in enumerate(q):
                self.c[self.ctx(q[:i])][s] += 1

    def ctx(self, hist):
        return tuple((["<s>"] * self.n + list(hist))[len(hist) + 1 :]) if self.n > 1 else ()

    def logp(self, hist, sym):
        c = self.c[self.ctx(hist)]
        return math.log((c[sym] + self.alpha) / (sum(c.values()) + self.alpha * len(self.vocab)))  # fmt: skip


def fold_of(docs):
    return {d: i % FOLDS for i, d in enumerate(sorted(docs))}


def seqs(labels, docs, k):
    # a chunk's first sentence has no transition: the sequence starts at the second
    return [[x[k] for x in chunk[1:]] for d in docs for chunk in labels[d] if len(chunk) > 1]


def ngram(labels_path):
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    fold = fold_of(labels)
    for k, kind in enumerate(KINDS):
        dist = Counter(s for q in seqs(labels, labels, k) for s in q)
        total = sum(dist.values())
        print(f"\n{kind}: {total} transitions  " + "  ".join(f"{s} {c / total:.0%}" for s, c in dist.most_common()))  # fmt: skip
        for n in (1, 2, 3, 4):
            ll = cnt = 0
            for f in range(FOLDS):
                lm = NGram(seqs(labels, [d for d in labels if fold[d] != f], k), n)
                for q in seqs(labels, [d for d in labels if fold[d] == f], k):
                    ll += sum(lm.logp(q[:i], s) for i, s in enumerate(q))
                    cnt += len(q)
            print(f"  {n}-gram  perplexity {math.exp(-ll / cnt):.3f}  (stories held out)")


def features(labs, sents, i, k):
    """What is known before sentence i is written: the two transitions before it and
    the look of the sentence before it. A CRF's features, used left to right (MEMM)."""
    t1, t2 = labs[i - 1][k] if i > 1 else "<s>", labs[i - 2][k] if i > 2 else "<s>"
    text, toks = sents[i - 1]
    return {
        "t1=" + t1: 1, "t2=" + t2: 1, f"t={t2}>{t1}": 1,
        "dialogue": text.lstrip()[:1] in "—–-̶", "question": text.rstrip(" \"”’").endswith("?"),
        "exclaim": text.rstrip(" \"”’").endswith("!"), "colon": text.rstrip().endswith(":"),
        f"len={min(len(toks) // 6, 5)}": 1,
        "speaker": any(t.feats.get("Person") in ("1", "2") for t in toks),
        "name": any(t.upos == "PROPN" for t in toks),
        "pronoun": any(t.upos == "PRON" for t in toks),
        f"t1={t1}&dialogue": text.lstrip()[:1] in "—–-̶",
    }  # fmt: skip


def loglinear(labels_path, conllu):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    docs, fold = read_docs(conllu), fold_of(labels)
    for k, kind in enumerate(KINDS):
        rows = [(fold[d], features(labs, c, i, k), labs[i][k])
                for d in labels for labs, c in zip(labels[d], chunks(docs[d]))
                for i in range(1, len(labs))]  # fmt: skip
        for name, keep in (("previous transitions only", lambda f: f.startswith("t") and "&" not in f),
                           ("+ the previous sentence's look", lambda f: True)):  # fmt: skip
            ll = 0
            for f in range(FOLDS):
                pick = lambda x: {a: b for a, b in x.items() if keep(a)}  # noqa: E731
                vec = DictVectorizer()
                X = vec.fit_transform([pick(x) for g, x, _ in rows if g != f])
                clf = LogisticRegression(max_iter=2000).fit(X, [y for g, _, y in rows if g != f])
                test = [(x, y) for g, x, y in rows if g == f]
                P = clf.predict_log_proba(vec.transform([pick(x) for x, _ in test]))
                col = {c: j for j, c in enumerate(clf.classes_)}
                # a type never seen in training: charge the smallest probability given
                ll += sum(p[col[y]] if y in col else p.min() for p, (_, y) in zip(P, test))
            print(f"{kind}: log-linear, {name:32} perplexity {math.exp(-ll / len(rows)):.3f}")


def select(conllu, model_dir, labels_path, out, items=600, lm_name="ytu-ce-cosmos/turkish-gpt2"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    docs = read_docs(conllu)
    make_identity = load_coref(model_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(lm_name)
    gpt = AutoModelForCausalLM.from_pretrained(lm_name).to(dev).eval()

    @torch.no_grad()
    def logp(prefix, text):
        """log P(text | prefix) under the word model."""
        a, b = tok.encode(prefix), tok.encode(" " + text)
        ids = torch.tensor([(a + b)[-1000:]], device=dev)
        lp = torch.log_softmax(gpt(ids).logits[0, :-1].float(), -1)
        tgt = ids[0, 1:]
        return lp[torch.arange(len(tgt)), tgt][-len(b) :].sum().item()

    ok = lambda s: 4 <= len(s[1]) <= 40  # noqa: E731  a sentence, not a title or a list
    rng = random.Random(0)
    spots = []
    for d, sents in docs.items():
        pool = [s for s in sents if ok(s)]
        start = 0
        for c in chunks(sents):
            for j in range(CONTEXT, len(c)):
                if ok(c[j]) and len(pool) > 30:
                    spots.append((d, start + j))
            start += len(c)
    rng.shuffle(spots)
    res = []
    for n, (d, j) in enumerate(spots[:items]):
        sents = docs[d]
        ctx = sents[j - CONTEXT : j]
        near = {id(s) for s in sents[max(0, j - CONTEXT) : j + 1]}
        others = rng.sample([i for i, s in enumerate(sents) if ok(s) and id(s) not in near], 9)
        prefix = " ".join(t for t, _ in ctx)
        cands = []
        for i in [j] + others:  # the real one first
            s = sents[i]
            labs = labels_of(ctx + [s], make_identity)
            # the prior after a bare full stop: this model was never shown a start token
            # (after one, its first prediction costs 12 000 nats and drowns everything)
            cands.append({"sent": i, "cond": logp(prefix, s[0]), "prior": logp(".", s[0]),
                          "tokens": len(s[1]), "labels": labs[1:]})  # fmt: skip
        res.append({"doc": d, "sent": j, "cands": cands})
        if (n + 1) % 25 == 0:
            print(n + 1, flush=True)
            Path(out).write_text(json.dumps({"labels": labels_path, "items": res}), encoding="utf-8")  # fmt: skip
    Path(out).write_text(json.dumps({"labels": labels_path, "items": res}), encoding="utf-8")
    report(out)


def report(out):
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    labels = json.loads(Path(data["labels"]).read_text(encoding="utf-8"))
    items, fold = data["items"], fold_of(labels)
    lms = {(f, k): NGram(seqs(labels, [d for d in labels if fold[d] != f], k), 3)
           for f in range(FOLDS) for k in range(len(KINDS))}  # fmt: skip

    # what a sentence drawn at random from the story does after the same context: the
    # author's model alone rewards the commonest type (no link), which random sentences
    # show even more often than the real next one
    noise = defaultdict(Counter)
    for it in items:
        for c in it["cands"][1:]:
            for k in range(len(KINDS)):
                q = [x[k] for x in c["labels"]]
                noise[fold[it["doc"]], k, tuple(q[-3:-1])][q[-1]] += 1

    def trans(it, c, k):  # log P(candidate's transition | the context's), other stories' model
        q = [x[k] for x in c["labels"]]
        return lms[fold[it["doc"]], k].logp(q[:-1], q[-1])

    def ratio(it, c, k):  # ... against the same for a random sentence (other folds' items)
        q = [x[k] for x in c["labels"]]
        cnt = sum((noise[f, k, tuple(q[-3:-1])] for f in range(FOLDS) if f != fold[it["doc"]]), Counter())  # fmt: skip
        return trans(it, c, k) - math.log((cnt[q[-1]] + 0.5) / (sum(cnt.values()) + 2.5))

    def rank(it, score):  # the real sentence's rank; ties count half
        s = [score(it, c) for c in it["cands"]]
        return 1 + sum(x > s[0] for x in s[1:]) + sum(x == s[0] for x in s[1:]) / 2

    def acc(its, score):  # candidates tied at the top share the point
        got = 0
        for it in its:
            s = [score(it, c) for c in it["cands"]]
            got += (s[0] == max(s)) / s.count(max(s))
        return got / len(its)

    pmi = lambda it, c: c["cond"] - c["prior"]  # noqa: E731
    print(f"{len(items)} items, 10 candidates each (chance 0.100)")
    print(f"word model, log P(sentence | context)        {acc(items, lambda it, c: c['cond']):.3f}")
    print(f"word model, minus log P(sentence) (PMI)      {acc(items, pmi):.3f}")
    grid = (0.25, 0.5, 1, 2, 4, 8, 16)
    for k, kind in enumerate(KINDS):
        real = Counter(it["cands"][0]["labels"][-1][k] for it in items)
        fake = Counter(c["labels"][-1][k] for it in items for c in it["cands"][1:])
        print(f"\n{kind}: real next / random sentence  " + "  ".join(
            f"{s} {real[s] / len(items):.0%}/{fake[s] / len(items) / 9:.0%}" for s, _ in fake.most_common()))  # fmt: skip
        for name, fn in (("author's model", trans), ("author's model / random", ratio)):
            print(f"  {name:24} alone {acc(items, lambda it, c: fn(it, c, k)):.3f}", end="")
            # weight chosen on the other folds' items, then applied to this fold's
            base = plus = wins = losses = 0
            for f in range(FOLDS):
                rest = [it for it in items if fold[it["doc"]] != f]
                w = max(grid, key=lambda w: acc(rest, lambda it, c: pmi(it, c) + w * fn(it, c, k)))
                for it in (it for it in items if fold[it["doc"]] == f):
                    a = rank(it, pmi) == 1
                    b = rank(it, lambda it, c: pmi(it, c) + w * fn(it, c, k)) == 1
                    base, plus = base + a, plus + b
                    wins, losses = wins + (b and not a), losses + (a and not b)
            n = wins + losses
            p = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** (n - 1) if n else 1
            print(f"   with the word model {base / len(items):.3f} -> {plus / len(items):.3f}"
                  f"  (gained {wins}, lost {losses}, sign test p={min(1, p):.3f})")  # fmt: skip


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda name, d, t=str: t(a[a.index(name) + 1]) if name in a else d  # noqa: E731
    if a[:1] == ["label"] and len(a) == 4:
        label(a[1], a[2], a[3])
    elif a[:1] == ["ngram"] and len(a) in (2, 3):
        ngram(a[1])
        if len(a) == 3:
            loglinear(a[1], a[2])
    elif a[:1] == ["select"] and len(a) >= 5:
        select(a[1], a[2], a[3], a[4], opt("--items", 600, int), opt("--lm", "ytu-ce-cosmos/turkish-gpt2"))  # fmt: skip
    elif a[:1] == ["report"] and len(a) == 2:
        report(a[1])
    else:
        sys.exit(__doc__)
