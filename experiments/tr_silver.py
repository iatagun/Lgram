"""
Silver coreference data for Turkish: raw text -> CoNLL-U a teacher model can label.

ITCC has 19 training documents; that is the bottleneck. This script makes more: it
fetches Turkish Wikipedia articles, parses them, and writes CoNLL-U whose empty nodes
(dropped subjects, implicit possessors) come from lgram.tr, plus ITCC's agreement
doubling (a subject node on a finite verb with an overt subject, a possessor node next
to an overt genitive) so the result looks like ITCC to the teacher and to
tr_coref_data.py. A teacher (CorPipe) then adds Entity annotations; tr_coref_data.py
turns its output into fastcoref training lines.

    python experiments/tr_silver.py fetch DOCS.jsonl --docs 400
    python experiments/tr_silver.py fetch-category DOCS.jsonl CATEGORY --site tr.wikisource.org
        [--min-words 300]   (news items are short: tr.wikinews.org needs about 100)
    python experiments/tr_silver.py conllu DOCS.jsonl OUT.conllu

fetch takes random long Wikipedia articles (expository prose); fetch-category takes every
page of a category, e.g. public-domain stories on Wikisource (narrative, dialogue).
Wiki text is CC BY-SA; DOCS.jsonl keeps title and revision id for attribution.
Pages overlapping benchmark_data/tr_*.txt (the out-of-domain tests) are skipped.
"""

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SITE = "tr.wikipedia.org"
UA = {"User-Agent": "centering-lgram-research (https://github.com/iatagun/Lgram)"}


def api(**params):
    url = f"https://{SITE}/w/api.php?" + urllib.parse.urlencode({"format": "json", **params})  # fmt: skip
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception as e:  # network hiccup or rate limit: wait and retry
            print("retry:", e, file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return {}


def paragraphs(extract, prose_only=True):
    paras = [re.sub(r"\[[^\]]*\]|\([^)]*\)", "", x) for x in extract.split("\n")]
    paras = [re.sub(r"\s+", " ", x).strip() for x in paras]
    if not prose_only:  # stories: keep short dialogue lines, drop only headings
        return [p for p in paras if p and not p.startswith("=")]
    # running prose only: headings, lists and table debris are short or unpunctuated
    return [p for p in paras if len(p.split()) >= 40 and p.count(". ") >= 2]


def page_doc(pageid, prose_only, max_words, min_words=300):
    """One page as a document, or None if it is too short or overlaps a test set."""
    d = api(action="query", prop="extracts|revisions", rvprop="ids",
            explaintext=1, pageids=pageid)  # fmt: skip
    page = d.get("query", {}).get("pages", {}).get(str(pageid), {})
    keep, words = [], 0
    for para in paragraphs(page.get("extract", ""), prose_only):
        if words >= max_words:
            break
        keep.append(para)
        words += len(para.split())
    held_out = "".join(f.read_text(encoding="utf-8")
                       for f in (ROOT / "benchmark_data").glob("tr_*.txt"))  # fmt: skip
    if words < min_words or any(len(k) >= 60 and k[:60] in held_out for k in keep):
        return None
    rev = page.get("revisions", [{}])[0].get("revid")
    return {"id": pageid, "title": page["title"], "revid": rev, "site": SITE,
            "words": words, "paragraphs": keep}  # fmt: skip


def fetch_category(out, category, max_words=3000, min_words=300):
    cont = {}
    with open(out, "a", encoding="utf-8") as f:
        while True:
            d = api(action="query", list="categorymembers", cmtitle=category,
                    cmnamespace=0, cmlimit=500, **cont)  # fmt: skip
            for m in d.get("query", {}).get("categorymembers", []):
                doc = page_doc(m["pageid"], False, max_words, min_words)
                if doc is not None:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    print(doc["title"], doc["words"], flush=True)
                time.sleep(0.2)
            if "continue" not in d:
                break
            cont = d["continue"]


def fetch(out, n_docs, min_bytes=15000, max_words=1200):
    done = set()
    if Path(out).exists():  # resumable
        done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")}
    with open(out, "a", encoding="utf-8") as f:
        while len(done) < n_docs:
            pages = api(action="query", generator="random", grnnamespace=0,
                        grnlimit=50, prop="info")  # fmt: skip
            for p in pages.get("query", {}).get("pages", {}).values():
                if p["length"] < min_bytes or p["pageid"] in done:
                    continue
                doc = page_doc(p["pageid"], True, max_words)
                if doc is None:
                    continue
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                f.flush()
                done.add(p["pageid"])
                print(len(done), doc["title"], doc["words"], flush=True)
                if len(done) >= n_docs:
                    break
                time.sleep(0.2)


def doubled_slots(toks):
    """ITCC's agreement doubling: (anchor token id, kind, pronoun form)."""
    from lgram.tr.centering import _PSOR_FORM, _SUBJECTS, _ZERO_FORM, _finite_carrier

    by_id = {t.id: t for t in toks}
    children = {}
    for t in toks:
        children.setdefault(t.head, []).append(t)
    predicates = {
        (t.head if t.deprel in ("cop", "aux") and t.head in by_id else t.id)
        for t in toks
        if t.feats.get("VerbForm") == "Fin"
    }
    for pid in sorted(predicates):
        carrier = _finite_carrier(by_id[pid], children)
        has_subject = any(c.deprel in _SUBJECTS for c in children.get(pid, []))
        if carrier is not None and has_subject:
            key = (carrier.feats.get("Person", "3"), carrier.feats.get("Number") == "Plur")  # fmt: skip
            yield pid, "zero", _ZERO_FORM.get(key, "o")
    for t in toks:
        psor = t.feats.get("Person[psor]")
        has_genitive = any(c.feats.get("Case") == "Gen" for c in children.get(t.id, []))
        if t.upos in ("NOUN", "PROPN") and psor and has_genitive:
            key = (psor, t.feats.get("Number[psor]") == "Plur")
            yield t.id, "possessor", _PSOR_FORM.get(key, "onun")


def empty_nodes(toks):
    """Token id -> pronoun forms of the empty nodes that follow it (possessors first)."""
    from lgram.tr.centering import analyze_parsed

    found = []
    analyze_parsed(["s"], [toks], identity=lambda i, m: found.append(m) or None)
    root = next((t.id for t in toks if t.head == 0), None)
    slots = [((root if m.pos == 0 else m.pos), m.kind, m.form)
             for m in found if m.kind in ("zero", "possessor")]  # fmt: skip
    slots += list(doubled_slots(toks))
    slots = [s for s in slots if s[0]]
    after = {}
    for tok_id, _kind, form in sorted(slots, key=lambda s: s[1] != "possessor"):
        after.setdefault(tok_id, []).append(form)
    return after


def is_tree(toks):
    """One root, and every token reaches it (the parser sometimes emits cycles)."""
    head = {t.id: t.head for t in toks}
    if sum(h == 0 for h in head.values()) != 1:
        return False
    for i in head:
        for _ in range(len(toks)):
            i = head.get(i, 0)
            if i == 0:
                break
        else:
            return False
    return True


def write_conllu(docs_path, out, max_sent_tokens=120):
    import warnings

    from lgram.tr.parser import JointParser, split_sentences

    warnings.simplefilter("ignore")
    parser = JointParser()
    n_sent = n_tok = n_empty = n_flat = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8", newline="\n") as g:
        for line in open(docs_path, encoding="utf-8"):
            doc = json.loads(line)
            doc_id = ("src" if "wikisource" in doc.get("site", "") else "wiki") + str(doc["id"])  # fmt: skip
            k = 0
            for p, para in enumerate(doc["paragraphs"]):
                for sent in split_sentences(para):
                    toks = parser.parse(sent)
                    # a run-on "sentence" is a list or a splitter miss: not worth labelling
                    if not toks or len(toks) > max_sent_tokens:
                        continue
                    after = empty_nodes(toks)
                    # CoNLL-U readers reject non-trees; the teacher reads the words,
                    # not the syntax, so a flat tree is enough for those sentences
                    heads = [(t.head, t.deprel) for t in toks]
                    if not is_tree(toks):
                        heads = [(0, "root")] + [(1, "dep")] * (len(toks) - 1)
                        n_flat += 1
                    if k == 0:
                        g.write(f"# newdoc id = {doc_id}\n")
                    k += 1
                    g.write(f"# sent_id = {doc_id}-p{p}-s{k}\n# text = {sent}\n")
                    for t, (head, deprel) in zip(toks, heads):
                        feats = "|".join(f"{a}={b}" for a, b in sorted(t.feats.items())) or "_"  # fmt: skip
                        g.write(f"{t.id}\t{t.form}\t_\t{t.upos}\t_\t{feats}\t{head}\t{deprel}\t_\t_\n")  # fmt: skip
                        for n, form in enumerate(after.get(t.id, []), 1):
                            g.write(f"{t.id}.{n}\t{form}\t_\tPRON\t_\t_\t_\t_\t{t.id}:dep\t_\n")  # fmt: skip
                            n_empty += 1
                    g.write("\n")
                    n_sent += 1
                    n_tok += len(toks)
            print(f"{doc['title']}: {n_sent} sentences so far, {time.time() - t0:.0f}s", flush=True)  # fmt: skip
    print(f"{n_sent} sentences ({n_flat} non-tree parses flattened), {n_tok} tokens, "
          f"{n_empty} empty nodes -> {out}")  # fmt: skip


if __name__ == "__main__":
    if "--site" in sys.argv:
        SITE = sys.argv[sys.argv.index("--site") + 1]
    if sys.argv[1] == "fetch-category":
        mw = int(sys.argv[sys.argv.index("--min-words") + 1]) if "--min-words" in sys.argv else 300
        fetch_category(sys.argv[2], sys.argv[3], min_words=mw)
    elif sys.argv[1] == "fetch":
        n = int(sys.argv[sys.argv.index("--docs") + 1]) if "--docs" in sys.argv else 400
        fetch(sys.argv[2], n)
    elif sys.argv[1] == "conllu":
        write_conllu(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
