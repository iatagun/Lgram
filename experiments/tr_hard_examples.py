"""
Hand-picked hard Turkish examples: what the rules and the model get right, case by case.

ITCC gives one average; this file shows which kinds of reasoning work. Each example in
tr_hard_examples.json lists its sentences and, per sentence, the expected Cp and Cb as
plain entity names ("ali", "anne"; null = no Cb). The transition is derived from those
(lgram.transition_eval.derive), so it cannot be annotated inconsistently. A sentence
may also name the topic move ("move": devam / yumuşak dönüş / içerme / tam dönüş), as
the user's lecture notes label it; that is scored separately.

A sentence counts as right only if the transition is right AND the Cb is that entity
(the strict criterion). These are known-hard cases: the output is a report, not a test.

Usage:
    python experiments/tr_hard_examples.py [MODEL_DIR]      # rules only without a model
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import tr_demo  # noqa: E402

from lgram.tr.centering import _base  # noqa: E402
from lgram.transition_eval import derive  # noqa: E402


def same(gold, label) -> bool:
    """Gold entity name vs a system label ("ayşe" ~ "Ayşe'yi", "anne" ~ "annes").

    ponytail: prefix match on the lower-cased base; fine for a hand-made set where
    the annotator picks distinct names, wrong for "ali" vs "aliye".
    """
    if gold is None or label is None:
        return gold is None and label is None
    return _base(label.split(" (")[0]).startswith(gold)


def main(argv):
    import logging

    import datasets
    from lgram.tr.parser import JointParser

    datasets.disable_progress_bar()
    logging.disable(logging.WARNING)
    tr_demo.PARSER = JointParser()
    if argv:
        from fastcoref import FCoref

        tr_demo.MODEL = FCoref(
            model_name_or_path=argv[0], nlp=None, enable_progress_bar=False
        )
    systems = ["rules"] + (["model"] if argv else [])
    examples = json.loads((HERE / "tr_hard_examples.json").read_text(encoding="utf-8"))
    score = {k: [0, 0] for k in systems}
    moves = {k: [0, 0] for k in systems}
    for ex in examples:
        res = tr_demo.analyze_sentences(ex["sentences"])["sentences"]
        assert len(res) == len(
            ex["sentences"]
        ), f"{ex['id']}: a sentence failed to parse"
        print(f"\n== {ex['id']}  ({ex['annotator']})")
        prev_cb = None
        for i, (sent, exp, r) in enumerate(zip(ex["sentences"], ex["expect"], res)):
            if i == 0:
                print(f"  1. {sent}")
                continue
            print(f"  {i + 1}. {sent}")
            if "cp" in exp:  # the BFP transition, strict
                gold_t = derive(exp["cp"], exp["cb"], prev_cb)
                print(f"       {'expected':9s} {gold_t:13s} Cb={exp['cb'] or '—'}  Cp={exp['cp']}")  # fmt: skip
                for k in systems:
                    u = r[k]
                    ok = u["transition"] == gold_t and same(exp["cb"], u["cb"])
                    score[k][0] += ok
                    score[k][1] += 1
                    print(f"       {k:9s} {u['transition']:13s} Cb={u['cb'] or '—'}  Cp={u['cp'] or '—'}"
                          f"   {'OK' if ok else 'WRONG'}")  # fmt: skip
            prev_cb = exp.get("cb")
            if exp.get("move"):  # the lecture notes' name for the move
                print(f"       {'expected':9s} {exp['move']}")
                for k in systems:
                    ok = r[k]["move"] == exp["move"]
                    moves[k][0] += ok
                    moves[k][1] += 1
                    print(f"       {k:9s} {r[k]['move']}   {'OK' if ok else 'WRONG'}")
    print()
    for k in systems:
        print(f"{k}: {score[k][0]}/{score[k][1]} transitions right (strict), "
              f"{moves[k][0]}/{moves[k][1]} topic moves right")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
