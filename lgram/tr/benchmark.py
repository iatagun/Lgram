"""
Sentence-order discrimination for Turkish Centering — needs no human ratings.

Question: does the rough-shift ratio prefer the original sentence order over random
reorderings of the same paragraph? If not, the Turkish transitions carry no ordering
signal and the module should not advance (validation-first, cf. the English GATE 1).

Each sentence is parsed once; permutations reuse the parses. The run is repeated
with zero-pronoun recovery ablated, so its contribution is measured, not assumed.

Corpus: plain text, paragraphs separated by blank lines.

Usage:
    python -m lgram.tr.benchmark corpus.txt [--min-sents 5] [--perms 20]
"""

from __future__ import annotations

import argparse
import random
import sys
from math import comb
from pathlib import Path
from typing import List, Sequence

from .centering import analyze_parsed
from .parser import JointParser, Token, split_sentences


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def evaluate(
    paragraphs: Sequence[Sequence[Sequence[Token]]],
    perms: int,
    seed: int,
    resolve_zero: bool,
) -> dict:
    rng = random.Random(seed)
    wins = ties = losses = 0
    pair_score = pair_total = 0.0
    for parses in paragraphs:
        texts = [str(i) for i in range(len(parses))]
        orig = analyze_parsed(texts, parses, resolve_zero).rough_shift_ratio
        perm_ratios: List[float] = []
        for _ in range(perms):
            order = list(range(len(parses)))
            rng.shuffle(order)
            shuffled = [parses[i] for i in order]
            perm_ratios.append(
                analyze_parsed(texts, shuffled, resolve_zero).rough_shift_ratio
            )
        pair_score += sum(
            1.0 if orig < r else 0.5 if orig == r else 0.0 for r in perm_ratios
        )
        pair_total += len(perm_ratios)
        mean_perm = sum(perm_ratios) / len(perm_ratios)
        if orig < mean_perm:
            wins += 1
        elif orig > mean_perm:
            losses += 1
        else:
            ties += 1
    return {
        "paragraphs": len(paragraphs),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "pairwise_acc": pair_score / pair_total if pair_total else 0.0,
        "sign_p": sign_test_p(wins, losses),
    }


def load_paragraphs(path: Path, min_sents: int) -> List[List[str]]:
    blocks = path.read_text(encoding="utf-8").split("\n\n")
    paras = [split_sentences(" ".join(b.split())) for b in blocks if b.strip()]
    return [p for p in paras if len(p) >= min_sents]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--min-sents", type=int, default=5)
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--max-paragraphs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    paras = load_paragraphs(args.corpus, args.min_sents)[: args.max_paragraphs]
    if not paras:
        print("No paragraphs with enough sentences.", file=sys.stderr)
        return 1
    print(f"{len(paras)} paragraphs, parsing {sum(map(len, paras))} sentences...")
    parser = JointParser()
    parsed = [[parser.parse(s) for s in p] for p in paras]

    for label, zero in (("full", True), ("no-zero-pronoun", False)):
        r = evaluate(parsed, args.perms, args.seed, zero)
        print(
            f"{label:16s} pairwise acc = {r['pairwise_acc']:.3f}  "
            f"paragraphs win/tie/loss = {r['wins']}/{r['ties']}/{r['losses']}  "
            f"sign-test p = {r['sign_p']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
