#!/usr/bin/env python3
"""
CLI for Centering-Lgram: Discourse cohesion analysis with Centering Theory.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Centering-Lgram: Discourse cohesion analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  centering-lgram analyze --text "John went to the store. He bought milk."
  centering-lgram clauses --text "She left because she was tired."
  centering-lgram full --text "John left. He was tired because he worked late."
  centering-lgram score --text "Alice met Bob. She greeted him."
  centering-lgram info
  centering-lgram version
""",
    )

    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="Full centering analysis")
    analyze.add_argument("--text", "-t", type=str, help="Text to analyze")
    analyze.add_argument("--file", "-f", type=Path, help="File to analyze")
    analyze.add_argument(
        "--model",
        "-m",
        type=str,
        default="en_core_web_sm",
        help="SpaCy model (default: en_core_web_sm)",
    )
    analyze.add_argument("--verbose", "-v", action="store_true")

    score = subparsers.add_parser("score", help="Cohesion score only")
    score.add_argument("--text", "-t", type=str)
    score.add_argument("--file", "-f", type=Path)
    score.add_argument(
        "--model",
        "-m",
        type=str,
        default="en_core_web_sm",
        help="SpaCy model (default: en_core_web_sm)",
    )

    clauses_parser = subparsers.add_parser(
        "clauses", help="Intra-sentential clause-level analysis"
    )
    clauses_parser.add_argument("--text", "-t", type=str, help="Sentence to analyze")
    clauses_parser.add_argument("--file", "-f", type=Path, help="File to analyze")
    clauses_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="en_core_web_sm",
        help="SpaCy model (default: en_core_web_sm)",
    )

    full_parser = subparsers.add_parser(
        "full", help="Full inter + intra-sentential analysis"
    )
    full_parser.add_argument("--text", "-t", type=str, help="Text to analyze")
    full_parser.add_argument("--file", "-f", type=Path, help="File to analyze")
    full_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="en_core_web_sm",
        help="SpaCy model (default: en_core_web_sm)",
    )

    subparsers.add_parser("info", help="Show package info")
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    handlers = {
        "version": _version,
        "info": _info,
        "analyze": _analyze,
        "score": _score,
        "clauses": _clauses,
        "full": _full,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


def _read(args) -> str:
    if args.text is not None:
        return args.text
    if args.file:
        path: Optional[Path] = args.file
        if path and not path.exists():
            raise FileNotFoundError(str(path))
        return path.read_text(encoding="utf-8")
    raise ValueError("Provide --text or --file")


def _version(_args) -> int:
    from lgram import __version__, __author__

    print(f"v{__version__}  {__author__}")
    return 0


def _info(_args) -> int:
    from lgram import show_info

    show_info()
    return 0


def _centering_analysis(text: str, model_name: str, verbose: bool) -> int:
    import spacy
    from lgram import EnhancedCenteringTheory

    nlp = spacy.load(model_name)
    ct = EnhancedCenteringTheory(nlp)
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    if not sentences:
        print("No sentences found.")
        return 1

    print(f"Sentences: {len(sentences)}")
    print("=" * 60)

    for i, sent in enumerate(sentences):
        st = ct.update_discourse(sent)
        t = st.transition.value if st.transition else "?"
        print(f"\n[{i+1}] {sent}")
        header = f"    {t}"
        if st.preferred_center:
            header += f"  |  Cp: {st.preferred_center}"
        if st.backward_center:
            header += f"  |  Cb: {st.backward_center}"
        print(header)
        if verbose and st.forward_centers:
            print(f"    Cf: {st.forward_centers}")

    result = ct.evaluate_cohesion(sentences)
    print(f"\n{'=' * 60}")
    print(f"Cohesion: {result['cohesion_score']:.3f}")
    for label, pct in result["transition_distribution"].items():
        bar = "=" * int(pct * 30)
        print(f"  {label:14s} {bar} {pct:.0%}")
    return 0


def _analyze(args) -> int:
    return _centering_analysis(_read(args), args.model, args.verbose)


def _score(args) -> int:
    import spacy
    from lgram import EnhancedCenteringTheory

    text = _read(args)
    nlp = spacy.load(args.model)
    ct = EnhancedCenteringTheory(nlp)
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    result = ct.evaluate_cohesion(sentences)
    print(f"Cohesion: {result['cohesion_score']:.3f}  ({len(sentences)} sentences)")
    for label, pct in result["transition_distribution"].items():
        print(f"  {label}: {pct:.0%}")
    return 0


def _clauses(args) -> int:
    import spacy
    from lgram import EnhancedCenteringTheory

    text = _read(args)
    nlp = spacy.load(args.model)
    ct = EnhancedCenteringTheory(nlp)

    doc = nlp(text)
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        clauses_data = ct.extract_clauses(sent_text)
        if len(clauses_data) < 2:
            print(f"[1 clause, skipping] {sent_text}")
            continue

        print(f"Sentence: {sent_text}")
        print(f"  Clauses found: {len(clauses_data)}")
        for clause_text, ctype in clauses_data:
            print(f"    [{ctype}] {clause_text}")

        result = ct.analyze_intra_sentential(sent_text)
        print(f"  Intra-sentential cohesion: {result['cohesion_score']:.3f}")
        for t in result["transitions"]:
            print(
                f"    {t['transition']:12s}  Cp={t['cp'] or '-':8s}  Cb={t['cb'] or '-'}  [{t['clause']}]"
            )
        print()
    return 0


def _full(args) -> int:
    import spacy
    from lgram import EnhancedCenteringTheory

    text = _read(args)
    nlp = spacy.load(args.model)
    ct = EnhancedCenteringTheory(nlp)

    result = ct.analyze_full(text)

    print(f"Sentence count: {result['sentence_count']}")
    print(
        f"Inter-sentential cohesion: {result['inter_sentential']['cohesion_score']:.3f}"
    )
    print(f"  {result['inter_sentential']['transition_distribution']}")
    print()

    for i, intra in enumerate(result["intra_sentential"]):
        sent = intra["sentence"]
        score = intra["cohesion_score"]
        clauses = intra["clause_count"]
        if clauses >= 2:
            print(f"  [{i+1}] {sent[:60]}{'...' if len(sent) > 60 else ''}")
            print(f"       clauses={clauses}  cohesion={score:.3f}")
            for t in intra["transitions"]:
                print(f"         {t['transition']:12s}  {t['clause'][:40]}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"Dependency missing: {e}", file=sys.stderr)
        sys.exit(1)
