"""
Centering-Lgram: Discourse Cohesion Analysis with Centering Theory

A library implementing Centering Theory (Merkezleme Kurami) by Grosz, Joshi,
and Weinstein for analyzing entity tracking and discourse cohesion.

Centering Theory models surface-level discourse cohesion through:
- Forward center (Cf) computation with grammatical salience weighting
- Backward center (Cb) inference with pronoun resolution
- Transition classification: Establish, Continue, Retain, Smooth-Shift, Rough-Shift
- Cohesion scoring based on transition distributions

Note: In linguistics, cohesion (bagdasiklik) concerns surface grammatical/lexical
links, while coherence (tutarlilik) concerns deeper semantic unity. This library
implements local discourse cohesion as modeled by Centering Theory.
"""

__version__ = "2.2.0"
__author__ = "Ilker Atagun"
__email__ = "ilker.atagun@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/iatagun/Lgram"

import logging

from .utils import setup_logging

try:
    from .core import (
        EnhancedCenteringTheory,
        CenteringState,
        DiscourseEntity,
        TransitionType,
    )
    from .analyzer import TextAnalyzer, TextReport, SentenceAnalysis, ParagraphAnalysis
    _import_success = True
except ImportError as e:
    _import_success = False
    import warnings

    warnings.warn(
        f"Could not import core components: {e}. "
        "Install with: pip install centering-lgram",
        ImportWarning,
    )


def show_info():
    status = "OK" if _import_success else "WARN"
    print(
        f"LGRAM v{__version__} [{status}]\n"
        f"Centering Theory (Grosz, Joshi, Weinstein)\n"
        f"Transition classification: Continue, Retain, Smooth-Shift, Rough-Shift\n"
        f"\nQuick Start:\n"
        f"    import spacy\n"
        f"    from lgram import EnhancedCenteringTheory\n"
        f"    nlp = spacy.load('en_core_web_sm')\n"
        f"    ct = EnhancedCenteringTheory(nlp)\n"
        f"    state = ct.analyze_utterance('John went to the store.')\n"
    )


__all__ = [
    "EnhancedCenteringTheory",
    "CenteringState",
    "DiscourseEntity",
    "TransitionType",
    "TextAnalyzer",
    "TextReport",
    "SentenceAnalysis",
    "ParagraphAnalysis",
    "setup_logging",
    "show_info",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

try:
    from .essay.grader import CAEASGrader
    from .essay.models import Essay, CAEASReport
    __all__ += ["CAEASGrader", "Essay", "CAEASReport"]
except ImportError:
    pass

logging.getLogger(__name__).addHandler(logging.NullHandler())
