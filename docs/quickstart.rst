Quick Start
===========

Installation
------------

.. code-block:: bash

    pip install centering-lgram
    python -m spacy download en_core_web_sm

Basic Usage
-----------

.. code-block:: python

    import spacy
    from lgram import EnhancedCenteringTheory

    nlp = spacy.load("en_core_web_sm")
    ct = EnhancedCenteringTheory(nlp)

    state = ct.analyze_utterance("John went to the store.")
    print(state.transition)  # Establish

    state = ct.update_discourse("He bought milk.")
    print(state.transition)  # Continue

    result = ct.evaluate_cohesion([
        "John went to the store.",
        "He bought milk.",
        "The store was busy.",
        "Then John left.",
    ])
    print(result["cohesion_score"])  # 0.0 - 1.0

High-Level API (TextAnalyzer)
-----------------------------

.. code-block:: python

    from lgram import TextAnalyzer

    ta = TextAnalyzer()
    report = ta.analyze("John went to the store. He bought milk. John paid. He left.")
    print(report.overall_cohesion)
    print(report.quality)  # high / medium / low
    print(ta.to_summary(report))

CLI
---

.. code-block:: bash

    centering-lgram analyze --text "John went to the store. He bought milk."
    centering-lgram score --text "John went. He bought. John paid. He left."
    centering-lgram clauses --text "She left because she was tired."
    centering-lgram full --text "John left. He was tired because he worked late."
    centering-lgram info
    centering-lgram version
