Architecture
============

Lgram uses a layered architecture:

::

                          +--------------------------+
                          |       CLI (cli.py)        |
                          |  6 commands: analyze,     |
                          |  score, clauses, full,    |
                          |  info, version            |
                          +-------------+------------+
                                        |
                          +-------------v------------+
                          |  TextAnalyzer (analyzer)  |
                          |  - analyze()              |
                          |  - entity_grid_score()    |
                          |  - texttile_segments()    |
                          |  - build_cohesion_graph() |
                          |  - lexical_chain_score()  |
                          |  - cohesion_trend()       |
                          |  - cohesion_heatmap()     |
                          |  - readability_score()    |
                          |  - suggest_improvements() |
                          |  - diff_cohesion()        |
                          |  - analyze_batch()        |
                          |  - analyze_llm()          |
                          +-------------+------------+
                                        |
                          +-------------v----------------+
                          |  EnhancedCenteringTheory      |
                          |  - compute_forward_centers()  |
                          |  - compute_backward_center()  |
                          |  - determine_transition()     |
                          |  - evaluate_cohesion()        |
                          |  - extract_clauses()          |
                          |  - analyze_intra_sentential() |
                          |  - detect_boundaries()        |
                          |  - validate_sequence()        |
                          |  - stream_start/feed/flush()  |
                          +-------------+----------------+
                                        |
                          +-------------v----------------+
                          |       spaCy NLP pipeline       |
                          |  (en_core_web_sm/md/lg)       |
                          |  optional: sentence-transformers|
                          +-------------------------------+

Plugin System
-------------

Analysis methods are organized as plugins via ``PluginRegistry``:

.. code-block:: python

    from lgram.plugins import registry, register_default_plugins

    register_default_plugins()

    for name in registry.list():
        print(name)

Cache Layer
-----------

Results are cached by text hash with configurable TTL:

.. code-block:: python

    from lgram.cache import AnalysisCache

    cache = AnalysisCache(max_size=256, default_ttl=300)
    result = cache.get(text)
    if result is None:
        result = ta.analyze(text)
        cache.set(text, result)

Model Registry
--------------

Automatic model selection with fallback:

.. code-block:: python

    from lgram.model_registry import get_model

    info = get_model("auto")  # tries md, falls back to sm
    print(info.model_name, info.similarity_threshold)

Discourse Relation Analysis
---------------------------

PDTB-lite approach detecting explicit connectives:

.. code-block:: python

    from lgram.discourse import DiscourseAnalyzer

    da = DiscourseAnalyzer()
    report = da.analyze("John went to the store because he needed milk.")
    print(report.relation_distribution)  # {"CONTINGENCY": 1}
    print(report.cohesion_score)
