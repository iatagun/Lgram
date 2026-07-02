# Changelog

## 2.2.0 (2025-07-02)

### New: Analysis Layer (TextAnalyzer)

- **High-level API** — `TextAnalyzer` class wraps EnhancedCenteringTheory
- `analyze()` — full text analysis: sentences, paragraphs, transitions, entities
- `analyze_batch()` — multi-text comparison with rankings
- **Entity Grid** (Barzilay & Lapata 2005) — entity role persistence across sentences
- **TextTiling** (Hearst 1994) — vector-based topic segmentation
- **Hybrid boundaries** — Centering + TextTiling intersection
- **Cohesion graph** — sentence adjacency matrix with density, centrality, communities
- **Lexical chains** — noun repetition + similarity chains
- **Cohesion trend** — sliding window with improving/declining/stable detection
- **Cohesion heatmap** — N×N similarity matrix with weak pair detection
- **Readability** — Flesch Reading Ease + combined score
- **Suggestions** — detect weak points, suggest fixes
- **Diff analysis** — compare two text versions
- **Benchmark suite** — 4 validation tests (permutation, degradation, cross-method, classification)

### New: Model Support

- `en_core_web_sm` (12 MB, baseline)
- `en_core_web_md` (40 MB, GloVe 300d vectors)
- `all-MiniLM-L6-v2` (80 MB, sentence-transformers, optional)
- Auto-adjusted similarity threshold per model
- MiniLM connected to core centering engine for full benefit

### Improvements

- Gender map expanded to 120+ names (English + Turkish)
- Title/honorific detection (Mr/Mrs/Ms)
- Suffix-based gender heuristics for unknown names
- `is_female` boolean for O(1) gender checks
- `is_person` inferred from gender map (not just spaCy NER)
- Male/female pronoun matching with early reject
- All public methods exception-safe (try/finally)
- `reset()` method added
- `save()`/`load()` include similarity_threshold and gender_lookup
- Empty/single-sentence texts return "insufficient_data"
- Zero dead code, zero unused imports

## 2.1.0 (2025-07-02)

- Gender-aware pronoun resolution (60-name map)
- Vector-based semantic coreference (md/lg models)
- Discourse boundary detection
- Annotated text output
- Rule validation (Rule 1 + Rule 2)
- LLM output evaluator
- Visualization (ASCII graph)
- Comparative analysis
- Streaming analysis (start/feed/flush)

## 2.0.0 (2025-07-01)

- Complete rewrite: Centering Theory only
- Removed all statistical ML components
- Single dependency: spacy>=3.4.0
- 5 transition types, configurable weights
- Intra-sentential clause analysis
- 49 tests
