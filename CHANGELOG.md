# Changelog

## CAEAS v0.3 (2025-07-04)

### New: Full 5-Layer Rubric with Real NLP Tools

- **Grammar Layer** — LanguageTool (binlerce kural) + LLM deep grammar check
  - Catches mechanical errors (missing "to", spelling, punctuation)
  - LLM supplement finds subject-verb agreement, missing subjects, article errors
  - Combined: 6 errors detected vs 1 with LanguageTool alone
- **Content Layer** — LM Studio / local LLM integration
  - Structured output via `response_format=json_schema`
  - Auto-detects running local server (LM Studio → Ollama)
  - Falls back to heuristic if no LLM available
- **Mechanics Layer** — pyspellchecker (70K word dictionary)
  - Spelling, capitalization, terminal punctuation check
- **Grammar/Cohesion Disambiguation** — DeepGrammarCheck via raw HTTP
  - Avoids OpenAI client library compatibility issues with thinking models
  - Uses `response_format` for clean JSON output from thinking models

### Improvements

- **Composite formula** — cohesion_score contributes 50% to composite_indicator
  - Fixes "composite dilutes cohesion signal" problem
  - Composite delta: 4p → 39p (near cohesion's 56p discriminative power)
  - `cohesion_weight` calibratable hyperparameter (default 0.50)
- **CEFR unified** — LLM estimate overrides heuristic word-count estimate
  - A1/A2 mapped to B1 (closest supported level)
- **CI width** — minimum 3.0 SEM when LLM deep check active (non-determinism penalty)
- **163 tests** (99 core + 38 CAEAS + 26 EFL), all passing

### Bug Fixes (from v0.2 code review)

- BUG-1: CI scale mismatch (0-1 vs 0-100) in prefilter
- BUG-2: Weight truncation — only 3 of 5 rubric weights used
- BUG-3: L1 analyzer not created with default `l1_language="tr"`
- BUG-7/8: ErrorTypology double/triple counting
- Dead code removal: 6 duplicated `_split_sentences`, duplicated QWK/ICC
- Shared modules: `utils.py`, `metrics.py`

### Documentation

- `CAEAS_DEVELOPMENT_LOG.md` — full build log + calibration protocol
- `CAEAS_V02_PLAN.md` — v0.2 implementation plan (completed)
- `examples/demo.py` — 10-section full feature demo
- `examples/full_test.py` — 5-layer discriminative validity test

---

## CAEAS v0.1–v0.2 (2025-07-04)

### v0.2: Production Hardening (6 risk mitigations)

- **PreFilter** — grammar/cohesion disambiguation layer (LanguageTool optional)
- **CEFRCalibrator** — per-level calibration curves with complexity-adjusted scoring
- **Terminology audit** — coherence→cohesion, verdict→suggestion, grade→analyze
- **Feedback-mode positioning** — "not a grading system, evidence for teacher judgment"
- **DataExporter** — research-quality JSON/CSV export with anonymization
- **ErrorTypology** — 8 error categories with L1 transfer tagging

### v0.1: Initial Architecture

- 5-layer evidence-based essay analysis (Content, Cohesion, Surface, Calibration, Confidence)
- EFL module: 5-dimension rubric, CEFR profiles (B1/B2/C1), L1 transfer analysis (Turkish)
- `CAEASGrader` with `analyze()` / `grade()` API
- Segment-aware cohesion analysis (intro/body/conclusion)
- Population calibration (QWK, ICC, isotonic regression)

---

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
