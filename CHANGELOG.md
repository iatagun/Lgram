# Changelog

## Unreleased

### Added
- **`lgram.tr` (experimental, `pip install centering-lgram[tr]`):** Turkish Centering on
  UD parses from DizgeBERT-Joint — zero-pronoun recovery from verb agreement, possessive
  suffixes as possessors, genderless 3rd-person anaphora. Diagnostic only (transitions,
  rough-shift ratio), no quality score. **Not validated:** on 159 Turkish Wikipedia
  paragraphs the rough-shift ratio prefers the original sentence order only 51–52% of
  the time (chance = 50%, p ≥ 0.10), below a plain adjacent-sentence lexical-overlap
  baseline (58.5%, p = 0.04). Benchmark: `python -m lgram.tr.benchmark corpus.txt`.
  On 222 eight-sentence windows of Turkish folk tales the order test is also weak
  (0.535, p = 0.34; overlap baseline 0.707) and gets *better* with zero-pronoun
  resolution ablated (0.572, p = 0.02). Provisional single-annotator audit (n = 99):
  zero-subject detection F1 0.82, but antecedent resolution only 10/25 correct
  (ceiling of the link-to-previous-Cf design: 19/25) and 7/7 false links when the
  antecedent is absent.

- **`lgram.tr` measured against gold coreference.** Centering derived from Turkish-ITCC
  (CorefUD 1.3; dropped subjects and possessors annotated) is now the yardstick
  (`experiments/itcc_centering.py`, `experiments/tr_baseline_itcc.py`). Rule-based
  transition accuracy over all 21 ITCC documents: 0.54 by label, **0.48 strict** (label
  right *and* the Cb is the right entity) — still experimental. For scale: with no
  anaphora resolution at all (repeated nouns and speaker/addressee only) strict accuracy
  is 0.46, and with perfect entity identity on the same mention slots it is 0.74. What the measurements changed: a dropped subject
  is posited on every finite clause, not only the root; an implicit possessor binds to
  its own clause's subject; anaphors try the previous Cb first; proper names are not
  stemmed. Without dropped subjects 65% of ITCC transitions have no Cb (34% with them).
- **Coreference hook for `lgram.tr`:** `analyze_parsed(..., identity=fn)` lets an external
  coreference model decide which entity each mention is. A BERTurk model in the
  fastcoref architecture, trained on ITCC (`experiments/tr_coref_*.py`), reaches strict
  accuracy 0.50 vs 0.48 for the rules in 5-fold cross-validation (McNemar p = 0.004,
  better in 13 of 21 documents). The model is not shipped: ITCC is CC BY-NC-SA.
- **Silver training data for the Turkish model (`experiments/tr_silver.py`).** Fetches
  Turkish Wikipedia articles or a Wikisource category, parses them and writes CoNLL-U
  with `lgram.tr`'s dropped subjects / possessors as empty nodes, for a teacher model
  (CorPipe 25) to label. A BERTurk model trained only on that silver data — 450
  Wikipedia articles plus 69 public-domain stories, no ITCC document — reaches strict
  accuracy **0.55** over all 21 ITCC documents (rules 0.48). Genre mattered more than
  volume: Wikipedia alone gave 0.48, the stories added the rest. Caveats: the teacher
  was trained on ITCC and is CC BY-NC-SA, and the checkpoint was selected on an ITCC
  slice. Not shipped.
- **`python -m lgram.transition_eval`:** transition accuracy against a hand-annotated
  Cp/Cb sheet, plus inter-annotator kappa (`--agree`). The bundled English sheet is a
  single-annotator annotation. `experiments/coref_centering.py` shows the built-in
  English engine at 0.32 on it and a fastcoref-based prototype at 0.82–0.84.

### Changed
- **NOCB is a transition of its own.** An utterance that shares no entity with the
  previous one has no Cb; it used to be labelled Rough-Shift (about half of all
  "Rough-Shifts" on English Wikipedia and Grimm) and is now `TransitionType.NOCB`.
  `transition_distribution` gains a `"NOCB"` key and its `"Rough-Shift"` share drops
  accordingly — **add the two if you compare against earlier numbers.** The cohesion
  score is unchanged (NOCB keeps the Rough-Shift weight), and the essay layers, genre
  calibration and `diff_cohesion` still report the sum. ELLIPSE feature cache → v1.1.
- **`lgram.tr.rough_shift_ratio`** counts Rough-Shift + NOCB, as before the split.
- **Docs:** READMEs now state the ELLIPSE validation result (GATE 1 failed) and narrow
  the scalar cohesion score to a descriptive, unvalidated statistic.

### Fixed
- **`lgram.tr` overt possessors under the KeNet scheme:** the possessor relation is
  `nmod`/`compound` there, not `nmod:poss`, so "Kahvenin numarası" got a dropped
  possessor (27% of possessor slots were spurious) and genitive possessors never
  entered the Cf.
- **`lgram.tr` imperatives** had no subject at all; a 2nd-person imperative now
  realizes the addressee, a 3rd-person one takes an antecedent.
- **`lgram.tr` warns** when the parses carry no `VerbForm=Fin` (IMST/BOUN schemes),
  instead of silently finding no dropped subjects.
- **`lgram.tr` hang on malformed parses:** DizgeBERT can emit a self-headed token or a
  head cycle (0.4–1% of sentences); possessor binding then looped forever.
- **`lgram.tr` implicit possessors** are linked to the previous sentence only when the
  possessed noun is the subject ("Annesi geldi"). Linking every implicit possessor back
  cancelled the whole gain of zero-subject and pronoun resolution (strict 0.46 → 0.48).
- **`lgram.tr` sentence splitter** no longer splits after abbreviations and initials
  ("Prof. Dr. Ahmet") or before a lower-case continuation ("21. yüzyıl", "Nerdeydin?
  dedi"), and keeps a closing quote with its sentence. Boundary errors on ITCC raw
  text: 436 → 319.
- **`import lgram.tr` without the `[tr]` extra** now says how to install it. Stems are
  cached (the stemmer was 80% of the runtime).
- **Silent fabricated scores:** failures no longer turn into plausible-looking numbers.
  - Cohesion layer: an analysis error in a segment now propagates instead of scoring 0.5.
  - Grammar layer: a crashed LanguageTool check now yields the neutral 50 with
    `raw_details["check_failed"]=True`, instead of "zero errors" → 100.
  - Mechanics layer: a crashed spell check no longer invents a 0.8 spelling score.
  - `ellipse_features` / `benchmark`: removed `0.0` / `0.5` fallbacks that would have
    silently corrupted feature matrices and method-agreement stats.
  - Swallowed exceptions in the essay layers and deep-grammar client are now logged.

## v2.3.1 (2026-07-14)

### Fixed
- **Fresh installs were broken:** typer >= 0.22 no longer depends on `click`, but
  spaCy 3.8's CLI imports it directly — so `import spacy` (and therefore
  `import lgram`) failed with `ModuleNotFoundError: No module named 'click'` in any
  clean environment. Added an explicit `click>=8.0` dependency. This was also why
  every CI run since 2026-07-02 failed at the "Install dependencies" step.

## v2.3.0 (2026-07-14)

### Fixed
- **Packaging:** `lgram/data/gender_map.json` was missing from both wheel and sdist
  (package-data and MANIFEST.in only included `*.py`), so pip installs silently ran
  with an empty gender map. Now shipped.
- **C2 crash:** an LLM CEFR estimate of "C2" crashed `CAEASGrader.analyze()`
  (`CEFR_PROFILES` only covers B1-C1). C2 now clamps to C1, mirroring the A1/A2→B1 clamp.
- **Gender-alternation detector:** the regex counted `he...he` sequences as "he/she
  alternation", flagging all-male-pronoun texts. Now counts actual he↔she switches.
- **Pro-drop detector:** no longer flags questions ("Is this correct?") as
  missing-subject sentences.
- **Suffix-based gender heuristic:** now applies only to NER PERSON tokens, so
  "London"/"China" are no longer treated as gendered persons in pronoun matching.
- **LLM content cache:** `_CACHE_MAX_SIZE` was defined but never enforced; the cache
  now evicts oldest entries like the other layers.
- **Composite indicator:** cohesion was double-counted (30% Organization rubric weight
  plus 50% blend). The rubric-weighted half now excludes Organization; cohesion
  contributes exactly `cohesion_weight` (50%).
- **Custom rubrics:** layer weights are now matched to rubric criteria by name instead
  of positional zip, so a rubric passed in a different order gets correct weights.

### Changed
- **save()/load() use JSON** instead of pickle (safe to share and inspect; pickle
  could execute arbitrary code on load).
- **GCDC benchmark methodology:** accuracy is now computed at a fixed midpoint
  threshold between class means instead of scanning for the accuracy-optimal
  threshold on the evaluation data. Inverted score direction is reported via
  `GCDCResult.inverted` instead of being silently flipped — the embedded `enron`
  subset is a known inverted domain.
- **CEFR estimation** now combines lexical diversity (Guiraud), complex discourse
  marker rate, and sentence length; word count only gates the level ceiling and
  confidence, instead of being the sole proxy.
- **spaCy models are cached per process** — repeated `TextAnalyzer` construction no
  longer reloads the model from disk.
- **Python floor raised to 3.9** (3.8 is EOL; CI matrix already started at 3.9).
- `tests` package no longer installed into site-packages.
- Repo-wide `black` formatting; flake8 clean with black-compatible config (`.flake8`);
  mypy is informational in CI until typing debt (81 errors) is paid down.
- Marketing/assessment docs moved from repo root to `docs/`.
- README tests badge now reflects actual CI status instead of a static number.

## CAEAS v0.3 (2025-07-04)

### Improvements (2025-07-06)
- **Token optimization** — LLM calls gated and cached; long essays compressed to high-signal sentence excerpts
- **Core library** — analyzer.py (+314 lines), centering_theory.py (+32 lines), benchmark.py, gcdc_benchmark.py updates
- **167 tests** (+4 token-conscious efficiency tests), all passing
- **New reports** — `MODEL_COMPLETENESS_REPORT.md`, `PRODUCTION_READINESS_ASSESSMENT.md`

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
