# Changelog

## 2.1.0 (2025-07-02)

### New Features

- **Gender-aware pronoun resolution** — 60-name gender map (male/female). "he" matches "bob" but not "alice". Configurable via `gender_map` constructor parameter.
- **Vector-based semantic coreference** — uses spaCy word vectors (md/lg models) to detect coreference between semantically similar entities (e.g., "study"↔"research"). Configurable `similarity_threshold`.
- **Discourse boundary detection** — `detect_boundaries()` identifies topic shift points via ROUGH-SHIFT clustering.
- **Annotated text output** — `annotate_text()` returns structured JSON per utterance (Cf, Cb, Cp, entities, gender).
- **Rule validation** — `validate_sequence()` checks Rule 1 (Cb from Cf(Ui-1)) and Rule 2 (pronoun realization constraint).
- **LLM output evaluator** — `analyze_llm()` scores LLM-generated text cohesion with quality rating (high/medium/low) and optional prompt comparison.
- **Visualization** — `visualize()` generates ASCII cohesion graph with transition symbols and score bar.
- **Comparative analysis** — `compare_texts()` ranks multiple texts by cohesion score.
- **Streaming analysis** — `stream_start()/stream_feed()/stream_flush()` for real-time incremental cohesion tracking.
- **`reset()` method** — Clear discourse history.

### Improvements

- `_pronoun_matches_entity` refactored with clean elif chain and pronoun-chain fallthrough
- All public methods now use `try/finally` for exception-safe state restoration
- `save()`/`load()` now serialize `similarity_threshold` and `_gender_lookup`
- `visualize()` computes score inline (no double-parse)
- `stream_start()` saves existing history before resetting
- `stream_flush()` restores previous history

### Terminology

- Renamed "coherence" → "cohesion" throughout (API, CLI, docs). `evaluate_coherence` kept as deprecated alias.
- Linguistically accurate: Centering Theory models surface-level cohesion (bağdaşıklık), not semantic coherence (tutarlılık).

## 2.0.0 (2025-07-01)

### Breaking Changes

- Removed all statistical n-gram language model components
- Removed T5 grammar correction, GPU acceleration, gender detection
- Package renamed from `lgram` to `centering-lgram` (the import remains `lgram`)
- Minimum dependencies reduced from 7 to 1 (`spacy>=3.4.0` only)

### Features

- Complete Centering Theory implementation (Grosz, Joshi, Weinstein)
- Five transition types: Establish, Continue, Retain, Smooth-Shift, Rough-Shift
- Configurable salience and POS weights
- Pronoun resolution with person/object/plural distinction
- Possessive pronoun handling (his, her, their)
- Compound plural antecedent detection
- Intra-sentential clause-level analysis
- Combined inter + intra-sentential analysis
- CLI with 6 commands: analyze, score, clauses, full, info, version
- Save/load discourse state
- 49 tests covering edge cases and false positives

### Bug Fixes (since original 1.x)

- Corrected transition classification for first utterance pair
- Fixed entity map overwrite on duplicate words
- Fixed pronoun filtering (possessive pronouns no longer candidate centers)
- Fixed coreference resolution across utterance boundaries
- Eliminated dead code and duplicate logic
