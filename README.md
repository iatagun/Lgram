# Centering-Lgram

Discourse cohesion analysis based on **Centering Theory** (Grosz, Joshi, and Weinstein, 1983). Measures how entities and topics flow across sentences — pronouns, repetitions, entity continuity.

> **Note:** _Cohesion_ (bağdaşıklık) = surface grammatical/lexical links. _Coherence_ (tutarlılık) = deeper semantic unity. Centering Theory models cohesion.

[![PyPI](https://img.shields.io/badge/pypi-centering--lgram-blue)](https://pypi.org/project/centering-lgram/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-49%20passed-green)]()

---

## Quick Start

```bash
pip install centering-lgram
python -m spacy download en_core_web_sm
```

```python
from lgram import TextAnalyzer

ta = TextAnalyzer()
r = ta.analyze("AI helps doctors. It speeds up diagnosis. These tools save lives.")
print(r.overall_cohesion)  # 0.87
print(r.quality)           # "high"
```

**For best results**, use the medium model or sentence-transformers:

```python
# SpaCy medium model (40 MB, GloVe vectors)
ta = TextAnalyzer("en_core_web_md")

# Or sentence-transformers (80 MB, MiniLM)
ta = TextAnalyzer(use_sentence_transformers=True)
```

---

## Model Comparison

| Model | Size | Cohesion* | Continue | Rough-Shift |
|---|---|---|---|---|
| `en_core_web_sm` (no vectors) | 12 MB | 0.500 | 14% | 71% |
| `en_core_web_md` (GloVe 300d) | 40 MB | **0.943** | 57% | 0% |
| `all-MiniLM-L6-v2` (384d) | 80 MB | **0.914** | 57% | 0% |

*\*Same news article, 7 sentences. Threshold auto-adjusted per model.*

---

## Core Concepts

Centering Theory tracks three discourse centers per utterance:

| Center | Notation | Definition |
|--------|----------|------------|
| **Forward Centers** | Cf | Entities ordered by grammatical salience |
| **Backward Center** | Cb | Entity linking to previous utterance |
| **Preferred Center** | Cp | Highest-ranked Cf |

Five transition types between utterances:

| Transition | Rule | Quality |
|------------|------|---------|
| **Establish** | First utterance | — |
| **Continue** | Cb(Ui) = Cb(Ui-1) = Cp(Ui) | Best |
| **Retain** | Cb(Ui) = Cb(Ui-1) ≠ Cp(Ui) | Good |
| **Smooth-Shift** | Cb(Ui) ≠ Cb(Ui-1) = Cp(Ui) | OK |
| **Rough-Shift** | Cb(Ui) ≠ Cb(Ui-1) ≠ Cp(Ui) | Poor |

---

## API — TextAnalyzer (High-Level)

```python
from lgram import TextAnalyzer

ta = TextAnalyzer()                    # default: en_core_web_sm
ta = TextAnalyzer("en_core_web_md")    # better vectors
ta = TextAnalyzer(use_sentence_transformers=True)  # best quality
```

### Core Analysis

| Method | Description |
|---|---|
| `analyze(text)` | Full analysis → `TextReport` (sentences, paragraphs, transitions, entities) |
| `analyze_batch(texts, labels)` | Compare multiple texts with rankings |
| `analyze_llm(response, prompt?)` | LLM output quality: high/medium/low + prompt comparison |

### Cohesion Metrics

| Method | Source | Description |
|---|---|---|
| `entity_grid_score(text)` | Barzilay & Lapata 2005 | Entity role persistence (S/O/X/-) across sentences |
| `lexical_chain_score(text)` | Halliday & Hasan 1976 | Noun repetition + similarity chains |
| `build_cohesion_graph(text)` | Graph-based | Sentence adjacency graph (density, centrality, communities) |
| `cohesion_trend(text)` | Sliding window | Cohesion change across text (improving/declining/stable) |
| `cohesion_heatmap(text)` | Matrix | N×N sentence similarity with weak pair detection |
| `combined_score(text)` | Hybrid | Cohesion × 0.6 + Readability × 0.4 |

### Segmentation & Quality

| Method | Description |
|---|---|
| `texttile_segments(text)` | Hearst 1994 topic segmentation |
| `hybrid_boundaries(text)` | Centering + TextTiling intersection (high confidence) |
| `suggest_improvements(text)` | Find weak points + fix suggestions |
| `annotate_weak_points(text)` | Mark `<<<WEAK>>>` at cohesion breaks |
| `diff_cohesion(original, revised)` | Compare two versions |
| `readability_score(text)` | Flesch Reading Ease + statistics |

### Export

| Method | Output |
|---|---|
| `to_dict(report)` | JSON-serializable dict |
| `to_json(report)` | JSON string |
| `to_summary(report)` | Human-readable report |

---

## API — EnhancedCenteringTheory (Low-Level)

```python
from lgram import EnhancedCenteringTheory
import spacy

nlp = spacy.load("en_core_web_sm")
ct = EnhancedCenteringTheory(nlp)

state = ct.analyze_utterance("John went to the store.")
ct.update_discourse("He bought milk.")

result = ct.evaluate_cohesion(["John went.", "He bought milk.", "The store was busy."])
```

Key methods: `compute_forward_centers`, `compute_backward_center`, `determine_transition`, `extract_clauses`, `detect_boundaries`, `validate_sequence`, `visualize`, `compare_texts`, `stream_start/feed/flush`, `save/load`, `reset`.

---

## CLI

```bash
centering-lgram analyze --text "John went to the store. He bought milk."
centering-lgram clauses --text "She left because she was tired."
centering-lgram full --text "John left. He was tired because he worked late."
centering-lgram score --text "Alice met Bob. She greeted him."
centering-lgram info
centering-lgram version
```

---

## How It Works

### Salience Ranking

Cf ordered by: grammatical role (S=4 > O=3 > other=2 > poss=1) + POS (PRON=3 > PROPN=2 > NOUN=1) + position + entity type (PERSON/ORG/GPE bonus) + pronoun antecedent bonus.

### Backward Center (5-level cascade)

1. **Possessive scan** — "his"/"her" → matched person entity
2. **Direct match** — entity appears in both Cf lists
3. **Pronoun resolution** — gender-aware (he→male, she→female)
4. **Coreference** — entity type matching + vector similarity fallback
5. **Compound plural** — multiple persons → "they"

### Gender-Aware Pronoun Resolution

120+ name gender map (English + Turkish) + title detection (Mr/Mrs) + suffix heuristics. Male pronoun "he" does NOT match female entity "Alice".

### Clause Detection

Dependency parse: main, conj, advcl, ccomp, acl, relcl. Separator tokens (commas, conjunctions) assigned to following clause.

---

## Architecture

```
lgram/
  __init__.py              # Package exports
  analyzer.py        924   # TextAnalyzer (17 methods)
  benchmark.py       290   # CohesionBenchmark (4 tests)
  cli.py             238   # 6 CLI commands
  core.py              7   # Re-export hub
  utils.py            20   # Logging
  models/
    __init__.py         7   # Sub-package exports
    centering_theory.py 1122 # Core engine
tests/
  test_lgram.py       209   # 15 core tests
  test_edges.py       312   # 34 edge case tests
docs/
  RESEARCH.md                # Literature survey
  IMPLEMENTATION_PLAN.md     # Implementation plan
```

**Dependencies:** `spacy>=3.4.0` only. Optional: `sentence-transformers` for MiniLM.

---

## Use Cases

| Domain | Application |
|---|---|
| **LLM Evaluation** | Cohesion scoring for GPT/Claude/Llama output |
| **Education** | Essay scoring, writing assistant feedback |
| **Linguistics** | Discourse analysis research |
| **Content Quality** | Blog/news fluency audits |
| **Translation** | Cross-language cohesion comparison |
| **Dialogue** | Conversation flow naturalness |
| **Forensics** | Statement consistency analysis |

---

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2025 Ilker Atagun.
