# Centering-Lgram

Discourse cohesion analysis based on **Centering Theory** (Grosz, Joshi, and Weinstein, 1983). Measures how entities and topics flow across sentences — pronouns, repetitions, entity continuity.

> **Note:** _Cohesion_ (bağdaşıklık) = surface grammatical/lexical links. _Coherence_ (tutarlılık) = deeper semantic unity. Centering Theory models cohesion.

---

## ⚠️ What This Tool Does NOT Measure

- **Factual accuracy** — a high cohesion score does NOT mean the content is correct
- **Hallucination** — a fluent-sounding LLM output can still be completely false
- **Overall quality** — cohesion is ONE dimension of text quality, not the whole picture

**Correct positioning:** Use centering-lgram as a **complementary** evaluator alongside faithfulness/accuracy checkers. It measures **"how smoothly does this read?"**, not **"is this correct?"**.

| What we measure | What we DON'T measure |
|---|---|
| Entity flow across sentences | Factual correctness |
| Pronoun resolution quality | Hallucination / faithfulness |
| Topic continuity / shifts | Logical reasoning |
| Readability (Flesch) | Relevance to prompt |
| Lexical repetition chains | Domain accuracy |

---

[![PyPI](https://img.shields.io/badge/pypi-centering--lgram-blue)](https://pypi.org/project/centering-lgram/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-163%20passed-green)]()

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

**For best results**, use the medium model:

```python
ta = TextAnalyzer("en_core_web_md")
```

---

## Genre Calibration

*Empirically derived transition patterns. Method: Tukey's fence (p75 + 1.5×IQR).*

### Brown Corpus (1960s)

*NLTK Brown: 1.1M words, 500 files, 15 categories. n=30/genre, HIGH confidence.*

| Genre | Rough normal (p25–p75) | Flag > (Tukey) | Continue mean | Conf. |
|---|---|---|---|---|
| **Narrative** | 11.5% – 27.3% | 51.0% | 49.8% | HIGH |
| **Expository** | 16.7% – 33.3% | 58.2% | 29.2% | HIGH |
| **Essay** | 11.5% – 27.7% | 52.0% | 30.7% | HIGH |

### Modern Corpus (2020s) — n=30/genre, HIGH confidence

| Genre | Rough normal (p25–p75) | Flag > (Tukey) | Continue mean | Conf. |
|---|---|---|---|---|
| **Narrative** | 0.0% – 25.0% | 62.5% | 25.0% | HIGH |
| **Expository** | 0.0% – 25.0% | 62.5% | 32.3% | HIGH |
| **Essay** | 0.0% – 25.0% | 62.5% | 20.3% | HIGH |

*Note: Single-author corpus. Stylistic homogeneity may narrow distributions.*

### Cross-Validation: Wikipedia (multi-author, 2024)

*n=12, MEDIUM confidence. Small sample — treat as observational, not conclusive.*

| Genre | n | Rough normal | Flag > | Continue | Conf. |
|---|---|---|---|---|---|
| **Expository** (Wikipedia) | 12 | 23.0% – 34.6% | 52.0% | 34.5% | MEDIUM |

*Wikipedia's 52% flag threshold is broadly consistent with Brown's 58%. Direction is as expected (multi-author corpora cluster together, single-author is the outlier). However, at n=12 this is an observation, not a validated claim. Two caveats: (1) Wikipedia's strict editing guidelines may make it a distinct sub-genre, not representative of general 2020s expository writing. (2) The 6% gap could be measurement noise, temporal change, or genre artifact — the current data cannot distinguish between these explanations.*

### Findings

1. **Rough-Shift >50% is abnormal** — all corpora agree. Flag thresholds: 51-63%.
2. **Wikipedia cross-validates the multi-author finding** — Brown (58%) and Wikipedia (52%) cluster together, while the single-author corpus (63%) diverges. Consistent with the stylistic homogeneity hypothesis. However, n=12 precludes strong conclusions about temporal stability.
3. **Wikipedia is not "general 2020s writing"** — its strict editing guidelines may constitute a distinct sub-genre. More diverse modern sources needed.

*Calibration is reproducible: `python -m lgram.brown_calibration`*

> ⚠️ **Reliability note:** Scores depend on the underlying embedding model. For production use, pick ONE model (recommended: `en_core_web_md`) and standardize on it. Compare texts only within the same genre — cross-genre comparison is meaningless because different genres have different natural transition patterns.

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
