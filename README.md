# Centering-Lgram

Discourse coherence analysis based on **Centering Theory** (Grosz, Joshi, and Weinstein, 1983). A specialized NLP library for analyzing how topics flow and shift across sentences and within complex clauses.

[![PyPI](https://img.shields.io/badge/pypi-centering--lgram-blue)](https://pypi.org/project/centering-lgram/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
  - [Inter-sentential (Sentence-Level)](#inter-sentential-sentence-level)
  - [Intra-sentential (Clause-Level)](#intra-sentential-clause-level)
  - [Combined Analysis](#combined-analysis)
  - [Configuration](#configuration)
  - [Inspection & Serialization](#inspection--serialization)
- [CLI Reference](#cli-reference)
- [Use Cases](#use-cases)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [License](#license)

---

## Installation

```bash
pip install centering-lgram
python -m spacy download en_core_web_sm
```

Requires Python 3.8+ and spaCy 3.4+.

---

## Quick Start

```python
import spacy
from lgram import EnhancedCenteringTheory

nlp = spacy.load("en_core_web_sm")
ct = EnhancedCenteringTheory(nlp)

# Analyze transitions between sentences
s1 = ct.update_discourse("John went to the store.")
s2 = ct.update_discourse("He bought milk.")
s3 = ct.update_discourse("The milk was fresh.")

print(f"U1->U2: {s2.transition.value}")   # Continue
print(f"U2->U3: {s3.transition.value}")   # Retain
```

---

## Core Concepts

Centering Theory models discourse coherence through three types of discourse centers:

| Center | Notation | Definition |
|--------|----------|------------|
| **Forward Centers** | Cf | Entities introduced in the current utterance, ordered by salience |
| **Backward Center** | Cb | Entity linking current utterance to previous discourse |
| **Preferred Center** | Cp | Highest-ranked forward center (Cf[0]) |

The relationship between these centers defines five transition types:

| Transition | Rule | Coherence |
|------------|------|-----------|
| **Establish** | First utterance, no previous center | — |
| **Continue** | Cb(Ui) = Cb(Ui-1) = Cp(Ui) | Best |
| **Retain** | Cb(Ui) = Cb(Ui-1) ≠ Cp(Ui) | Good |
| **Smooth-Shift** | Cb(Ui) ≠ Cb(Ui-1) = Cp(Ui) | Acceptable |
| **Rough-Shift** | Cb(Ui) ≠ Cb(Ui-1) ≠ Cp(Ui) | Poor |

---

## API Reference

### Inter-sentential (Sentence-Level)

```python
import spacy
from lgram import EnhancedCenteringTheory, CenteringState, TransitionType

nlp = spacy.load("en_core_web_sm")
ct = EnhancedCenteringTheory(nlp)
```

#### `analyze_utterance(utterance: str) -> CenteringState`

Analyze a single utterance without modifying discourse history. Useful for inspection.

```python
state = ct.analyze_utterance("John went to the store.")
print(state.forward_centers)   # ['john', 'store']
print(state.preferred_center)  # 'john'
print(state.backward_center)   # None (first utterance)
print(state.transition)        # TransitionType.ESTABLISH
```

#### `update_discourse(utterance: str) -> CenteringState`

Analyze utterance AND add it to discourse history. Subsequent calls use accumulated context.

```python
ct.update_discourse("John went to the store.")
state = ct.update_discourse("He bought milk.")
print(state.backward_center)   # 'john'
print(state.transition)        # TransitionType.CONTINUE
```

#### `evaluate_coherence(utterances: List[str]) -> Dict[str, Any]`

Batch-evaluate a sequence of utterances. Does NOT modify instance state.

```python
result = ct.evaluate_coherence([
    "John went to the store.",
    "He bought milk.",
    "The store was busy.",
    "Then John left.",
])
print(result["coherence_score"])          # 0.0 – 1.0
print(result["transition_distribution"])  # {Continue: 0.5, ...}
print(result["total_transitions"])        # 4
```

#### `compute_forward_centers(utterance: str) -> Tuple[List[str], Dict]`

Low-level: compute centers and entity metadata without discourse context.

```python
centers, entities = ct.compute_forward_centers("Alice met Bob at the park.")
print(centers)  # ['alice', 'bob', 'park']
```

#### `get_coherent_next_center() -> Optional[str]`

Suggest the most coherent center to continue with.

```python
center = ct.get_coherent_next_center()
```

#### `get_discourse_summary() -> Dict[str, Any]`

Inspect current discourse state.

```python
summary = ct.get_discourse_summary()
# {'recent_centers': [...], 'recent_transitions': [...],
#  'current_cb': 'john', 'current_cp': 'he', 'discourse_length': 5}
```

---

### Intra-sentential (Clause-Level)

Analyze coherence within a single complex sentence by splitting it into clauses.

```python
ct = EnhancedCenteringTheory(nlp)
```

#### `extract_clauses(sentence: str) -> List[Tuple[str, str]]`

Split a sentence into clauses. Returns list of `(clause_text, clause_type)`.

```python
clauses = ct.extract_clauses(
    "John went to the store because he needed milk, but the store was closed."
)
# [("John went to the store", "main"),
#  ("because he needed milk", "advcl"),
#  (", but the store was closed.", "conj")]
```

Supported clause types: `main`, `conj` (coordinated), `advcl` (adverbial), `ccomp` (complement), `acl`/`relcl` (relative).

#### `analyze_intra_sentential(sentence: str) -> Dict[str, Any]`

Full clause-level analysis with transitions and coherence score.

```python
result = ct.analyze_intra_sentential(
    "John went to the store because he needed milk."
)
print(result["clause_count"])     # 2
print(result["coherence_score"])  # 0.0 – 1.0
for t in result["transitions"]:
    print(f"{t['clause']} -> {t['transition']}")
# John went to the store -> Establish
# because he needed milk -> Continue
```

---

### Combined Analysis

#### `analyze_full(text: str) -> Dict[str, Any]`

Runs both inter-sentential and intra-sentential analysis on a text.

```python
result = ct.analyze_full(
    "John went to the store. He bought milk because he was hungry."
)
print(result["sentence_count"])                 # 2
print(result["inter_sentential"]["coherence_score"])  # sentence-level
for intra in result["intra_sentential"]:        # per-sentence clause-level
    print(f"  {intra['sentence']}: score={intra['coherence_score']}")
```

---

### Configuration

```python
ct = EnhancedCenteringTheory(
    nlp_model=nlp,
    history_limit=20,                         # max discourse history (default: 10)
    salience_weights={                        # customize grammatical role weights
        "nsubj": 5.0,
        "dobj": 4.0,
        "pobj": 2.0,
        "poss": 1.5,
    },
    pos_weights={                             # customize POS-based weights
        "PRON": 4.0,
        "PROPN": 3.0,
        "NOUN": 1.0,
    },
)
```

### Inspection & Serialization

```python
# Save state for later analysis
ct.update_discourse("Alice met Bob.")
ct.update_discourse("She smiled.")
ct.save("discourse_state.pkl")

# Reload in another session
ct2 = EnhancedCenteringTheory(nlp)
ct2.load("discourse_state.pkl")
assert len(ct2.discourse_history) == 2

# Reset state
ct.discourse_history.clear()
```

**Warning:** `save()` / `load()` use pickle. Only load files from trusted sources.

---

## CLI Reference

```bash
# Sentence-level analysis
centering-lgram analyze --text "John went to the store. He bought milk."
centering-lgram analyze --file document.txt --verbose

# Coherence score only
centering-lgram score --text "Alice met Bob. She greeted him."
centering-lgram score --file essay.txt

# Intra-sentential clause analysis
centering-lgram clauses --text "She left because she was tired."
centering-lgram clauses --file sentences.txt

# Full combined analysis
centering-lgram full --text "John left. He was tired because he worked late."
centering-lgram full --file passage.txt

# Package info
centering-lgram info
centering-lgram version

# Use a different spaCy model
centering-lgram analyze --model en_core_web_lg --text "..."

# Pipe from stdin
cat document.txt | xargs centering-lgram score --text
```

---

## Use Cases

| Domain | Application |
|--------|-------------|
| **Linguistics Research** | Discourse analysis, coherence measurement, pronoun resolution studies |
| **Education** | Automated essay scoring, writing assistant feedback, readability analysis |
| **LLM Evaluation** | Assess coherence of generated text from GPT, Claude, Llama etc. |
| **Machine Translation** | Quality control for translation discourse integrity |
| **Summarization** | Evaluate summary-original coherence |
| **Dialogue Systems** | Measure conversation flow naturalness |
| **Content Quality** | Blog, news, academic writing fluency audits |
| **Forensic Linguistics** | Statement coherence analysis |

---

## How It Works

### Salience Ranking

Forward centers (Cf) are ordered by a weighted score combining:

1. **Grammatical role** — subject (4.0) > direct object (3.0) > indirect/oblique (2.0) > possessive (1.0)
2. **Part of speech** — pronoun (3.0) > proper noun (2.0) > common noun (1.0)
3. **Position** — earlier tokens are more salient
4. **Entity type** — PERSON, ORG, GPE entities get +1.5 bonus
5. **Pronoun antecedent** — pronouns with clear discourse antecedents get +1.0 bonus

### Backward Center Resolution

Cb is computed through a cascade of increasingly sophisticated methods:

1. **Direct match** — center appears in both previous and current Cf
2. **Possessive scan** — person-possessives (his, her) override direct matches
3. **Pronoun resolution** — personal/object/plural pronoun matching
4. **Coreference** — entity-type-based coreference (person-to-pronoun, object-to-pronoun)
5. **Compound plural** — multiple persons → "they" resolution

### Clause Detection

Sentences are split using spaCy's dependency parse. Subordinate clauses (advcl, ccomp, acl, relcl) and coordinated clauses (conj) are extracted, with separator tokens (commas, conjunctions) assigned to the following clause.

---

## Testing

```bash
# Install dev dependencies
pip install centering-lgram[dev]

# Run all tests
pytest tests/

# Run specific suites
pytest tests/test_lgram.py     # 15 core tests
pytest tests/test_edges.py     # 34 edge case / false positive tests
pytest tests/ -v               # verbose output
```

Test coverage includes: empty input, punctuation-only, long sentences, pronoun chains, possessive resolution, plural antecedents, clause extraction, state isolation, serialization roundtrip, and known limitations (gender disambiguation).

---

## Architecture

```
lgram/
  __init__.py         # Package init, metadata, exports
  core.py             # Re-export hub
  cli.py              # Command-line interface (6 commands)
  utils.py            # Logging utility
  models/
    __init__.py       # Sub-package exports
    centering_theory.py  # Core implementation (600 lines)
tests/
  test_lgram.py       # Core API tests (15 tests)
  test_edges.py       # Edge case / false positive tests (34 tests)
```

Single dependency: `spacy>=3.4.0`. No PyTorch, no transformers, no NumPy.

---

## License

MIT License — see [LICENSE](LICENSE)

Copyright (c) 2024 Ilker Atagun
