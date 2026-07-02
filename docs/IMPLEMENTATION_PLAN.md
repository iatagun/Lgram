# Implementation Plan — Analysis Layer v2.2

> Based on literature survey (RESEARCH.md). Builds on TextAnalyzer in `lgram/analyzer.py`.

---

## Phase 1: Entity Grid Model (P0)

**Goal**: Add entity-role-based coherence scoring complementary to Centering Theory.

### Method

For a text with N sentences and M entities:

1. **Extract entities** per sentence (reuse Cf from Centering Theory)
2. **Determine grammatical role** per entity per sentence:
   - `S` = Subject (nsubj, nsubjpass)
   - `O` = Object (dobj, iobj, pobj)
   - `X` = Present but not S or O (other dependency)
   - `-` = Absent
3. **Build transition matrix**: for each entity, compute role transition probabilities
   (P(S|S), P(O|S), P(X|S), P(-|S), P(S|O), ...)
4. **Score**: sum of log probabilities of observed role transitions

### Implementation

```python
# New method on TextAnalyzer
def entity_grid_score(text: str) -> float:
    pass

# New dataclass
@dataclass
class EntityGrid:
    entities: List[str]
    matrix: List[List[str]]  # sentences x entities, values: S/O/X/-
    transition_probabilities: Dict[Tuple[str,str], float]
    score: float
```

### Files affected
- `lgram/analyzer.py` — new methods
- `lgram/models/centering_theory.py` — no change needed (reuses Cf)

### Tests
- 5 known coherent/incoherent sentence orderings
- GCDC-style permutation test
- WSJ article ordering benchmark

---

## Phase 2: TextTiling Integration (P0)

**Goal**: Lexical cohesion-based segmentation as complement to centering-based `detect_boundaries()`.

### Method

1. Split text into pseudo-sentences (every ~20 tokens)
2. Build token frequency vectors per block
3. Compute cosine similarity between adjacent blocks
4. Smooth similarity curve
5. Find valleys where `depth_score > threshold`
6. Merge with Centering Theory boundaries (intersection = high confidence)

### Implementation

```python
# New method on TextAnalyzer
def texttile_segments(text: str, w: int = 20, k: int = 10) -> List[int]:
    """TextTiling segmentation — sentence indices of boundaries."""
    pass

def hybrid_boundaries(text: str) -> List[int]:
    """Intersection of centering + TextTiling boundaries (high confidence)."""
    pass
```

### Files affected
- `lgram/analyzer.py` — new methods

---

## Phase 3: Cohesion Graph (P1)

**Goal**: Build and analyze a sentence-to-sentence cohesion graph.

### Method

1. Nodes = sentences
2. Edges = cohesion weight (entity overlap + vector similarity + transition score)
3. Compute graph metrics:
   - **Global cohesion**: graph density
   - **Local cohesion**: average neighbor similarity
   - **Topic drift**: modularity-based community detection
   - **Key sentences**: eigenvector centrality

### Implementation

```python
@dataclass
class CohesionGraph:
    adjacency: List[List[float]]
    density: float
    avg_similarity: float
    communities: List[List[int]]
    central_sentences: List[int]

# New method on TextAnalyzer
def build_cohesion_graph(text: str) -> CohesionGraph:
    pass
```

### Files affected
- `lgram/analyzer.py` — new methods

### Dependencies
- Pure Python (no external graph lib needed; adjacency matrix + NumPy optional)

---

## Phase 4: Lexical Chain Approximation (P1)

**Goal**: Lightweight lexical cohesion via vector similarity chains.

### Method

1. Extract all nouns per sentence
2. Build chain: noun_A(t1) → noun_B(t2) if similarity > threshold
3. Chain length = cohesion strength
4. Score = mean chain length / max possible

### Implementation

```python
def lexical_chain_score(text: str, threshold: float = 0.5) -> float:
    pass
```

---

## Phase 5: Benchmark Suite (P1)

**Goal**: Standardized evaluation against known benchmarks.

### GCDC-style Test

```python
def benchmark_gcdc_style() -> Dict:
    """Run cohesion scoring on coherent vs permuted documents."""
    pass
```

### WSJ Benchmark

```python
def benchmark_wsj() -> Dict:
    """Replicate Barzilay & Lapata 2008 entity grid evaluation."""
    pass
```

### Built-in Test Suite

- 10 coherent document samples
- 10 permuted variants
- Expect: coherent > permuted for all metrics

---

## Phase 6: Sentence Transformer Support (P2)

**Goal**: Optional `sentence-transformers` dependency for better similarity.

### Method

- If `sentence-transformers` installed, use `all-MiniLM-L6-v2`
- Otherwise fall back to spaCy vectors (current behavior)
- Lazily loaded, zero impact if not installed

```python
class TextAnalyzer:
    def __init__(self, use_sentence_transformers: bool = False):
        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                self._st_model = None
```

---

## Architecture After Implementation

```
lgram/
  analyzer.py
    ├── TextAnalyzer
    │   ├── analyze()              # existing — full text analysis
    │   ├── analyze_batch()        # existing — batch comparison
    │   ├── entity_grid_score()    # NEW — entity grid model
    │   ├── texttile_segments()    # NEW — TextTiling
    │   ├── hybrid_boundaries()    # NEW — centering + TextTiling
    │   ├── build_cohesion_graph() # NEW — graph analysis
    │   ├── lexical_chain_score()  # NEW — lexical chains
    │   ├── to_dict/to_json/to_summary()  # existing — exports
    │   └── benchmark()            # NEW — built-in evals
    │
    ├── EntityGrid        # NEW dataclass
    ├── CohesionGraph     # NEW dataclass
    ├── TextReport        # existing
    ├── ParagraphAnalysis # existing
    └── SentenceAnalysis  # existing
```

---

## Timeline

| Phase | Description | Est. effort | Priority |
|---|---|---|---|
| P0 | Entity Grid Model | 2 hours | Critical |
| P0 | TextTiling | 1 hour | Critical |
| P1 | Cohesion Graph | 1.5 hours | High |
| P1 | Lexical Chains | 0.5 hours | High |
| P1 | Benchmark Suite | 1 hour | High |
| P2 | Sentence Transformers | 0.5 hours | Medium |
| P2 | WSJ Benchmark | 1 hour | Medium |

---

## Success Criteria

After implementation, a `TextAnalyzer` instance should:

1. **Score coherent text higher than random order** (baseline: current Centering-only already does this)
2. **Entity grid adds orthogonal signal** — some texts with same centering score get different entity grid scores
3. **TextTiling boundaries agree with human annotations** on standard test sets
4. **Cohesion graph metrics correlate with human coherence ratings**
5. **Benchmark suite runs < 30s** on 20 short documents
6. **No new required dependencies** (sentence-transformers optional)
