# Discourse Analysis Methods — Literature Survey

> Research compiled for centering-lgram analysis layer design.
> Focus: methods applicable to automatic text cohesion/coherence evaluation.

---

## 1. Centering Theory & Extensions

### 1.1 Core Theory (Grosz, Joshi, Weinstein 1983/1995)

**Status: ✅ Implemented in `centering_theory.py`**

Three centers per utterance:
- **Cf (Forward centers)**: Entities introduced, ordered by salience
- **Cb (Backward center)**: Entity linking to previous discourse
- **Cp (Preferred center)**: Highest-ranked Cf

Transitions: Establish, Continue, Retain, Smooth-Shift, Rough-Shift

### 1.2 BFP Algorithm (Brennan, Friedman, Pollard 1987)

- Operationalizes centering for pronoun resolution
- **Key rule**: Prefer transitions that maintain Cb = Cp over those that don't
- **Relevance**: Already encoded in our transition weights

### 1.3 Strube & Hahn 1999 — Functional Centering

- Ranks centers by **information status** not grammatical role
- Given > New > Inferable hierarchy
- **Relevance**: Could add `givenness` scoring as alternative salience strategy

### 1.4 Poesio et al. 2004 — Corpus Study

- Large-scale evaluation of centering claims on real corpora
- Found centering violations in ~30% of naturally-occurring text
- **Relevance**: Validates our `validate_sequence()` approach

---

## 2. Entity-Based Coherence Models

### 2.1 Entity Grid (Barzilay & Lapata 2005/2008)

**HIGH PRIORITY for implementation**

- Models text as matrix: sentences × discourse entities
- Each cell = {S, O, X, -} (Subject, Object, eXists, Absent)
- Transition probability between entity role sequences
- **Key insight**: coherent texts have "smooth" entity role transitions
- Trains on coherent/incoherent sentence orderings

**Implementation approach:**
- Extract entities per sentence (already have Cf)
- Build role matrix (S, O, X, -) from dependency parse
- Compute transition probabilities between adjacent role vectors
- Score = sum of log probabilities

**Relevance score: 9/10** — directly complementary to Centering Theory

### 2.2 Entity Transition Model

- Simpler variant: just count role transitions per entity
- Coherent texts: entities tend to stay in same role (S→S, O→O)
- **Relevance**: Lightweight alternative to full entity grid

### 2.3 Kibble & Power 2004 — Constraint-Based Evaluation

- Formalizes centering as constraint satisfaction
- NOCB (No Cb violation) + COHERE (preference ordering)
- **Relevance**: Formal framework for our `validate_sequence()`

---

## 3. Lexical Cohesion (Halliday & Hasan 1976)

### 3.1 Lexical Chains (Morris & Hirst 1991)

**MEDIUM PRIORITY**

- Chains of semantically related words across text
- Uses WordNet/thesaurus for relatedness
- **Relevance**: spaCy vectors can approximate this

### 3.2 Cohesion Taxonomy

- Reference (pronouns, demonstratives) ✅ implemented
- Substitution (one, do, so) ❌
- Ellipsis (omitted elements) ❌ (requires deep syntax)
- Conjunction (however, therefore, and) ❌
- Lexical (repetition, synonymy, hyponymy) — partial via vectors

### 3.3 Latent Semantic Analysis (LSA) for Cohesion

- Foltz, Kintsch & Landauer 1998
- Cohesion = cosine similarity between adjacent sentence vectors
- **Relevance**: Already have `_vector_similarity`

---

## 4. Discourse Segmentation

### 4.1 TextTiling (Hearst 1994/1997)

**HIGH PRIORITY**

- Detects topic shifts using lexical co-occurrence
- Sliding window of token blocks → cosine similarity between blocks
- Valley in similarity = topic boundary
- Depth score = (peak_left - valley) + (peak_right - valley)

**Implementation approach:**
- Already have sentence-level vector similarity
- Smooth similarities, find valleys with depth >= σ
- Complementary to our `detect_boundaries()` (centering-based)

### 4.2 LCSeg (Galley et al. 2003)

- Uses lexical chains for segmentation
- **Relevance**: Lower — lexical chains are heavy

### 4.3 Bayesian Segmentation (Eisenstein & Barzilay 2008)

- Topic models (LDA) for segmentation
- **Relevance**: Too heavy for lightweight library

---

## 5. Graph-Based Discourse Analysis

### 5.1 Discourse Graph Construction

- Nodes = sentences, Edges = cohesion strength (similarity, entity overlap)
- Graph metrics: density, clustering coefficient, centrality
- **Relevance**: Could add `build_cohesion_graph()` method

### 5.2 LexRank (Erkan & Radev 2004)

- Graph-based sentence importance for summarization
- PageRank on sentence similarity graph
- **Relevance**: Summarization use case

---

## 6. Discourse Parsing

### 6.1 RST (Rhetorical Structure Theory — Mann & Thompson 1988)

- Hierarchical tree of discourse relations (nucleus-satellite)
- Relations: Elaboration, Contrast, Cause, Condition, etc.
- **Relevance**: Heavy — requires specialized parser (e.g., DPLP, StageDP)

### 6.2 PDTB (Penn Discourse Treebank — Prasad et al. 2008)

- Shallow discourse parsing: connectives + argument spans
- **Relevance**: Lighter than RST, still requires trained parser

### 6.3 Decision: Skip for now

- Both require large trained models
- Outside scope of lightweight spaCy-based library
- Future: integrate with `spacy-discourse` if community adopts it

---

## 7. Neural & LLM-Based Approaches

### 7.1 BERT/Transformers for Coherence

- Entity tracking via attention (not feasible without transformers)
- **Relevance**: Not in scope (dependency constraint)

### 7.2 LLM-as-Judge for Cohesion

- GPT/Claude evaluating its own output's cohesion
- Our `analyze_llm()` already provides this from the outside
- **Relevance**: ✅ implemented

### 7.3 Coherence Scoring with Sentence Transformers

- `all-MiniLM-L6-v2` for sentence embeddings → cosine similarity matrix
- Simple, effective, requires `sentence-transformers` package
- **Relevance**: Optional dependency for better similarity

---

## 8. Evaluation Frameworks

### 8.1 GCDC (GCDC — Grounded Coherence Detection Corpus)

- Lai & Tetreault 2018
- 4 domains: Yahoo, Clinton, Enron, Yelp
- Binary classification: coherent vs incoherent documents
- **Relevance**: Could benchmark against GCDC

### 8.2 WSJ Benchmark (Barzilay & Lapata 2008)

- Original news articles vs randomly permuted sentences
- Standard entity grid evaluation
- **Relevance**: Simple to replicate for validation

### 8.3 Permuted Document Task

- Shuffle sentences, measure coherence degradation
- Our `evaluate_cohesion` already shows this works well

---

## 9. Implementation Priorities

| Priority | Method | Effort | Impact |
|---|---|---|---|
| **P0** | Entity Grid model | Medium | High |
| **P0** | TextTiling integration | Low | High |
| **P1** | Lexical chain approximation | Low | Medium |
| **P1** | Cohesion graph construction | Medium | Medium |
| **P1** | GCDC-style benchmarking | Medium | High |
| **P2** | Sentence transformer integration | Low | Medium |
| **P2** | WSJ benchmark replication | Medium | Medium |
| **P3** | RST/PDTB integration | High | Low (heavy) |
| **P3** | LSA cohesion | Already done (vectors) | — |

---

## 10. Key Papers Reference

| Paper | Year | Key Contribution |
|---|---|---|
| Grosz, Joshi, Weinstein | 1983 | Centering Theory foundation |
| Grosz, Joshi, Weinstein | 1995 | Centering formalization (transitions) |
| Brennan, Friedman, Pollard | 1987 | BFP algorithm |
| Halliday & Hasan | 1976 | Cohesion in English (taxonomy) |
| Morris & Hirst | 1991 | Lexical chains |
| Hearst | 1994/1997 | TextTiling segmentation |
| Barzilay & Lapata | 2005/2008 | Entity grid coherence model |
| Foltz, Kintsch, Landauer | 1998 | LSA for cohesion |
| Mann & Thompson | 1988 | Rhetorical Structure Theory |
| Prasad et al. | 2008 | Penn Discourse Treebank |
| Erkan & Radev | 2004 | LexRank summarization |
| Lai & Tetreault | 2018 | GCDC benchmark |
