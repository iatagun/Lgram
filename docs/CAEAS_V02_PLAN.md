# CAEAS v0.2 — Production Hardening Plan

## Scope: EFL Writing Feedback Tool (Turkish L1, CEFR B1-C1)

**Positioning change:** "Essay Assessment System" → "Cohesion-Aware Writing Feedback Tool"
**Philosophy:** Evidence provider, not authority. Formative, not summative.

---

## 1. Grammar/Cohesion Disambiguation Layer [HIGH]

### Problem
spaCy dependency parse assumes grammatical correctness. EFL texts contain:
- Missing subjects ("Is very important" → pro-drop)
- Wrong pronouns ("he" for "she" → gender transfer)
- Missing articles ("Cat sat on mat")
- Tense inconsistency

When spaCy misparses, Cb/Cf computation produces nonsense → false cohesion errors.

### Solution: `PreFilter` layer inserted BEFORE centering analysis

```
ESSAY TEXT
    │
    ├─► PreFilter (grammar check via LanguageTool, optional)
    │     ├─ Tag errors: GRAMMAR vs COHESION
    │     ├─ If pronoun error → flag as L1 transfer, NOT cohesion break
    │     ├─ If parse looks broken → downgrade confidence
    │     └─ Output: filtered text + error tags
    │
    ├─► Layer 2: Cohesion (Lgram) — on best-effort parse
    └─► Cross-reference: grammar errors vs cohesion findings
```

### Implementation
- `lgram/essay/prefilter.py`
  - `PreFilter` class with optional LanguageTool dependency
  - `analyze()` → returns `PreFilterReport` with error tags
  - Detection rules: pronoun mismatch, article omission, subject drop
  - Confidence downgrade when parse quality is suspect
- LanguageTool: optional, `pip install language-tool-python`
- Fallback: heuristic-based detection when LT not available

### Risk Mitigation
- spaCy still runs, but results tagged with confidence
- Grammar errors that overlap with cohesion → flagged for teacher review
- Never claims "this is definitely a cohesion error" when grammar is broken

---

## 2. CEFR-Level Calibration Pipeline [HIGH]

### Problem
Current CEFR_PROFILES are heuristic, not empirical.
Single cohesion threshold doesn't account for:
- B1 students use simple sentences → fewer cohesion breaks
- C1 students attempt complex structures → more risk, paradoxically lower scores

### Solution: Per-level calibration curves

```python
class CEFRCalibrator:
    """
    Requires: 30+ human-scored essays per CEFR level.
    Produces: per-level QWK, ICC, cohesion thresholds, and recalibration curves.
    """
    def calibrate_level(level: str, machine_scores, human_scores) -> LevelCalibration:
        ...
```

### Implementation
- `lgram/essay/cefr_calibration.py`
  - `CEFRCalibrator` class
  - `LevelCalibration` dataclass with per-level metrics
  - Separate cohesion thresholds per level (not shared)
  - Flag when student attempts above-level complexity → adjust expectations
- Update `CEFR_PROFILES` with empirical thresholds (placeholder until data)

### Complexity-Adjusted Scoring
- Sentence complexity proxy: avg dependency tree depth, clause count
- If student attempts complex structures: RAISE threshold tolerance
- If student uses only simple structures: LOWER ceiling (can't score above X)
- This prevents the paradox: "better writer gets lower cohesion score"

---

## 3. Terminology Audit [HIGH]

### Changes (global find-replace)

| Current | New | Reason |
|---|---|---|
| "coherence" | "cohesion" | We measure surface cohesion, not deep coherence |
| "assessment system" | "feedback tool" | Positioning: not authoritative |
| "judge" (LLM) | "content analyzer" | "Judge" implies final authority |
| "verdict" | "suggestion" | System recommends, human decides |
| "grade" (verb) | "analyze" or "review" | Not assigning grades |
| "score" (in user-facing text) | "cohesion indicator" or keep score but frame as "estimated" | Transparency |
| "EXCELLENT/GOOD/etc" | Replace with descriptive feedback | Categorical labels are value judgments |
| "CAEAS" acronym expansion | "Cohesion-Aware" not "Coherence-Aware" | Accuracy |

### Files affected
- `lgram/essay/__init__.py` — module docstring
- `lgram/essay/models.py` — `verdict` → `suggestion`, class docs
- `lgram/essay/grader.py` — method names, verdict strings, docstrings
- `lgram/essay/layer5_confidence.py` — trigger messages, justification text
- `lgram/essay/layer1_content.py` — "judge" → "analyzer", docstrings
- `lgram/essay/efl.py` — rubric description
- `lgram/essay/layer2_cohesion.py` — docstrings

---

## 4. Feedback-Mode Positioning [HIGH]

### Changes

**Before (v0.1):**
```
Verdict: GOOD (72/100) — solid performance with minor areas for improvement.
```

**After (v0.2):**
```
Cohesion analysis complete. Estimated overall cohesion indicator: 72/100 (CI: 65-79).

Key observations:
  - Organization: Strong paragraph-level cohesion (89/100)
  - Surface quality: Adequate sentence variety, vocabulary could be richer
  - Content indicators suggest thesis is present but evidence is limited

Suggested teacher review points:
  - Paragraph 2, sentence 3: Pronoun reference may be ambiguous
  - Consider: Does this paragraph need a clearer transition from the previous one?

NOTE: This is automated feedback for teacher consideration, not a grade.
The teacher's professional judgment is the final authority.
```

### Implementation
- Rename `CAEASReport.verdict` → `CAEASReport.suggestion`
- Replace categorical labels with descriptive observations
- Add teacher-oriented framing to all output
- `grader.grade()` → `grader.analyze()` (new name, old kept as deprecated alias)
- Remove "grade boundary" language from confidence layer
- Change threshold names: "pass/fail" → "review recommended"

---

## 5. Data Collection & Export Module [MEDIUM]

### Purpose
Enable research-quality data collection for:
- CEFR-level calibration
- Error typology analysis
- Cross-institutional comparison

### Implementation
- `lgram/essay/export.py`
  - `export_calibration_json(essays, reports)` → structured JSON
  - `export_error_typology(reports)` → error frequency table (CSV/JSON)
  - `anonymize_report(report)` → strip identifying text
  - `compare_institutions(reports_a, reports_b)` → cross-cohort stats
- Output format aligned with ICLE/TOEFL11 conventions where possible

---

## 6. Error Typology Framework [MEDIUM]

### Purpose
Systematically categorize cohesion issues in EFL writing.
Creates research contribution independent of product.

### Implementation
- `lgram/essay/typology.py`
  - `ErrorType` enum: PRONOUN_REFERENCE, MISSING_TRANSITION, TOPIC_SHIFT, L1_TRANSFER, ...
  - `TypologyReport` dataclass: frequency, severity, examples, L1 correlation
  - `build_typology(reports: List[CAEASReport])` → error distribution
  - Framework for publishing: "Common cohesion errors in Turkish EFL writing"

### Error Categories (initial)
1. **PRONOUN_AMBIGUITY** — unclear antecedent (L1 transfer: Turkish pro-drop)
2. **MISSING_COHESIVE_DEVICE** — no transition where expected
3. **ABRUPT_TOPIC_SHIFT** — sudden change without signaling
4. **OVERUSE_REPETITION** — repeated full NP instead of pronoun
5. **GENDER_MISMATCH** — he/she confusion (L1: Turkish "o" = no gender)
6. **ARTICLE_OMISSION** — affects referent tracking (L1: Turkish = no articles)

---

## Implementation Order

| Step | File | Priority | Est. Time |
|---|---|---|---|
| 1. PreFilter (grammar/cohesion) | `prefilter.py` (new) | HIGH | 30 min |
| 2. CEFR calibration | `cefr_calibration.py` (new) | HIGH | 25 min |
| 3. Terminology audit | 7 files (edit) | HIGH | 20 min |
| 4. Feedback positioning | `grader.py`, `models.py`, `layer5_confidence.py` | HIGH | 25 min |
| 5. Complexity-adjusted scoring | `cefr_calibration.py` | MEDIUM | 15 min |
| 6. Data export | `export.py` (new) | MEDIUM | 15 min |
| 7. Error typology | `typology.py` (new) | MEDIUM | 20 min |
| 8. Tests + integration | `tests/test_prefilter.py` etc | HIGH | 20 min |

**Total:** ~2.5 hours

---

## What We Do NOT Change

- Core Lgram (centering_theory.py) — stays as-is
- Existing TextAnalyzer API — unchanged
- Layer 4 (PopulationCalibrator) — stays, CEFRCalibrator complements it
- Layer 5 confidence math — stays, only messaging changes
- pyproject.toml dependencies — LanguageTool stays optional
