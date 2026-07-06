# CAEAS — Cohesion-Aware Essay Analysis System

**EFL essay feedback for teachers. Not a grading system.**

## What It Does

Analyzes student essays and tells the teacher what to look at:

- *"Paragraph 3 has a cohesion break — check this sentence."*
- *"This student drops subject pronouns (Turkish L1 transfer)."*
- *"5 grammar errors: subject-verb agreement, missing articles, wrong verb forms."*

Teacher keeps full control. All processing is local — no data leaves the machine.

## Architecture

```
Essay → 5 layers → Teacher feedback
  ├── Grammar     LanguageTool + local LLM deep check
  ├── Content     LM Studio / local LLM (offline, free)
  ├── Cohesion    Lgram Centering Theory (Grosz, Joshi, Weinstein 1983)
  ├── Surface     Readability + vocabulary metrics
  └── Mechanics   pyspellchecker (spelling, punctuation)
```

## Quick Start

```bash
pip install centering-lgram[essay]
python -m spacy download en_core_web_md
```

```python
from lgram.essay import CAEASGrader, Essay

grader = CAEASGrader()
essay = Essay(title="My Essay", text="Social media has changed...")
report = grader.analyze(essay)

print(report.cohesion_score)       # 0-100 pure cohesion
print(report.composite_indicator)  # all 5 layers combined
print(report.suggestion)           # teacher-facing recommendation
```

### With local LLM (LM Studio)

1. Download [LM Studio](https://lmstudio.ai), load any model, start server on port 1234
2. System auto-detects and uses it for content + grammar analysis — free, offline, private
3. LLM calls are gated and cached; long essays are compressed to high-signal sentence excerpts

```python
grader = CAEASGrader(use_llm=True)
```

## Test Status

```
167 tests passing (99 core + 42 CAEAS + 26 EFL)
```

## Scope

- **Target:** Turkish EFL learners, CEFR B1-C1
- **Positioning:** Formative feedback tool, not summative grading
- **Key differentiator:** L1-specific error detection (pro-drop, gender-neutral, article-less)
- **Benchmark:** Yavuz (2025) — EFL teachers vs ChatGPT/Bard on 5-dimension rubric

## Current Phase

Prototype functional. Seeking pilot institution for real-world calibration with teacher-scored essays.

## Documentation

- [Development Log](../../docs/CAEAS_DEVELOPMENT_LOG.md) — full build history + calibration protocol
- [CHANGELOG](../../CHANGELOG.md) — version history
- [V02 Plan](../../docs/CAEAS_V02_PLAN.md) — implementation plan (completed)
- [Model Completeness Report](../../MODEL_COMPLETENESS_REPORT.md) — token optimization + test coverage
- [Production Readiness](../../PRODUCTION_READINESS_ASSESSMENT.md) — SaaS viability assessment
- [examples/demo.py](../../examples/demo.py) — full feature demo
- [examples/full_test.py](../../examples/full_test.py) — 5-layer discriminative test

## Dependencies

```
Required:  spacy >= 3.4.0, en_core_web_md
Optional:  language-tool-python (grammar), pyspellchecker (mechanics),
           openai (LLM client for LM Studio)
```

## License

MIT — see [LICENSE](../../LICENSE).
