# Production Release Plan — centering-lgram v2.2.0

## Status: 64 tests passing, zero bugs

---

## Phase 1: Package Polish (30 min)

- [ ] [b1] `pyproject.toml` — add `python -m spacy download en_core_web_sm` as post-install? No (PyPI policy). Add warning in README.
- [ ] [b2] Add `exclude` to `[tool.setuptools.packages.find]` to prevent tests in wheel (tests already included, fine for now)
- [ ] [b3] Verify `pip install centering-lgram` from PyPI works end-to-end
- [ ] [b4] Add `long_description` from README.md to pyproject.toml (already done)
- [ ] [b5] Add version to `centering-lgram --version` output (already done)

## Phase 2: Docker (15 min)

- [ ] [b6] Create `Dockerfile` — multi-stage, Python 3.12-slim, auto-download en_core_web_sm
- [ ] [b7] Create `docker-compose.yml` — ready-to-run example
- [ ] [b8] Test: `docker build -t centering-lgram . && docker run centering-lgram analyze --text "Test."`

## Phase 3: CI/CD (20 min)

- [ ] [b9] Create `.github/workflows/test.yml` — run pytest on push/PR
- [ ] [b10] Create `.github/workflows/publish.yml` — build + publish to PyPI on tag push
- [ ] [b11] Add badge to README: [![Tests](https://github.com/iatagun/Lgram/actions/workflows/test.yml/badge.svg)]

## Phase 4: Examples (15 min)

- [ ] [b12] Create `examples/` directory
- [ ] [b13] `examples/basic_analysis.py` — 10-line demo
- [ ] [b14] `examples/llm_evaluation.py` — analyze LLM output
- [ ] [b15] `examples/compare_texts.py` — compare two versions
- [ ] [b16] `examples/streamlit_demo.py` — web interface (optional)

## Phase 5: Final Checks (15 min)

- [ ] [b17] Run full test suite one final time
- [ ] [b18] `pip install -e .` test in clean venv
- [ ] [b19] Verify `centering-lgram --help` works
- [ ] [b20] Verify `from lgram import TextAnalyzer` works
- [ ] [b21] Git tag: `git tag v2.2.0 && git push --tags`
- [ ] [b22] `twine upload dist/*` to PyPI

## Phase 6: Post-Release (on main)

- [ ] [b23] Merge `production/v2.2.0` → `main`
- [ ] [b24] Update PyPI description
- [ ] [b25] Announce (LinkedIn, Twitter, Reddit r/Python, HN)

---

## Estimated Total: ~2 hours

All items are independent within their phase. Can be done in any order.
