# Changelog

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
