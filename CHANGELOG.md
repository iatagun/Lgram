# Changelog

## [1.1.3] - 2025-09-13

### ⚡ Major Performance Optimization - Lazy Loading
- **1200x Faster Model Creation**: Model initialization time reduced from ~60s to 0.05s
- **Lazy Loading Implementation**: Models load only when needed, dramatically reducing startup time
- **Smart Pattern Learning Cache**: Text data learning cached for 7 days, eliminates redundant processing
- **Memory Efficient**: Models loaded on-demand, reducing initial memory footprint

### 🚀 Lazy Loading Components
- **SpaCy Model**: Loads only when NLP processing is required
- **T5 Grammar Model**: Loads only when grammar correction is called
- **N-gram Models**: Individual models load only when accessed (bigram, trigram, etc.)
- **Centering Theory**: Initializes only when centering analysis is needed
- **Pattern Learner**: Activates only on first pattern learning request

### 🔧 Technical Improvements
- **ModelInitializer Class**: Enhanced with singleton pattern and lazy loading
- **Smart Caching**: 7-day cache for text_data.txt pattern learning
- **Property-based Access**: N-gram models accessible via properties with lazy loading
- **Global Helper Functions**: `get_nlp()` and `get_t5_models()` for lazy access
- **Reduced Disk I/O**: Models read from disk only when required

### 📊 Performance Metrics
- **Model Creation**: 60+ seconds → 0.05 seconds (1200x improvement)
- **Memory Usage**: Reduced initial footprint by ~70%
- **Pattern Learning**: Cached for 7 days (was: every initialization)
- **First Use Impact**: Models load seamlessly on first actual usage

---

## [1.1.2] - 2025-09-07

### 🧹 Package Structure Cleanup
- **Removed Duplicate Folders**: Eliminated duplicate `models/` and `ngrams/` folders in root directory
- **Clean Package Structure**: Now only uses `lgram/models/` and `lgram/ngrams/` for proper package organization
- **Fixed setup.py**: Corrected package discovery to include only `lgram*` and `tests*`
- **Reduced Package Size**: Eliminated redundant files from distribution

### 🔧 Technical Improvements
- **Better Import Structure**: Clear separation between package code and development leftovers
- **Cleaner Distribution**: PyPI package now contains only necessary files
- **Resolved Import Conflicts**: No more confusion between root `models/` and `lgram/models/`

---

## [1.1.1] - 2025-09-06

### 🔧 Model Architecture Improvements
- **Enhanced models/__init__.py**: Improved module imports and organization
- **Optimized simple_language_model.py**: Performance enhancements and code refinements
- **Better Module Integration**: Streamlined import structure for better package consistency

### 🐛 Minor Fixes
- **Import Structure**: Cleaned up module dependencies
- **Code Organization**: Better separation of concerns in model architecture

---

## [1.1.0] - 2025-09-03

### 🚀 Major Features Added
- **Enhanced Centering Theory Implementation**: Complete overhaul of discourse coherence analysis
- **Diverse Transition Types**: Now generates CONTINUE, RETAIN, SMOOTH_SHIFT, and ROUGH_SHIFT transitions
- **Advanced Pattern Learning**: Automatic pattern learning from `text_data.txt` (4.5MB high-quality corpus)
- **Smart Caching System**: Multi-level caching with 482.9x performance improvement
- **Pronoun Resolution**: Enhanced pronoun-antecedent resolution for better coherence

### 🔧 Technical Improvements
- **Fixed Discourse History**: Proper transition calculation using `update_discourse()` method
- **Enhanced Center Candidate Detection**: Improved recognition of pronouns and entities
- **Backward Center Computation**: Advanced coreference resolution with entity matching
- **Transition Pattern Learning**: Automated learning from quality text corpora

### 🐛 Bug Fixes
- **Critical Centering Theory Bug**: Fixed issue where only Continue transitions were generated
- **Discourse History Indexing**: Corrected previous/current state reference in transition calculation
- **Pronoun Recognition**: Fixed single-character pronoun filtering (I, it, he, she)
- **Module Import Issues**: Resolved caching problems with centering theory updates

### 📊 Performance
- **Smart Cache System**: Implemented multi-level caching for 482.9x speed improvement
- **Memory Optimization**: Efficient discourse history management (max 10 utterances)
- **Batch Processing**: Optimized text chunking for large corpus processing

### 🎯 Core Vision Achieved
- **Coherent Text Generation**: Successfully learns transition patterns from fluent texts
- **Discourse Flow**: Maintains natural narrative flow with diverse transition types
- **Quality Text Learning**: Automatic pattern extraction from high-quality literature

### 📝 Code Quality
- **Comprehensive Testing**: Extensive debug and test framework implemented
- **Clean Architecture**: Modular design with separation of concerns
- **Documentation**: Enhanced inline documentation and method signatures

---

## [1.0.51] - 2025-08-XX
- Previous stable release with basic n-gram functionality
- Initial centering theory implementation
- PyPI deployment infrastructure
