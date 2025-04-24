
# 🧠 Centering-Based Coherence & N-Gram Language Generation Framework

This project is a comprehensive framework that combines **Centering Theory**, **Transition Analysis**, **Contextual N-gram Generation**, and **Grammatical Correction** to produce **contextually coherent sentences** and linguistically analyze the output.

## 🔍 Main Components

### 1. `CenteringModel`
Defines transition types between sentence pairs based on **Centering Theory** (`CON`, `RET`, `SSH`, `RSH`, `EST`, `NTT`) and calculates a **total coherence score** by assigning weights to each transition.

- Input: Free-form text (str)
- Output: Transition scores, types, and detailed pairwise information

### 2. `TransitionAnalyzer`
Analyzes sentence pairs to extract:
- **Noun phrases** (`noun_chunks`)
- **Anaphoric relations**
- **Transition types**

This analysis supports both statistical and linguistic evaluation.

### 3. `EnhancedLanguageModel`
Generates context-aware, fluent sentences using a **Kneser-Ney smoothed n-gram model** enhanced with POS tagging.

#### Key Features:
- Generation using 2- to 6-gram models
- Syntactic analysis and centering using `SpaCy`
- Linguistic center tracking via `get_center_from_sentence`
- Contextual word selection via `choose_word_with_context`
- Completeness check via `is_complete_thought`
- Theme consistency via `post_process_sentences`

### 4. `dynamicngramparaphraser.py`
Performs **contextual paraphrasing** based on n-grams. Selects the **best alternative match** for each word depending on its position and syntactic role.

- Supports **dependency-based reordering** (`reorder_sentence`)
- Combines vector similarity and frequency with `select_best_match`

### 5. `analyze_transitions.py`
Invokes the `CenteringModel` to analyze all sentence transitions in a text and returns the results as a `DataFrame`, including:
- `current_sentence`
- `next_sentence`
- `transition_type`
- `score`
- `total_score`

## 🗂 File Structure

.
├── analyze_transitions.py
├── centering_model.py
├── chunk.py
├── dynamicngramparaphraser.py
├── simple_language_model.py
├── get_gender.py
├── transition_analyzer.py
├── corrections.json
├── ngrams/
│   ├── bigram_model.pkl
│   ├── trigram_model.pkl
│   ├── fourgram_model.pkl
│   ├── fivegram_model.pkl
│   ├── sixgram_model.pkl
│   └── text_data.txt

## 🚀 Usage Example

### Transition Analysis
```python
from analyze_transitions import analyze_transitions

text = "Least of all do they thus dispose of the murdered. Guardsman take small farmer well who loathe every precaution the officer."
df = analyze_transitions(text)
print(df)
```

### Sentence Generation
```python
from simple_language_model import EnhancedLanguageModel

model = EnhancedLanguageModel("Some training text.")
sentence = model.generate_sentence(start_words=["The", "man"], length=12)
print("Generated:", sentence)
```

### Coherence Report
```python
sentences = ["The man left.", "She stayed at home."]
cleaned_sentences, report = model.post_process_sentences(sentences)
print(report)
```

## 🛠 Requirements

- Python 3.8+
- `spacy`
- `numpy`
- `scikit-learn`
- `tqdm`
- `pandas`

### SpaCy Model
```bash
python -m spacy download en_core_web_lg
```

## 🎯 Purpose

This project provides a powerful infrastructure for researchers, developers, and linguistics enthusiasts working in **textual coherence**, **discourse transition**, and **automated sentence generation**. Whether you're generating text, analyzing transitions, or evaluating textual consistency — this is your **linguistic lab**.

## 📫 Contact

Suggestions, contributions, or even coffee invites are welcome:  
**Linguistic Alchemist** | [lgram.site](https://lgram.site)
