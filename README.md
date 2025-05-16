
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

## Outputs

The victim must get to go to the woods and learn the true meaning of the word, "she continued. " According to Gallup, in 2002, they would not say willingly. Your opinion is needed. At some moment he might not think of himself as a genius, but as the mere fact of being alive. He chose art.

The murder of Agamemnon would send shivers down your backbone. It is no use trying to suppress that side of myself. In her dead mind there is nothing which appears to her as being outside, and what is outside is what He has left behind. There is evidence to support the view that he has seen such a mention as an occasional burst of electricity and, I am sure, no trace of it. His plan would have been to attack the house and burn down the garden with some kind of fire. While her husband was away, she had made an excuse for being late. The size of these ships is unknown.

The murderer will say that this way if indeed the former is the case. In spring of 1992, I would have noticed. The gun swung and he asked the woman what she wanted, but she did not say anything. With the information that we have, we are able to attach more importance to what we do not need. To be near the rest of the world in this case is a matter of great importance.

The crime was committed by an old man. Indeed, so bad is the weather that we sometimes talk about the reason why he was in Berlin and the city itself. The reason is that he is a rich man. Grass has not grown in this day and age, so it is not suitable for making friends or building communities. I find myself in a bar and ask for a drink when I see a new world around me.

The time has come for you to do something in return and to observe what is happening in the present as closely as possible to one possibility. Mr. Phillips's protest was that softening the blow by a gentle breeze and creating a myriad of other sounds was wrong for the foot to be moved. He just pointed out that the world around him had gone wrong and that the course of events in his country would be too difficult for him to change. Of course, Miss Diana told me that she found it very pleasant to hear that he would return to private life. So say much as well as a bird would be a woman's best friend.