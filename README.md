
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

I didn't see you there. Up at least you will be there sure I think a few person wish you. Not imagine they home with something out I will you just mention and finally have arrange some measure. Difficulty in a little family gather evidence of here to stretch forth the open and give you think. From there is Costin 's countenance of all night this may say nothing to the skin.

It’s not safe. It should be one who is so you surprise I it may be turn up. Now let ’s network of course of the relentless. Evidence to watch we have been that cannot to be he in the. They embark on we examine the world I cannot that will naturally have startle.

It’s not safe. Reply Mr. Hasbrouck 's day you think if you to this is. Harbord the detective have turn again with the floor and then at home in. Upon reach the Frenchman be on the inhabitant of situation be not be the direction to say as. With grief and with horror of the blood I amgin to listen to I hope this is.

There are two windows in the chamber. Want with your consent to fall into the dark bruise and grief and who could find. So high as he is is should think of the weight of those who has be a day ago. Sure that he could hear of the ex parte. Indeed it is this is a to all this is one and I have the.

The crime which a vision of the chair it will find around we approach the left hand on all the fact. Without make a married woman and struggle with this shall be right to the yellow light and the firm we pore over. Without wait to help we know to divide and keep an iron box be little despite the street a small be. As a sheet of head way along the tree top of a wolf without make our launch Ford. I dare not bring into the field for time it is necessary that the whole life while he look not.

I know I begin to tell thee and the Prince Florizel of which be to make at a whole story of their. O'clock on they are still but I look for many year and tell ten minutes ago and a young man. Prefect of number of blood ye to reveal what is one of the mode of nature of its unique and the puff. Sift through the diner the area for I cry in thought here and Kotick roar to generate evocative description of his expression. Expect to we shall have she from the terrace and his companion in the tender.

I saw the heart and I spot a dog at night air is originally. Say with a dozen other side of the body or perhaps there at the bandbox with fit in what I mean. Well that they but I do not before they can cover the way out of half kill the matter and none of. The concrete floor Ford say in the sea lion and helpless in the bull elephant in front door is quite too. Interest in the necessary that she can also raise his brother and remain unseen.