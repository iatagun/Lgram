import nltk
from nltk import word_tokenize, pos_tag
from itertools import chain
from get_gender import get_gender, gender_dict
import pandas as pd

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

class TransitionAnalyzer:
    def __init__(self, text):
        self.text = text

    def preprocess(self):
        sentences = nltk.sent_tokenize(self.text)
        sentence_pairs = [(sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)]
        return sentence_pairs

    def pos_tag_sentences(self, sentences):
        return [pos_tag(word_tokenize(sentence)) for sentence in sentences]

    def extract_entities(self, tagged_sentences):
        entities = []
        for sentence in tagged_sentences:
            entities.append(set(word for word, tag in sentence if tag in ['NNP', 'PRP', 'PRP$', 'NN', 'NNS']))
        return entities

    def get_gender(self, entity):
        if entity in gender_dict:
            return gender_dict[entity]
        return None

    def classify_transition(self, current_entities, next_entities, current_sentence, next_sentence):
        specific_current_entities = {e for e in current_entities if e[0].isupper()}
        specific_next_entities = {e for e in next_entities if e[0].isupper()}
        pronouns = {'She': 'female', 'He': 'male', 'It': None, 'They': None}

        transition_words = set()

        # Handle direct pronoun references by mapping pronouns to entities
        for pronoun, gender in pronouns.items():
            if pronoun in next_entities:  # Check pronoun in the next sentence
                # Find the entity in the current sentence that matches the gender of the pronoun
                for entity in specific_current_entities:
                    if gender and self.get_gender(entity) == gender:
                        transition_words.add((pronoun, entity))  # Correctly map the pronoun to the entity
                        break  # Break after finding a match

        # If no pronoun match is found, check if there is direct name-to-name retention
        if specific_current_entities & specific_next_entities:
            transition_words.update((e, e) for e in specific_current_entities & specific_next_entities)

        if transition_words:
            return "Retaining Transition", transition_words

        # Check for entity overlap
        if current_entities & next_entities:
            if len(next_entities - current_entities) == 0:
                return "Continuation Transition", current_entities & next_entities
            else:
                return "Retaining Transition", current_entities & next_entities
        elif len(next_entities) > 0 and len(current_entities) > 0:
            return "New Topic Transition", (current_entities, next_entities)
        else:
            return "Shift Transition", (current_entities, next_entities)

    def analyze(self):
        sentence_pairs = self.preprocess()
        results = []

        for current_sentences, next_sentences in sentence_pairs:
            current_tagged = self.pos_tag_sentences([current_sentences])
            next_tagged = self.pos_tag_sentences([next_sentences])

            current_entities = self.extract_entities(current_tagged)
            next_entities = self.extract_entities(next_tagged)

            combined_current_entities = set(chain.from_iterable(current_entities))
            combined_next_entities = set(chain.from_iterable(next_entities))

            transition, transition_words = self.classify_transition(combined_current_entities, combined_next_entities, current_sentences, next_sentences)

            # Sonuçları yazdır
            results.append({
                'current_sentences': current_sentences,
                'next_sentences': next_sentences,
                'transition': transition,
                'current_entities': combined_current_entities,
                'next_entities': combined_next_entities,
                'current_words': current_sentences.split(),  # Cümledeki kelimeler
                'next_words': next_sentences.split(),         # Cümledeki kelimeler
                'transition_words': transition_words          # Geçiş türüne neden olan sözcük ikilileri
            })

        return results
    def prepare_training_data(results):
        data = []
        for result in results:
            features = {
                'current_entities': ' '.join(result['current_entities']),
                'next_entities': ' '.join(result['next_entities']),
                'current_words': ' '.join(result['current_words']),
                'next_words': ' '.join(result['next_words']),
                'transition': result['transition']
            }
            data.append(features)

        df = pd.DataFrame(data)
        return df
# Örnek kullanım:
text = (
    "Sarah went to the bakery to buy some bread. She has been going to this bakery for years. "
    "Sarah was excited to try the new sourdough bread. The bakery had recently started making it. "
    "It was a delightful surprise to find that the bakery was also offering pastries now."
)
analyzer = TransitionAnalyzer(text)
results = analyzer.analyze()


# Sonuçları yazdırın
for idx, result in enumerate(results):
    print(f"Pair {idx + 1}:")
    print("Current Sentences:", result['current_sentences'])
    print("Next Sentences:", result['next_sentences'])
    print("Transition Type:", result['transition'])
    print("Current Entities (Possible Centers):", result['current_entities'])
    print("Next Entities (Possible Centers):", result['next_entities'])
    print("Transition Words:", result['transition_words'])  # Geçiş türünü belirleyen sözcük ikilileri
    print()
def prepare_training_data(results):
    data = []
    for result in results:
        features = {
            'current_entities': ' '.join(result['current_entities']),
            'next_entities': ' '.join(result['next_entities']),
            'current_words': ' '.join(result['current_words']),
            'next_words': ' '.join(result['next_words']),
            'transition': result['transition']
        }
        data.append(features)

    df = pd.DataFrame(data)
    return df

df = prepare_training_data(results)
print(df)