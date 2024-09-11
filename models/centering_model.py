import nltk
from nltk import word_tokenize, pos_tag
from itertools import chain

# NLTK'nin gerekli verilerini indirin
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class TransitionAnalyzer:
    def __init__(self, text):
        self.text = text

    def preprocess(self):
        # Metni cümlelere ayırın
        sentences = nltk.sent_tokenize(self.text)
        # Cümle çiftlerini oluşturun
        sentence_pairs = [(sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)]
        return sentence_pairs

    def pos_tag_sentences(self, sentences):
        # Her cümleyi POS etiketleri ile işleyin
        return [pos_tag(word_tokenize(sentence)) for sentence in sentences]

    def extract_entities(self, tagged_sentences):
        # Her cümledeki varlıkları çıkartın
        entities = []
        for sentence in tagged_sentences:
            entities.append(set(word for word, tag in sentence if tag in ['NNP', 'PRP', 'PRP$', 'NN', 'NNS']))
        return entities

    def classify_transition(self, current_entities, next_entities):
        # Centering Theory'ye göre geçiş türlerini sınıflandırın
        if current_entities & next_entities:
            if len(next_entities - current_entities) == 0:
                return "Continuation Transition"
            else:
                return "Retaining Transition"
        elif len(next_entities) > 0 and len(current_entities) > 0:
            return "New Topic Transition"
        else:
            return "Shift Transition"

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
            
            transition = self.classify_transition(combined_current_entities, combined_next_entities)
            results.append({
                'current_sentences': current_sentences,
                'next_sentences': next_sentences,
                'transition': transition,
                'current_entities': combined_current_entities,
                'next_entities': combined_next_entities
            })

        return results

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
    print()
