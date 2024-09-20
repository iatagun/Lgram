import spacy
from itertools import chain

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

class TransitionAnalyzer:
    def __init__(self, text):
        self.text = text

    def preprocess(self):
        # Split the text into sentences
        doc = nlp(self.text)
        sentences = list(doc.sents)
        sentence_pairs = [(sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)]
        return sentence_pairs

    def extract_nps(self, sentence):
        # Extract noun phrases from a given sentence
        return [chunk.text for chunk in sentence.noun_chunks]

    def classify_transition(self, current_entities, next_entities, prev_cb):
        cb = prev_cb
        cp = next_entities if next_entities else set()

        if not current_entities and not cp:
            return "Center Establishment (EST)"  # No entities present
        if cb == current_entities:  # Retaining current CB
            if cb == cp:
                return "Center Continuation (CON)"  # Same Center
            else:
                return "Center Retaining (RET)"  # Retaining, but different mention
        elif cb != current_entities:  # New entities introduced
            if cb & cp:  # Overlapping entities
                return "Smooth Shift (SSH)"  # Shift to a more salient Center
            else:
                # Check for proper nouns and pronouns
                proper_nouns = {e for e in current_entities if e.istitle()}
                pronouns = {e for e in next_entities if e in ["He", "She", "They", "It"]}
                
                if proper_nouns and pronouns:
                    return "Center Continuation (CON)"  # Pronoun refers back to a proper noun
                else:
                    return "Rough Shift (RSH)"  # Major shift, unrelated
        else:
            return "New Topic Transition"  # Unrelated entities

    def annotate_anaphoric_relations(self, current_entities, next_entities):
        anaphoric_info = {}
        for entity in current_entities:
            if entity in next_entities:
                anaphoric_info[entity] = {"type": "identity", "antecedent": entity}
        return anaphoric_info

    def analyze(self):
        sentence_pairs = self.preprocess()
        results = []
        prev_cb = set()

        for current_sentence, next_sentence in sentence_pairs:
            current_nps = self.extract_nps(current_sentence)
            next_nps = self.extract_nps(next_sentence)

            combined_current_nps = set(current_nps)
            combined_next_nps = set(next_nps)

            transition = self.classify_transition(combined_current_nps, combined_next_nps, prev_cb)
            anaphoric_relations = self.annotate_anaphoric_relations(combined_current_nps, combined_next_nps)

            prev_cb = combined_current_nps  # Update CB for next iteration

            results.append({
                'current_sentences': current_sentence.text,
                'next_sentences': next_sentence.text,
                'transition': transition,
                'current_nps': combined_current_nps,
                'next_nps': combined_next_nps,
                'anaphoric_relations': anaphoric_relations
            })

        return results

# Example text
text = (
    "Alice was excited about her upcoming vacation. She had been planning it for months. "
    "Her friend Bob decided to join her. They both agreed that visiting Paris would be the highlight of the trip. "
    "Bob had always dreamed of seeing the Eiffel Tower. Later, they discussed what to do in the city."
)

analyzer = TransitionAnalyzer(text)
results = analyzer.analyze()

# Print results
for idx, result in enumerate(results):
    print(f"Pair {idx + 1}:")
    print("Current Sentences:", result['current_sentences'])
    print("Next Sentences:", result['next_sentences'])
    print("Transition Type:", result['transition'])
    print("Current NPs (Possible Centers):", result['current_nps'])
    print("Next NPs (Possible Centers):", result['next_nps'])
    print("Anaphoric Relations:", result['anaphoric_relations'])
    print()
