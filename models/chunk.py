import random
import pickle
from collections import defaultdict
import spacy
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
import os
import re
import json
from transition_analyzer import TransitionAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Load the SpaCy English model with word vectors
nlp = spacy.load("en_core_web_lg")  # Use medium model for better embeddings
nlp.max_length = 1030000 # or even higher
# N-gram modellerini kontrol et ve yükle
text_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\text_data.txt"
bigram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\bigram_model.pkl"
trigram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\trigram_model.pkl"
fourgram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\fourgram_model.pkl"
fivegram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\fivegram_model.pkl"
sixgram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\sixgram_model.pkl"
corrections_file = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\corrections.json"

class EnhancedLanguageModel:
    def __init__(self, text, n=2):
        self.n = n  # N-gram size
        self.model, self.total_counts = self.build_model(text)  # Combine model and total counts
        self.load_ngram_models()  # N-gram modellerini yükle
        self.dummy_text = "dummy_text"
        self.correct_grammar(self.dummy_text)
        
        
    def build_model(self, text):
        model = defaultdict(lambda: defaultdict(int))
        doc = nlp(text.lower())
        
        # Create n-grams
        tokens = [token.text for token in doc if token.is_alpha]  # Filter out non-alpha tokens
        n_grams = [tuple(tokens[i:i+self.n]) for i in range(len(tokens)-self.n+1)]
        
        for n_gram in n_grams:
            prefix = n_gram[:-1]  # All but last word
            next_word = n_gram[-1]
            model[prefix][next_word] += 1
        
        # Normalize probabilities using Kneser-Ney smoothing
        total_counts = defaultdict(int)
        for prefix, next_words in model.items():
            total_count = sum(next_words.values())
            total_counts[prefix] = total_count
            
            # Kneser-Ney smoothing
            for word in next_words:
                # Use max(0, count) to avoid negative counts
                next_words[word] = (next_words[word] + 1) / (total_count + len(next_words))

            # Implement Kneser-Ney continuation probability
            for word in next_words:
                continuation_count = sum(1 for ngram in n_grams if ngram[-1] == word)
                next_words[word] += (continuation_count / len(tokens))

        return dict(model), dict(total_counts)  # Convert to regular dicts

    def load_ngram_models(self):
        with open(bigram_path, 'rb') as f:
            self.bigram_model = pickle.load(f)
        with open(trigram_path, 'rb') as f:
            self.trigram_model = pickle.load(f)
        with open(fourgram_path, 'rb') as f:
            self.fourgram_model = pickle.load(f)
        with open(fivegram_path, 'rb') as f:
            self.fivegram_model = pickle.load(f)
        with open(sixgram_path, 'rb') as f:
            self.sixgram_model = pickle.load(f)

    def generate_sentence(self, start_words=None, length=10):
        if start_words is None:
            start_words = random.choice(list(self.trigram_model.keys()))
        else:
            start_words = tuple(start_words)
            if len(start_words) < self.n - 1:
                raise ValueError(f"start_words must contain at least {self.n - 1} words.")

        current_words = list(start_words)
        sentence = current_words.copy()
        transition_analyzer = TransitionAnalyzer("")  # Dummy init

        for _ in tqdm(range(length), desc="Generating words", position=0, leave=False):
            prefix = tuple(current_words[-(self.n-1):])

            # Collect next words from n-gram models
            next_words = {}
            for model_attr in ['bigram_model', 'trigram_model', 'fourgram_model', 'fivegram_model', 'sixgram_model']:
                model = getattr(self, model_attr, None)
                if model and prefix in model:
                    next_words.update(model[prefix])

            if not next_words:
                break

            # Transition analysis and contextual word choice
            last_sentence = ' '.join(current_words)
            corrected_sentence = self.correct_grammar(last_sentence)
            transition_analyzer = TransitionAnalyzer(corrected_sentence)
            context_word = self.get_center_from_sentence(corrected_sentence, transition_analyzer)

            next_word = self.choose_word_with_context(next_words, context_word)

            if next_word != current_words[-1]:
                sentence.append(next_word)
                current_words.append(next_word)

            # Optional early stopping if sentence becomes complete and well-formed
            if len(sentence) >= length // 2:
                partial_sentence = ' '.join(sentence)
                if self.is_complete_thought(partial_sentence):
                    break

        sentence_text = ' '.join(sentence).strip()
        sentence_text = self.correct_grammar(sentence_text)
        return self.clean_text(sentence_text)



    def correct_grammar(self, sentence):
        """Correct common grammar mistakes in the given sentence."""
        """Load corrections from a JSON file and correct grammar in the given sentence."""
        # Load corrections from the JSON file
        with open(corrections_file, 'r', encoding='utf-8') as f:
            corrections = json.load(f)

        # Ensure the input is a string
        if not isinstance(sentence, str):
            raise ValueError("Input must be a string.")

        # Correct the grammar in the given sentence
        for wrong, right in corrections.items():
            sentence = sentence.replace(wrong, right)

        return sentence

    def is_complete_thought(self, sentence):
        """Check if the current sentence forms a complete thought with refined linguistic and structural criteria."""
        if not sentence:
            return False

        # Join if input is a list
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)

        # Correct grammar before processing
        sentence = self.correct_grammar(sentence)

        # Parse sentence with SpaCy
        doc = nlp(sentence)

        # Basic checks
        if len(doc) < 3 or doc[-1].text not in ['.', '!', '?', '...']:
            return False

        # Structural checks
        has_subject = any(tok.dep_ in ('nsubj', 'nsubjpass', 'expl') for tok in doc)
        has_verb = any(tok.pos_ in ('VERB', 'AUX') for tok in doc)
        subject_count = sum(1 for tok in doc if tok.dep_ in ('nsubj', 'nsubjpass', 'expl'))

        # Problematic conjunction usage
        if doc[0].pos_ == 'CCONJ' or doc[-2].pos_ == 'CCONJ':
            return False
        if any(tok.pos_ == 'CCONJ' and tok.i == len(doc) - 2 for tok in doc):
            return False
        if any(tok.text.lower() in ['and', 'but', 'or'] and tok.i == 0 for tok in doc):
            return False

        # Ellipses without verb
        if doc[-1].text == '...' and not has_verb:
            return False

        # Incomplete clause types
        has_dependent_clause = any(tok.dep_ in ('advcl', 'acl', 'relcl', 'csubj', 'csubjpass', 'xcomp') for tok in doc)
        if has_dependent_clause and not has_subject:
            return False

        # Conjunctive verbs without main clause
        only_conjunct_verbs = all(tok.dep_ in ('conj', 'xcomp') for tok in doc if tok.pos_ == 'VERB')
        if only_conjunct_verbs and not has_subject:
            return False

        # Negative without supportive clause
        has_negation = any(tok.dep_ == 'neg' for tok in doc)
        if has_negation and not has_dependent_clause:
            return False

        # Final check
        if not (has_subject and has_verb and subject_count >= 1):
            return False

        return True


    def get_center_from_sentence(self, sentence, transition_analyzer):
        """
        Extracts the most salient noun phrase (center) from the sentence based on centering theory 
        and TransitionAnalyzer results.
        """
        transition_analyzer = TransitionAnalyzer(sentence)
        
        # Run analysis
        transition_results = transition_analyzer.analyze()  # Correct method call   
        doc = nlp(sentence)
        
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]  # Get all noun phrases
        
        # Initialize structures to hold centers based on grammatical roles
        subjects = []
        objects = []
        pronouns = []
        candidates = defaultdict(int)

        # Classify noun phrases based on their roles in the sentence
        for token in doc:
            if token.dep_ in ('nsubj', 'nsubjpass'):  # Nominal subjects
                subjects.append(token.text)
                candidates[token.text] += 2
            elif token.dep_ in ('dobj', 'pobj', 'attr', 'oprd'):  # Objects and attributes
                objects.append(token.text)
                candidates[token.text] += 1
            elif token.pos_ == 'PRON':  # Pronouns
                pronouns.append(token.text)
                candidates[token.text] += 1

        # Prioritize subjects based on centering theory principles
        if subjects:
            subjects = sorted(subjects, key=lambda x: (-candidates[x], sentence.index(x)))
            return subjects[0]

        # Use transition data to identify continuity with next sentences
        for result in transition_results:
            if result['current_sentences'] == sentence:
                if result['transition'] == "Center Continuation (CON)":
                    # Prefer noun phrases that continue across sentences
                    continuation_nps = set(result['current_nps']).intersection(result['next_nps'])
                    if continuation_nps:
                        return next(iter(continuation_nps))

                elif result['transition'] in ("Smooth Shift (SSH)", "Rough Shift (RSH)"):
                    # For shifts, select salient noun phrase in the current sentence
                    # Sort objects by both frequency and position for more precise centering
                    objects = sorted(objects, key=lambda x: (-candidates[x], sentence.index(x)))
                    if objects:
                        return objects[0]
                    elif pronouns:
                        return pronouns[0]  # Prioritize pronouns as fallback option

        # If no transition match, use max scoring among noun phrases
        valid_noun_phrases = [np for np in noun_phrases if np in candidates]
        if valid_noun_phrases:
            return max(valid_noun_phrases, key=candidates.get, default=None)

        return None  # Return None if no valid noun phrases exist


    def choose_word_with_context(self, next_words, context_word=None, semantic_threshold=0.2, position_index=0, structure_template=None, prev_pos=None, pos_bigrams=None):
        if not next_words:
            return None

        word_choices = list(next_words.keys())
        probabilities = np.array(list(next_words.values()), dtype=float)
        probabilities = np.maximum(probabilities, 0)
        total = probabilities.sum()
        probabilities = probabilities / total if total > 0 else np.ones_like(probabilities) / len(probabilities)

        # POS filtreleme (structure_template varsa)
        if structure_template:
            target_pos = structure_template[position_index % len(structure_template)]
            valid_words, valid_vectors, valid_probs, valid_pos = [], [], [], []

            for word, prob in zip(word_choices, probabilities):
                doc = nlp(word)
                if doc and doc[0].pos_ == target_pos:
                    valid_words.append(word)
                    valid_vectors.append(doc[0].vector)
                    valid_probs.append(prob)
                    valid_pos.append(doc[0].pos_)

            if not valid_words:
                print(f"[WARN] No words matched POS '{target_pos}'. Fallback to all.")
                valid_words = word_choices
                valid_vectors = [nlp(word).vector for word in word_choices]
                valid_probs = probabilities
                valid_pos = [nlp(word)[0].pos_ for word in word_choices]

            else:
                word_choices = valid_words
                probabilities = np.array(valid_probs)
                word_vectors = np.array(valid_vectors)
        else:
            word_vectors = np.array([nlp(word).vector for word in word_choices])
            valid_pos = [nlp(word)[0].pos_ for word in word_choices]


        # Bağlam varsa similarity hesapla
        if context_word:
            context_vector = nlp(context_word).vector
            if context_vector is None or np.all(context_vector == 0):
                print("[ERROR] Context vector is empty. Fallback.")
                return np.random.choice(word_choices, p=probabilities)

            context_vector = context_vector.reshape(1, -1)
            similarity_scores = cosine_similarity(context_vector, word_vectors).flatten()
            similarity_scores = np.maximum(similarity_scores, 0)
            similarity_scores[similarity_scores < semantic_threshold] = 0
        else:
            similarity_scores = np.ones_like(probabilities)

        # POS bigram geçiş puanlarını uygula
        if pos_bigrams and prev_pos:
            transition_scores = np.array([
                pos_bigrams.get((prev_pos, curr_pos), 0.01) for curr_pos in valid_pos
            ])
        else:
            transition_scores = np.ones_like(probabilities)

        # Tümünü birleştir: final score = sim * prob * trans
        final_scores = similarity_scores * probabilities * transition_scores
        if final_scores.sum() == 0:
            print("[LOG] All scores 0, fallback to uniform.")
            final_scores = np.ones_like(probabilities) / len(probabilities)
        else:
            final_scores /= final_scores.sum()

        chosen_word = np.random.choice(word_choices, p=final_scores)
        print(f"[LOG] ➡️ Chosen Word: {str(chosen_word)} | POS: {nlp(str(chosen_word))[0].pos_}")
        return chosen_word




    def clean_text(self, text):
        # Check if the input is empty or None
        if not text:
            return ""
        # Normalize spaces: ensure only one space between words and trim excess spaces at the ends
        text = ' '.join(text.split())
        # Remove unwanted spaces before punctuation marks
        text = re.sub(r'\s([?.!,:;])', r'\1', text)
        # Fix double punctuation issues
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        # Handle misplaced leading and trailing quotes and whitespace
        text = text.strip("'\"")
        # Capitalize the first letter of the sentence
        text = text[0].capitalize() + text[1:] if text else ""
        # Add a period at the end of the sentence if not already there
        if text and text[-1] not in ['.', '!', '?']:
            text += '.'
        # Handle any punctuation-related formatting issues
        text = re.sub(r'(\w)([.,!?;])', r'\1 \2', text)  # Ensure space before punctuation
        # Handle cases where punctuation might be attached without space
        text = re.sub(r'([,.!?;])(\w)', r'\1 \2', text)
        # Ensure proper spacing around punctuation marks
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces in general
        text = re.sub(r'(\s)([.,!?;])', r'\2', text)  # Remove spaces before punctuation
        return text



    def post_process_sentences(self, sentences, entity_diversity_threshold=None, noun_phrase_diversity_threshold=None):
        """Post-processes sentences to ensure coherence and thematic consistency.

        Args:
            sentences (list of str): The sentences to process.
            entity_diversity_threshold (int): Minimum number of unique named entities required for coherence.
            noun_phrase_diversity_threshold (int): Minimum number of unique noun phrases required for coherence.

        Returns:
            tuple: A tuple containing the adjusted sentences and a detailed report on coherence.
        """
        # Default thresholds if not provided
        entity_diversity_threshold = entity_diversity_threshold or 5  # Default threshold
        noun_phrase_diversity_threshold = noun_phrase_diversity_threshold or 5  # Default threshold

        # Combine sentences for holistic analysis
        full_text = ' '.join(sentences)
        doc = nlp(full_text)  # Use SpaCy to analyze the full text

        # Extract unique named entities and noun phrases
        entities = {ent.text for ent in doc.ents}
        noun_phrases = {chunk.text for chunk in doc.noun_chunks}

        # Initialize report with detailed coherence metrics
        report = {
            "total_entity_count": len(entities),
            "total_noun_phrase_count": len(noun_phrases),
            "entity_diversity_score": len(entities) / max(1, len(sentences)),
            "noun_phrase_diversity_score": len(noun_phrases) / max(1, len(sentences)),
            "needs_rephrasing": len(entities) < entity_diversity_threshold or len(noun_phrases) < noun_phrase_diversity_threshold,
            "suggestions": [],
            "distribution_analysis": {},
            "word_frequency_analysis": {},
        }

        # Check diversity thresholds and provide feedback
        if len(entities) < entity_diversity_threshold:
            report["suggestions"].append(
                f"Low named entity diversity detected ({len(entities)} entities). "
                "Consider rephrasing to include a greater variety of entities."
            )
        if len(noun_phrases) < noun_phrase_diversity_threshold:
            report["suggestions"].append(
                f"Low noun phrase diversity detected ({len(noun_phrases)} noun phrases). "
                "Consider rephrasing to introduce new themes or ideas."
            )

        # Additional checks and suggestions can be added here...

        # Distribution analysis: Track entity and noun phrase occurrences per sentence
        for i, sent in enumerate(doc.sents):
            sent_entities = {ent.text for ent in sent.ents}
            sent_noun_phrases = {chunk.text for chunk in sent.noun_chunks}

            # Record the presence of entities and noun phrases in each sentence
            report["distribution_analysis"][f"sentence_{i + 1}"] = {
                "entities": list(sent_entities),
                "noun_phrases": list(sent_noun_phrases)
            }

            # Provide targeted rephrasing suggestions for repetitive content
            if i > 0:  # Compare with the previous sentence
                prev_sent_entities = {ent.text for ent in doc.sents[i - 1].ents}
                prev_sent_noun_phrases = {chunk.text for chunk in doc.sents[i - 1].noun_chunks}

                if sent_entities & prev_sent_entities:
                    report["suggestions"].append(
                        f"Sentence {i + 1} shares entities with Sentence {i}. "
                        "Consider rephrasing to improve thematic variety."
                    )
                if sent_noun_phrases & prev_sent_noun_phrases:
                    report["suggestions"].append(
                        f"Sentence {i + 1} shares noun phrases with Sentence {i}. "
                        "Consider rephrasing to avoid repetition."
                    )

        # Word frequency analysis to avoid overuse of common words or phrases
        word_freq = Counter(token.text.lower() for token in doc if not token.is_stop and not token.is_punct)
        frequent_words = {word: count for word, count in word_freq.items() if count > 1}

        # Add frequency analysis to report
        report["word_frequency_analysis"]["frequent_words"] = frequent_words
        if frequent_words:
            report["suggestions"].append(
                "Some words or phrases are repeated frequently, which may impact readability. "
                "Consider rephrasing to introduce variation."
            )

        # Adjust sentences by cleaning them individually
        adjusted_sentences = [self.clean_text(sentence) for sentence in sentences]

        return adjusted_sentences, report


    def advanced_length_adjustment(self, last_sentence, base_length):
        """ Adjust length based on the structure of the last sentence with improved clause and complexity handling. """
        last_words = last_sentence.split()
        last_length = len(last_words)

        # Improved clause count based on multiple conjunctions and punctuation
        clause_count = sum(last_sentence.count(conj) for conj in [',', 'and', 'but', 'or', 'yet']) + 1

        # Analyze the sentence using SpaCy for part-of-speech and dependency structure
        doc = nlp(last_sentence)
        noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
        verb_count = sum(1 for token in doc if token.pos_ == "VERB")
        adjective_count = sum(1 for token in doc if token.pos_ == "ADJ")
        adverb_count = sum(1 for token in doc if token.pos_ == "ADV")

        # Dynamic complexity factor using both POS counts and dependency relations
        complexity_factor = ((noun_count + verb_count + adjective_count + adverb_count) +
                            sum(1 for token in doc if token.dep_ in {"conj", "advcl", "relcl"})) // 2

        # Refined length variability based on last sentence complexity and length
        length_variability = ((last_length - base_length) + complexity_factor) // 3  # Adjust for finer control

        # Set adjusted length with enhanced variability and ensure it’s within limits
        adjusted_length = max(5, min(base_length + random.randint(-3, 3) + clause_count + complexity_factor + length_variability, 16))

        return adjusted_length

    def generate_and_post_process(self, num_sentences=10, input_words=None, length=20):
        generated_sentences = []
        max_attempts = 5  # Max attempts to generate a coherent sentence

        for i in tqdm(range(num_sentences), desc="Generating sentences", position=1, leave=True):
            attempts = 0
            coherent_sentence = False

            while attempts < max_attempts and not coherent_sentence:
                if i == 0:
                    generated_sentence = self.generate_sentence(start_words=input_words, length=length)
                else:
                    last_sentence = generated_sentences[-1]
                    adjusted_length = self.advanced_length_adjustment(last_sentence, length)
                    generated_sentence = self.generate_sentence(length=adjusted_length)

                if self.is_sentence_coherent(generated_sentence, previous_sentences=generated_sentences):
                    if not self.is_complete_thought(generated_sentence):
                        print(f"[SKIP] Not a complete thought: {generated_sentence}")
                        attempts += 1
                        continue

                    generated_sentences.append(generated_sentence)
                    coherent_sentence = True
                else:
                    attempts += 1
                    print(f"Attempt {attempts}: Generated incoherent sentence: {generated_sentence}")

            if not coherent_sentence:
                print(f"Max attempts reached for generating sentence {i + 1}. Adding the incoherent sentence.")
                generated_sentence = self.correct_grammar(generated_sentence)
                generated_sentences.append(generated_sentence)

        final_text = ' '.join(generated_sentences)
        final_text = self.correct_grammar(final_text)
        final_text = self.post_process_text(final_text)
        return final_text


    def get_proper_nouns(self, text):
        """Identify proper nouns in the text using SpaCy."""
        doc = nlp(text)
        proper_nouns = []

        for token in doc:
            if token.pos_ == 'PROPN':  # Check if the token is a proper noun
                proper_nouns.append(token.text)

        return list(set(proper_nouns))  # Return unique proper nouns

    def post_process_text(self, text):
        """Post-process the text to ensure proper punctuation and grammar rules."""
        # Split sentences by multiple delimiters (. ! ?)
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        cleaned_sentences = []

        for sentence in sentences:
            cleaned_sentence = sentence.strip()
            if cleaned_sentence:
                # Capitalize the first character of the sentence
                cleaned_sentence = cleaned_sentence[0].upper() + cleaned_sentence[1:]

                # Clean up spaces around punctuation
                cleaned_sentence = re.sub(r'\s+([,.!?])', r'\1', cleaned_sentence)  # Remove space before punctuation
                cleaned_sentence = re.sub(r'([,.!?])\s+', r'\1 ', cleaned_sentence)  # Ensure space after punctuation
                cleaned_sentence = re.sub(r'\s{2,}', ' ', cleaned_sentence)  # Remove double spaces

                # Remove trailing commas
                cleaned_sentence = cleaned_sentence.rstrip(",")

                # Handle redundant punctuation (e.g., "Hello!!" -> "Hello!")
                cleaned_sentence = re.sub(r'([.!?])\1+', r'\1', cleaned_sentence)

                # Handle unnecessary conjunctions at the beginning
                cleaned_sentence = re.sub(r'^(And|But|Or)\b,?\s+', '', cleaned_sentence, flags=re.IGNORECASE)

                # Handle quotations (optional - remove or stylize)
                cleaned_sentence = re.sub(r'"(.*?)"', r'“\1”', cleaned_sentence)  # Replace " with proper quotation marks

                # Optional: Capitalize proper nouns dynamically using SpaCy NER
                doc = nlp(cleaned_sentence)
                for entity in doc.ents:
                    if entity.label_ in ["PERSON", "ORG", "GPE"]:  # Capitalize certain entities
                        cleaned_sentence = re.sub(r'\b' + re.escape(entity.text.lower()) + r'\b', entity.text, cleaned_sentence)

                cleaned_sentences.append(cleaned_sentence)

        # Join the cleaned sentences and ensure final text ends with a period
        final_output = ' '.join(cleaned_sentences).strip()
        if final_output and not final_output.endswith('.'):
            final_output += '.'
        
        final_output = self.correct_grammar(final_output)  # Final grammar check
        return final_output


    def is_sentence_coherent(self, sentence, previous_sentences=None):
        """Evaluate the coherence of a generated sentence using semantic similarity and syntactic features."""
        if not sentence or len(sentence.split()) < 4 or sentence[-1] not in ['.', '!', '?']:
            return False

        current_doc = nlp(sentence)
        current_embedding = current_doc.vector

        if previous_sentences is None or len(previous_sentences) == 0:
            return True

        previous_embeddings = [nlp(prev).vector for prev in previous_sentences]

        if not previous_embeddings:
            return True

        similarities = [
            np.dot(current_embedding, emb) / 
            (np.linalg.norm(current_embedding) * np.linalg.norm(emb) + 1e-8)
            for emb in previous_embeddings
        ]
        avg_similarity = np.mean(similarities)

        # === Dinamik eşik belirleme ===
        avg_len = np.mean([len(prev.split()) for prev in previous_sentences])
        var_len = np.var([len(prev.split()) for prev in previous_sentences])

        noun_count = sum(1 for token in current_doc if token.pos_ == "NOUN")
        verb_count = sum(1 for token in current_doc if token.pos_ == "VERB")
        adj_count = sum(1 for token in current_doc if token.pos_ == "ADJ")
        adv_count = sum(1 for token in current_doc if token.pos_ == "ADV")
        clause_count = sum(1 for token in current_doc if token.dep_ in ("advcl", "relcl", "ccomp", "xcomp"))

        complexity_factor = (noun_count + verb_count + adj_count + adv_count + clause_count) / 4.0

        threshold = 0.6
        if avg_len > 15:
            threshold += 0.1
        elif avg_len < 8:
            threshold -= 0.1
        if var_len > 5:
            threshold += 0.05
        if complexity_factor > 2.5:
            threshold += 0.05
        elif complexity_factor < 1.5:
            threshold -= 0.05

        # Ek olarak anlamsal çeşitlilik için ceza uygula
        unique_lemmas = set(token.lemma_ for token in current_doc if token.pos_ in {"NOUN", "VERB"})
        if len(unique_lemmas) < 3:
            threshold += 0.05

        return avg_similarity > threshold



    def save_model(self, filename, compress=False):
        """
        Save the language model and total counts to a file.

        Args:
            filename (str): The name of the file to save the model.
            compress (bool): If True, saves the model using compression. Default is False.
        
        Raises:
            IOError: If an I/O error occurs during file operations.
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Convert defaultdicts to regular dicts for pickling
            model_data = (self.model, self.total_counts)

            if compress:
                import gzip
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(model_data, f)  # Save as compressed file
            else:
                with open(filename, 'wb') as f:
                    pickle.dump(model_data, f)  # Save as regular file
                    
            print(f"Model saved successfully to {filename}")

        except IOError as e:
            print(f"Error saving model to {filename}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            model, total_counts = pickle.load(f)
        instance = cls("dummy text")  # We need to create an instance
        instance.model = defaultdict(lambda: defaultdict(int), model)  # Load as defaultdict
        instance.total_counts = total_counts
        return instance

def load_text_from_file(file_path):
    """
    Load text from a file, preprocess it by removing punctuation and symbols,
    and return a SpaCy Doc object.
    
    Args:
        file_path (str): The path to the text file.
        
    Returns:
        spacy.tokens.Doc: A SpaCy Doc object containing the preprocessed text.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

            # Clean the text: remove extra whitespace and newline characters
            text = ' '.join(text.split())
            text = text.strip()

            # Process the text with SpaCy
            doc = nlp(text)  # Create a SpaCy Doc object

            # Remove punctuation and symbols
            cleaned_tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
            cleaned_text = ' '.join(cleaned_tokens)

            # Return a new SpaCy Doc object from the cleaned text
            return nlp(cleaned_text)
            
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    
    return None  # Return None if there's an error


# Metni dosyadan yükle
file_path = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\text_data.txt'
text = load_text_from_file(file_path)

# Model dosyasının varlığını kontrol et
model_file = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\language_model.pkl'

try:
    # Mevcut modeli yüklemeye çalış
    language_model = EnhancedLanguageModel.load_model(model_file)
    print("Loaded existing model.")
except (FileNotFoundError, EOFError):
    # Model yoksa yeni bir tane oluştur
    language_model = EnhancedLanguageModel(text, n=2)
    language_model.save_model(model_file)  # Yeni oluşturulan modeli kaydet
    print("Created and saved new model.")

# Belirtilen sayıda cümle üret
num_sentences = 5  # Üretilecek cümle sayısı
input_words = "Least of all do they thus dispose of the murdered.".split()

# Entegre edilmiş yöntemle başlangıç metni üret
generated_text = language_model.generate_and_post_process(num_sentences=num_sentences, input_words=input_words, length=5)
print("Generated Text:\n", generated_text)






