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
        transition_analyzer = TransitionAnalyzer("")  # Initialize with dummy sentence

        for _ in tqdm(range(length), desc="Generating words"):
            prefix = tuple(current_words[-(self.n-1):])  # Last n-1 words

            # Get next words from the appropriate model, dynamically adjusting for n-grams
            next_words = {}
            if self.n >= 2 and prefix in self.bigram_model:
                next_words.update(self.bigram_model[prefix])
            if self.n >= 3 and prefix in self.trigram_model:
                next_words.update(self.trigram_model[prefix])
            if hasattr(self, 'fourgram_model') and self.n >= 4 and prefix in self.fourgram_model:
                next_words.update(self.fourgram_model[prefix])
            if hasattr(self, 'fivegram_model') and self.n >= 5 and prefix in self.fivegram_model:
                next_words.update(self.fivegram_model[prefix])
            if hasattr(self, 'sixgram_model') and self.n >= 6 and prefix in self.sixgram_model:
                next_words.update(self.sixgram_model[prefix])

            # Stop if no viable next words
            if not next_words:
                break

            # Process current sentence for transition analysis and grammar correction
            last_sentence = ' '.join(current_words)
            corrected_sentence = self.correct_grammar(last_sentence)
            transition_analyzer = TransitionAnalyzer(corrected_sentence)
            context_word = self.get_center_from_sentence(corrected_sentence, transition_analyzer)
            
            # Choose the next word based on context and avoid repeating the last word
            next_word = self.choose_word_with_context(next_words, context_word)

            # Ensure next word is not the same as the last word and is contextually appropriate
            if next_word != current_words[-1]:
                sentence.append(next_word)
                current_words.append(next_word)

            # Check for complete thought after generating a reasonable length
            if len(sentence) > length // 2 and self.is_complete_thought(sentence):
                break

        sentence_text = ' '.join(sentence).strip()
        sentence_text = self.correct_grammar(sentence_text)  # Final grammar check
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
        """Check if the current sentence forms a complete thought with refined criteria."""
        if not sentence:
            return False

        # Ensure the input is a string
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)  # Join list elements into a single string

        # Correct grammar mistakes before analyzing the structure
        sentence = self.correct_grammar(sentence)

        # Process the sentence with SpaCy to analyze its structure
        doc = nlp(sentence)

        # Check if the last word is punctuation indicating a complete thought
        if doc[-1].text not in ['.', '!', '?', '...']:
            return False

        # Check if there are at least three words in the sentence
        if len(doc) < 3:
            return False

        # Ensure the sentence contains at least one subject and one verb
        has_subject = any(token.dep_ in ('nsubj', 'nsubjpass', 'expl') for token in doc)  # Includes implicit subjects
        has_verb = any(token.pos_ in ('VERB', 'AUX') for token in doc)  # Includes modal auxiliaries

        # Check if the sentence starts or ends with conjunctions, or has conjunctions before punctuation
        if doc[0].pos_ == 'CCONJ' or doc[-2].pos_ == 'CCONJ' or any(token.pos_ == 'CCONJ' and token.i == len(doc) - 2 for token in doc):
            return False

        # Check for the presence of dependent clauses indicating incomplete thoughts
        has_dependent_clause = any(token.dep_ in ('advcl', 'acl', 'relcl', 'csubj', 'csubjpass', 'xcomp') for token in doc)

        # Adjust conditions to allow for some flexibility in structure
        subject_count = sum(1 for token in doc if token.dep_ in ('nsubj', 'nsubjpass', 'expl'))
        verb_conflict = any(token.dep_ == 'conj' and token.head.pos_ == 'VERB' for token in doc)

        # Handle incomplete thoughts caused by ellipses or unfinished clauses
        if doc[-1].text == '...' and not has_verb:
            return False

        # Return True only if the structure is coherent
        return has_subject and has_verb and not has_dependent_clause and subject_count >= 1 and not verb_conflict

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


    def choose_word_with_context(self, next_words, context_word=None, semantic_threshold=0.99):
        if not next_words:
            return None  # No next words available

        word_choices = list(next_words.keys())
        probabilities = np.array(list(next_words.values()), dtype=float)  # Ensure float type

        # Ensure probabilities are non-negative
        probabilities = np.maximum(probabilities, 0)

        # Normalize probabilities
        total = probabilities.sum()
        if total > 0:
            probabilities /= total  # Normalize to sum to 1
        else:
            probabilities = np.ones_like(probabilities) / len(probabilities)  # Equal distribution if all probabilities are zero

        if context_word:
            # Check if the context word has a valid vector representation
            context_vector = nlp(context_word).vector
            if context_vector is None or np.all(context_vector == 0):
                return np.random.choice(word_choices, p=probabilities)  # Fallback to probabilities if context vector is invalid
            
            context_vector = context_vector.reshape(1, -1)  # Reshape for cosine_similarity
            
            # Compute similarity scores with the context word using cosine similarity
            word_vectors = np.array([nlp(word).vector for word in word_choices])
            
            # Use cosine similarity to compute the relationship between the context and candidate words
            similarity_scores = cosine_similarity(context_vector, word_vectors).flatten()

            # Apply semantic threshold: zero out scores below the threshold
            similarity_scores = np.maximum(similarity_scores, 0)  # Ensure no negative values
            similarity_scores[similarity_scores < semantic_threshold] = 0  # Apply threshold

            # If all similarity scores are below the threshold, fallback to probabilities
            if np.sum(similarity_scores) == 0:
                return np.random.choice(word_choices, p=probabilities)

            # Scale similarity scores using softmax to ensure better distribution
            exp_similarities = np.exp(similarity_scores)  # Exponentiate the similarity scores
            adjusted_probabilities = exp_similarities * probabilities  # Combine with original probabilities
            
            # Normalize adjusted probabilities
            adjusted_probabilities = np.exp(adjusted_probabilities) / np.sum(np.exp(adjusted_probabilities))

            # Increase the influence of similarity scores dynamically based on their range
            if len(similarity_scores) > 0:
                mean_similarity = np.mean(similarity_scores)
                std_similarity = np.std(similarity_scores)
                influence_factor = 1 + (similarity_scores - mean_similarity) / (std_similarity + 1e-5)  # Avoid division by zero
                
                # Apply influence factor to adjusted probabilities
                adjusted_probabilities *= influence_factor
                adjusted_probabilities /= adjusted_probabilities.sum()  # Normalize again

            # Make selection based on the final probabilities
            chosen_word = np.random.choice(word_choices, p=adjusted_probabilities)
        else:
            chosen_word = np.random.choice(word_choices, p=probabilities)

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

        for i in tqdm(range(num_sentences), desc="Generating sentences"):
            attempts = 0
            coherent_sentence = False

            while attempts < max_attempts and not coherent_sentence:
                # Allow variability in sentence length based on previous sentences
                if i == 0:
                    generated_sentence = self.generate_sentence(start_words=input_words, length=length)
                else:
                    last_sentence = generated_sentences[-1]
                    # Generate a new sentence with adjusted length
                    adjusted_length = self.advanced_length_adjustment(last_sentence, length)

                    # Generate a new sentence
                    generated_sentence = self.generate_sentence(length=adjusted_length)

                # Check the quality of the generated sentence
                if self.is_sentence_coherent(generated_sentence, previous_sentences=generated_sentences):
                    generated_sentences.append(generated_sentence)
                    coherent_sentence = True  # Mark as coherent
                else:
                    attempts += 1
                    print(f"Attempt {attempts}: Generated incoherent sentence: {generated_sentence}")

            if not coherent_sentence:
                print(f"Max attempts reached for generating sentence {i + 1}. Adding the incoherent sentence.")
                # Grammar correction before appending
                generated_sentence = self.correct_grammar(generated_sentence)
                generated_sentences.append(generated_sentence)  # Add the incoherent sentence instead of a placeholder

        final_text = ' '.join(generated_sentences)
        
        # Perform grammar correction for the entire generated text
        final_text = self.correct_grammar(final_text)  # Final grammar check
        
        # Post-process the text (ensure punctuation and proper spacing)
        final_text = self.post_process_text(final_text)  # Call the new post-processing method

        # Return the final processed text
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
        # Check for basic sentence validity
        if not sentence or len(sentence.split()) < 4 or sentence[-1] not in ['.', '!', '?']:
            return False

        # Process the current sentence with SpaCy
        current_doc = nlp(sentence)
        current_embedding = current_doc.vector

        # If there are no previous sentences, we consider it coherent
        if previous_sentences is None or len(previous_sentences) == 0:
            return True

        # Calculate semantic similarity with previous sentences
        previous_embeddings = [nlp(prev_sentence).vector for prev_sentence in previous_sentences]
        
        # Handle case where previous embeddings could be empty (safety)
        if not previous_embeddings:
            return True

        # Calculate cosine similarity
        similarities = [
            np.dot(current_embedding, prev_embedding) / 
            (np.linalg.norm(current_embedding) * np.linalg.norm(prev_embedding) + 1e-8)  # Avoid division by zero
            for prev_embedding in previous_embeddings
        ]

        # Calculate average similarity
        avg_similarity = np.mean(similarities)

        # Determine a dynamic threshold based on previous sentences
        threshold = 0.6  # Default threshold
        avg_length = np.mean([len(prev_sentence.split()) for prev_sentence in previous_sentences])
        length_variance = np.var([len(prev_sentence.split()) for prev_sentence in previous_sentences])

        # Adjust threshold based on sentence complexity
        noun_count = sum(1 for token in current_doc if token.pos_ == "NOUN")
        verb_count = sum(1 for token in current_doc if token.pos_ == "VERB")
        adj_count = sum(1 for token in current_doc if token.pos_ == "ADJ")
        adv_count = sum(1 for token in current_doc if token.pos_ == "ADV")

        # Calculate complexity factor
        complexity_factor = (noun_count + verb_count + adj_count + adv_count) / 3.0

        # Adjust threshold based on average sentence length and complexity
        if avg_length > 15:
            threshold += 0.1
        elif avg_length < 8:
            threshold -= 0.1

        # Adjust for length variance in previous sentences
        if length_variance > 5:
            threshold += 0.05

        # Adjust threshold based on complexity factor
        if complexity_factor > 2:
            threshold += 0.05

        # Final coherence decision based on similarity and threshold
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
num_sentences = 7  # Üretilecek cümle sayısı
input_words = "This was it. The final push.".split()

# Entegre edilmiş yöntemle başlangıç metni üret
generated_text = language_model.generate_and_post_process(num_sentences=num_sentences, input_words=input_words, length=10)
print("Generated Text:\n", generated_text)






