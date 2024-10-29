import random
import pickle
from collections import defaultdict
import spacy
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
import os

# Load the SpaCy English model with word vectors
nlp = spacy.load("en_core_web_md")  # Use medium model for better embeddings

class EnhancedLanguageModel:
    def __init__(self, text, n=2):
        self.n = n  # N-gram size
        self.model, self.total_counts = self.build_model(text)  # Combine model and total counts
        
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

    def generate_sentence(self, start_words=None, length=10):
        if start_words is None:
            start_words = random.choice(list(self.model.keys()))
        else:
            start_words = tuple(start_words)

        current_words = list(start_words)
        sentence = current_words.copy()

        for _ in tqdm(range(length), desc="Generating words"):
            prefix = tuple(current_words[-(self.n-1):])  # Get the last n-1 words
            next_words = self.model.get(prefix, {})

            if not next_words:
                break  # Stop if no next words

            # Determine the center based on the last generated sentence
            last_sentence = ' '.join(current_words)
            context_word = self.get_center_from_sentence(last_sentence)

            # Choose the next word considering the context
            next_word = self.choose_word_with_context(next_words, context_word)

            # Avoid repeating the last word in the sentence
            if next_word != current_words[-1]:
                sentence.append(next_word)
                current_words.append(next_word)

            # Optionally, implement a stopping condition for coherence
            if self.is_complete_thought(sentence):
                break

        sentence_text = ' '.join(sentence).strip()
        return self.clean_text(sentence_text)

    def is_complete_thought(self, sentence):
        """Check if the current sentence forms a complete thought with stricter criteria."""
        if not sentence:
            return False

        # Ensure the input is a string, join if it's a list
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)  # Join list elements into a single string

        # Process the sentence with SpaCy to analyze its structure
        doc = nlp(sentence)

        # Check if the last word is a punctuation mark indicating a complete thought
        if doc[-1].text not in ['.', '!', '?']:
            return False

        # Check if there are at least three words in the sentence (including punctuation)
        if len(doc) < 3:  # A complete thought typically has at least a subject, verb, and possibly an object
            return False

        # Ensure the sentence contains at least one subject and one verb
        has_subject = any(token.dep_ in ('nsubj', 'nsubjpass') for token in doc)
        has_verb = any(token.pos_ == 'VERB' for token in doc)
        
        # Ensure the sentence does not begin or end with conjunctions
        if doc[0].pos_ == 'CCONJ' or doc[-2].pos_ == 'CCONJ':  # Check second-last token for conjunction
            return False

        # Ensure there is no dependent clause that would make the sentence incomplete
        has_dependent_clause = any(token.dep_ in ('advcl', 'acl', 'relcl') for token in doc)

        # Return True only if both subject and verb are present, and it does not contain dependent clauses
        return has_subject and has_verb and not has_dependent_clause




    def get_center_from_sentence(self, sentence):
        """Extract the most salient noun phrase (center) from the sentence based on centering theory."""
        doc = nlp(sentence)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]  # Get all noun phrases

        # Initialize variables to hold the centers based on their grammatical roles
        subjects = []
        objects = []
        pronouns = []
        candidates = defaultdict(int)  # To track frequency of noun phrases

        # Iterate through tokens to classify noun phrases based on their grammatical roles
        for token in doc:
            if token.dep_ in ('nsubj', 'nsubjpass'):  # Nominal subjects
                subjects.append(token.text)
                candidates[token.text] += 2  # Count occurrences with higher weight for subjects
            elif token.dep_ in ('dobj', 'pobj'):  # Direct objects
                objects.append(token.text)
                candidates[token.text] += 1  # Count occurrences
            elif token.pos_ == 'PRON':  # Include pronouns with medium weight
                pronouns.append(token.text)
                candidates[token.text] += 1

        # Determine the most salient noun phrase based on centering theory
        if subjects:
            # Prioritize subjects and sort by frequency, then by first appearance in the sentence
            subjects = sorted(subjects, key=lambda x: (-candidates[x], sentence.index(x)))
            return subjects[0]  # Return the most salient subject as the center

        if objects:
            # If no subjects, consider objects
            objects = sorted(objects, key=lambda x: (-candidates[x], sentence.index(x)))
            return objects[0]  # Return the most salient object as the center

        if pronouns:
            # As a last resort, consider pronouns if no subject or object is found
            pronouns = sorted(pronouns, key=lambda x: (-candidates[x], sentence.index(x)))
            return pronouns[0]  # Return the most salient pronoun as the center

        return None  # Return None if no noun phrases are found


    def choose_word_with_context(self, next_words, context_word=None, semantic_threshold=0.7):
        if not next_words:
            return None  # No next words available

        word_choices = list(next_words.keys())
        probabilities = np.array(list(next_words.values()))

        # Ensure probabilities are non-negative
        probabilities = np.maximum(probabilities, 0)

        # Normalize probabilities
        total = probabilities.sum()
        if total > 0:
            probabilities /= total  # Normalize to sum to 1
        else:
            probabilities = np.ones_like(probabilities) / len(probabilities)  # Equal distribution if all probabilities are zero

        if context_word:
            # Get context vector
            context_vector = nlp(context_word).vector
            
            # Compute similarity scores with the context word
            similarity_scores = np.array([np.dot(context_vector, nlp(word).vector) for word in word_choices])

            # Ensure similarity scores are non-negative and apply semantic threshold
            similarity_scores = np.maximum(similarity_scores, 0)
            similarity_scores = np.where(similarity_scores >= semantic_threshold, similarity_scores, 0)

            # Scale similarity scores to adjust probabilities
            adjusted_probabilities = probabilities * (similarity_scores + 1)  # Add 1 to avoid negative similarities
            adjusted_probabilities /= adjusted_probabilities.sum()  # Normalize to sum to 1

            # Increase the influence of similarity scores dynamically
            influence_factor = 1.6 if similarity_scores.max() < 0.7 else 1.0
            adjusted_probabilities = adjusted_probabilities ** influence_factor

            # Normalize again
            adjusted_probabilities /= adjusted_probabilities.sum()  # Normalize again

            # Make selection
            chosen_word = np.random.choice(word_choices, p=adjusted_probabilities)
        else:
            chosen_word = np.random.choice(word_choices, p=probabilities)

        return chosen_word


    def clean_text(self, text):
        """Cleans the input text by removing unwanted spaces, fixing punctuation issues,
        normalizing spaces, and ensuring proper capitalization.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The cleaned text.
        """
        
        # Remove unwanted spaces before punctuation
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ;", ";").replace(" :", ":").replace(" ?", "?").replace(" !", "!")

        # Fix double punctuation issues
        text = text.replace("..", ".").replace(",,", ",").replace("!!", "!").replace("??", "?")

        # Remove misplaced leading and trailing quotes
        text = text.strip("'").strip('"')

        # Normalize spaces: ensure only one space between words
        text = ' '.join(text.split())
        
        # Capitalize the first letter of the sentence
        if text:
            text = text[0].capitalize() + text[1:]

        # Add a period at the end of the sentence if not already there
        if text and text[-1] not in ['.', '!', '?']:
            text += '.'

        # Handle potential cases of trailing punctuation without space
        text = text.replace('."', '"').replace("'", "'").replace(" ,", ",").replace(" ;", ";").replace(" :", ":")
        
        return text


    def post_process_sentences(self, sentences, entity_diversity_threshold=2):
        """Post-processes sentences to ensure coherence and thematic consistency.

        Args:
            sentences (list of str): The sentences to process.
            entity_diversity_threshold (int): Minimum number of unique entities required for coherence.

        Returns:
            tuple: A tuple containing the adjusted sentences and a report on coherence.
        """
        
        # Combine sentences into a single text for analysis
        full_text = ' '.join(sentences)

        # Use SpaCy to process the text
        doc = nlp(full_text)

        # Extract named entities and noun phrases
        entities = set(ent.text for ent in doc.ents)  # Extract named entities
        noun_phrases = set(chunk.text for chunk in doc.noun_chunks)  # Extract noun phrases

        # Prepare a report on coherence
        report = {
            "entity_count": len(entities),
            "noun_phrase_count": len(noun_phrases),
            "needs_rephrasing": len(entities) < entity_diversity_threshold,
        }

        # Check for low entity diversity and provide feedback
        if report["needs_rephrasing"]:
            report["suggestion"] = "Low entity diversity detected. Rephrasing might be needed."

        # Adjust sentences by cleaning
        adjusted_sentences = [self.clean_text(sentence) for sentence in sentences]

        return adjusted_sentences, report


    def generate_and_post_process(self, num_sentences=10, input_words=None, length=20):
        generated_sentences = []
        max_attempts = 9  # Max attempts to generate a coherent sentence

        for i in tqdm(range(num_sentences), desc="Generating sentences"):
            attempts = 0
            while attempts < max_attempts:
                # Allow variability in sentence length based on previous sentences
                if i == 0:
                    generated_sentence = self.generate_sentence(start_words=input_words, length=length)
                else:
                    # Adjust length based on previous sentence length
                    last_length = len(generated_sentences[-1].split())
                    adjusted_length = max(5, min(length + random.randint(-2, 2), 30))  # Example: keep between 5 and 30
                    generated_sentence = self.generate_sentence(length=adjusted_length)

                # Check the quality of the generated sentence
                if self.is_sentence_coherent(generated_sentence, previous_sentences=generated_sentences):
                    generated_sentences.append(generated_sentence)
                    break  # Break if coherent
                else:
                    attempts += 1
                    print(f"Attempt {attempts}: Generated incoherent sentence: {generated_sentence}")

            if attempts == max_attempts:
                print(f"Max attempts reached for generating sentence {i + 1}. Adding a placeholder.")
                generated_sentences.append("This sentence could not be generated coherently.")  # Placeholder

        # Post-process generated sentences for coherence and flow
        processed_sentences = self.post_process_sentences(generated_sentences)
        final_text = ' '.join(processed_sentences)
        return final_text


    def is_sentence_coherent(self, sentence, previous_sentences=None):
        """Evaluate the coherence of a generated sentence using semantic similarity."""
        if not sentence or len(sentence.split()) < 4 or sentence[-1] not in ['.', '!', '?']:
            return False

        # Process the current sentence with SpaCy
        current_doc = nlp(sentence)
        current_embedding = current_doc.vector

        # Calculate semantic similarity with previous sentences
        if previous_sentences:
            previous_embeddings = [nlp(prev_sentence).vector for prev_sentence in previous_sentences]
            similarities = [
                current_embedding.dot(prev_embedding) / (np.linalg.norm(current_embedding) * np.linalg.norm(prev_embedding)) 
                for prev_embedding in previous_embeddings
            ]

            # Calculate average similarity
            avg_similarity = np.mean(similarities)

            # Dynamic threshold based on average sentence length
            threshold = 0.3  # Default threshold
            avg_length = np.mean([len(prev_sentence.split()) for prev_sentence in previous_sentences])
            
            # Adjust threshold based on previous sentence length
            if avg_length > 15:  # Example: if previous sentences are longer than 15 words
                threshold = 0.4

            # Check if the average similarity is above the threshold
            return avg_similarity > threshold

        return True  # If no previous sentences to compare, consider coherent

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

# Load the text from the file
file_path = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_data.txt'
text = load_text_from_file(file_path)

# Check if the model file exists
model_file = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\language_model.pkl'

try:
    # Attempt to load the existing model
    language_model = EnhancedLanguageModel.load_model(model_file)
    print("Loaded existing model.")
except (FileNotFoundError, EOFError):
    # If the model does not exist, create a new one
    language_model = EnhancedLanguageModel(text, n=3)
    language_model.save_model(model_file)  # Save the newly created model
    print("Created and saved new model.")

# Generate the specified number of sentences
num_sentences = 10 # Number of sentences to generate
input_words = ["he", "was"]  # Words to be used

# Generate initial text using your integrated method
generated_text = language_model.generate_and_post_process(num_sentences=num_sentences, input_words=input_words, length=23)
print("Generated Text:\n", generated_text)



