import random
import pickle
from collections import defaultdict, Counter
import spacy
import numpy as np
from tqdm import tqdm
import os
import re
import json
import datetime
from transition_analyzer import TransitionAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

# Load the SpaCy English model with word vectors
nlp = spacy.load("en_core_web_md")
nlp.max_length = 300000000 # or even higher

# N-gram modelleri yolları
text_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\text_data.txt"
bigram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\bigram_model.pkl"
trigram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\trigram_model.pkl"
fourgram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\fourgram_model.pkl"
fivegram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\fivegram_model.pkl"
sixgram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\sixgram_model.pkl"
corrections_file = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\corrections.json"

class EnhancedLanguageModel:
    def __init__(self, text, n=2):
        self.n = n
        self.model, self.total_counts = self.build_model(text)
        self.load_ngram_models()

    def get_log_file(self):
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        return os.path.join(logs_dir, f"daily_log_{today}.txt")

    def log(self, message):
        log_file = self.get_log_file()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def build_model(self, text):
        model = defaultdict(lambda: defaultdict(int))
        doc = nlp(text.lower())
        tokens = [token.text for token in doc if token.is_alpha]
        n_grams = [tuple(tokens[i:i+self.n]) for i in range(len(tokens)-self.n+1)]

        for n_gram in n_grams:
            prefix = n_gram[:-1]
            next_word = n_gram[-1]
            model[prefix][next_word] += 1

        total_counts = defaultdict(int)
        for prefix, next_words in model.items():
            total_count = sum(next_words.values())
            total_counts[prefix] = total_count
            for word in next_words:
                next_words[word] = (next_words[word] + 1) / (total_count + len(next_words))
            for word in next_words:
                continuation_count = sum(1 for ngram in n_grams if ngram[-1] == word)
                next_words[word] += (continuation_count / len(tokens))

        return dict(model), dict(total_counts)

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

        for _ in tqdm(range(length), desc="Generating words", position=0, leave=False, dynamic_ncols=True, mininterval=0.05, maxinterval=0.3):
            prefix = tuple(current_words[-(self.n-1):])
            next_words = {}
            found = False

            # Önce en büyük n-gramlardan başlayarak prefix arıyoruz
            for model_attr in ['sixgram_model', 'fivegram_model', 'fourgram_model', 'trigram_model', 'bigram_model']:
                model = getattr(self, model_attr, None)
                if model and prefix in model:
                    next_words = model[prefix]
                    found = True
                    break  # bulunca çıkıyoruz

            if not found or not next_words:
                break  # hiçbir modelde bulunamadıysa döngü bitiyor

            # Devam ediyoruz: artık next_words var
            last_sentence = ' '.join(current_words)
            corrected_sentence = self.correct_grammar(last_sentence)
            transition_analyzer = TransitionAnalyzer(corrected_sentence)
            context_word = self.get_center_from_sentence(corrected_sentence, transition_analyzer)
            
            # Burada bağlama göre kelime seçimi yapıyoruz
            next_word = self.choose_word_with_context(next_words, context_word)

            if next_word != current_words[-1]:
                sentence.append(next_word)
                current_words.append(next_word)

            if len(sentence) >= length // 2:
                partial_sentence = ' '.join(sentence)
                if self.is_complete_thought(partial_sentence):
                    break

        sentence_text = ' '.join(sentence).strip()
        sentence_text = self.correct_grammar(sentence_text)
        return self.clean_text(sentence_text)


    def correct_grammar(self, sentence):
        with open(corrections_file, 'r', encoding='utf-8') as f:
            corrections = json.load(f)
        if not isinstance(sentence, str):
            raise ValueError("Input must be a string.")
        for wrong, right in corrections.items():
            sentence = sentence.replace(wrong, right)
        return sentence

    def is_complete_thought(self, sentence):
        if not sentence:
            return False

        if isinstance(sentence, list):
            sentence = ' '.join(sentence)

        sentence = self.correct_grammar(sentence)
        doc = nlp(sentence)

        if len(doc) < 3:
            return False

        if doc[-1].text not in ['.', '!', '?']:
            return False

        has_subject = any(tok.dep_ in ('nsubj', 'nsubjpass', 'expl') for tok in doc)
        has_verb = any(tok.pos_ in ('VERB', 'AUX') for tok in doc)

        if not (has_subject and has_verb):
            return False

        subject_count = sum(1 for tok in doc if tok.dep_ in ('nsubj', 'nsubjpass', 'expl'))
        verb_count = sum(1 for tok in doc if tok.pos_ in ('VERB', 'AUX'))

        # ✨ Subj-Verb dengesi kontrolü
        if subject_count > 2 and verb_count == 0:
            return False

        # ✨ Bağlaçlarla başlama kontrolü
        if doc[0].pos_ == 'CCONJ' or (len(doc) > 1 and doc[1].pos_ == 'CCONJ'):
            return False

        # ✨ Çok düşük içerik (sadece pronoun veya adverb) kontrolü
        content_tokens = [tok for tok in doc if tok.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV')]
        if len(content_tokens) < 2:
            return False

        # ✨ Bağımlı cümleler kontrolü
        dependent_clauses = any(tok.dep_ in ('advcl', 'acl', 'relcl', 'csubj', 'csubjpass', 'xcomp') for tok in doc)
        if dependent_clauses and not has_subject:
            return False

        # ✨ Çoklu bağlaç yapısı ve kopukluk kontrolü
        conj_tokens = [tok for tok in doc if tok.dep_ == 'conj']
        if len(conj_tokens) > 2 and not has_subject:
            return False

        # ✨ Negation handling
        has_negation = any(tok.dep_ == 'neg' for tok in doc)
        if has_negation and not has_verb:
            return False

        # ✨ Quote veya Parenthesis düzgün kapanmış mı
        if sentence.count('"') % 2 != 0 or sentence.count('(') != sentence.count(')'):
            return False

        return True


    def get_center_from_sentence(self, sentence, transition_analyzer):
        transition_analyzer = TransitionAnalyzer(sentence)
        transition_results = transition_analyzer.analyze()
        doc = nlp(sentence)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        subjects, objects, pronouns = [], [], []
        candidates = defaultdict(int)

        for token in doc:
            if token.dep_ in ('nsubj', 'nsubjpass'):
                subjects.append(token.text)
                candidates[token.text] += 2
            elif token.dep_ in ('dobj', 'pobj', 'attr', 'oprd'):
                objects.append(token.text)
                candidates[token.text] += 1
            elif token.pos_ == 'PRON':
                pronouns.append(token.text)
                candidates[token.text] += 1

        if subjects:
            subjects = sorted(subjects, key=lambda x: (-candidates[x], sentence.index(x)))
            return subjects[0]

        for result in transition_results:
            if result['current_sentences'] == sentence:
                if result['transition'] == "Center Continuation (CON)":
                    continuation_nps = set(result['current_nps']).intersection(result['next_nps'])
                    if continuation_nps:
                        return next(iter(continuation_nps))
                elif result['transition'] in ("Smooth Shift (SSH)", "Rough Shift (RSH)"):
                    objects = sorted(objects, key=lambda x: (-candidates[x], sentence.index(x)))
                    if objects:
                        return objects[0]
                    elif pronouns:
                        return pronouns[0]

        valid_noun_phrases = [np for np in noun_phrases if np in candidates]
        if valid_noun_phrases:
            return max(valid_noun_phrases, key=candidates.get, default=None)
        return None
    def choose_word_with_context(self, next_words, context_word=None, semantic_threshold=0.05, position_index=0, structure_template=None, prev_pos=None, pos_bigrams=None):
        if not next_words:
            return None

        word_choices = list(next_words.keys())
        probabilities = np.array(list(next_words.values()), dtype=float)
        probabilities = np.maximum(probabilities, 0)

        total = probabilities.sum()
        if total > 0:
            probabilities /= total
        else:
            probabilities = np.ones_like(probabilities) / len(probabilities)

        valid_words, valid_vectors, valid_probs, valid_pos = [], [], [], []

        if structure_template:
            target_pos = structure_template[position_index % len(structure_template)]
            for word, prob in zip(word_choices, probabilities):
                doc = nlp(word)
                if doc and len(doc) > 0 and doc[0].pos_ == target_pos:
                    valid_words.append(word)
                    valid_vectors.append(doc[0].vector)
                    valid_probs.append(prob)
                    valid_pos.append(doc[0].pos_)
            if not valid_words:
                self.log(f"[WARN] No matching POS '{target_pos}'. Fallback to all words.")
                valid_words = word_choices
                valid_vectors = [nlp(word).vector for word in word_choices]
                valid_probs = probabilities
                valid_pos = [nlp(word)[0].pos_ for word in word_choices]
        else:
            valid_words = word_choices
            valid_vectors = [nlp(word).vector for word in word_choices]
            valid_probs = probabilities
            valid_pos = [nlp(word)[0].pos_ for word in word_choices]

        word_vectors = np.array(valid_vectors)
        probabilities = np.array(valid_probs)

        if context_word:
            context_vector = nlp(context_word).vector
            if context_vector is None or np.all(context_vector == 0):
                self.log("[ERROR] Context vector is empty. Fallback to uniform sampling.")
                similarity_scores = np.ones_like(probabilities)
            else:
                context_vector = context_vector.reshape(1, -1)
                similarity_scores = cosine_similarity(context_vector, word_vectors).flatten()
                similarity_scores = np.maximum(similarity_scores, 0)
                similarity_scores[similarity_scores < semantic_threshold] = 0
        else:
            similarity_scores = np.ones_like(probabilities)

        if pos_bigrams and prev_pos:
            transition_scores = np.array([
                pos_bigrams.get((prev_pos, curr_pos), 0.01)
                for curr_pos in valid_pos
            ])
        else:
            transition_scores = np.ones_like(probabilities)

        combined_scores = similarity_scores * probabilities * transition_scores

        if combined_scores.sum() == 0:
            self.log("[WARN] Combined scores zero. Fallback to uniform distribution.")
            combined_scores = np.ones_like(probabilities) / len(probabilities)
        else:
            combined_scores /= combined_scores.sum()

        chosen_word = np.random.choice(valid_words, p=combined_scores)
        self.log(f"[LOG] ➡️ Chosen Word: '{str(chosen_word)}' | Target POS: {structure_template[position_index % len(structure_template)] if structure_template else 'N/A'} | Chosen POS: {nlp(str(chosen_word))[0].pos_}")
        return chosen_word

    def clean_text(self, text):
        if not text:
            return ""
        text = ' '.join(text.split())
        text = re.sub(r'\s([?.!,:;])', r'\1', text)
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        text = text.strip("'\"")
        text = text[0].capitalize() + text[1:] if text else ""
        if text and text[-1] not in ['.', '!', '?']:
            text += '.'
        text = re.sub(r'(\w)([.,!?;])', r'\1 \2', text)
        text = re.sub(r'([,.!?;])(\w)', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\s)([.,!?;])', r'\2', text)
        return text

    def reorder_sentence(sentence):
        """Reorder the sentence structure based on dependency parsing."""
        doc = nlp(sentence)
        reordered_tokens = []

        # Create a list of tokens based on their dependency types
        subjects = []
        verbs = []
        objects = []
        modifiers = []

        for token in doc:
            if token.dep_ in ('nsubj', 'nsubjpass'):  # Subjects
                subjects.append(token)
            elif token.dep_ in ('ROOT', 'VERB'):  # Verbs
                verbs.append(token)
            elif token.dep_ in ('dobj', 'pobj'):  # Direct objects
                objects.append(token)
            else:  # Modifiers (adjectives, adverbs, etc.)
                modifiers.append(token)

        # Enhanced reordering strategy: Subject-Verb-Object with checks
        if subjects and verbs:
            # Select a subject, ensuring it's unique to avoid ambiguity
            selected_subject = random.choice(subjects)
            reordered_tokens.append(selected_subject)

            # Select a verb, ensuring it agrees with the subject (singular/plural)
            selected_verb = None
            for verb in verbs:
                if (selected_subject.tag_ == 'NNS' and verb.tag_ == 'VBZ') or \
                (selected_subject.tag_ == 'NN' and verb.tag_ == 'VBP'):
                    continue  # Skip if there's a disagreement
                selected_verb = verb
                break
            if selected_verb:
                reordered_tokens.append(selected_verb)

            # Select objects, ensuring there's no conflict
            if objects:
                # Ensure at least one object is present
                selected_objects = [obj for obj in objects if obj.head == selected_verb]  # Ensure they relate to the selected verb
                if selected_objects:
                    reordered_tokens.extend(selected_objects)

            # Add modifiers based on their relationship with the subject/verb
            if selected_verb:
                for mod in modifiers:
                    if mod.head == selected_verb or mod.head == selected_subject:
                        reordered_tokens.append(mod)

        # Ensure at least one subject, verb, and object to form a complete thought
        if not reordered_tokens or len(reordered_tokens) < 3:
            return "Sentence could not be reordered meaningfully."

        return " ".join([token.text for token in reordered_tokens])


    def generate_and_post_process(self, num_sentences=10, input_words=None, length=20):
        generated_sentences = []
        max_attempts = 5

        for i in tqdm(range(num_sentences), desc="Generating sentences", position=1, leave=True, dynamic_ncols=True, mininterval=0.05, maxinterval=0.3):

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
                        self.log(f"[SKIP] Not a complete thought: {generated_sentence}")
                        attempts += 1
                        continue
                    generated_sentences.append(generated_sentence)
                    coherent_sentence = True
                else:
                    attempts += 1
                    self.log(f"Attempt {attempts}: Generated incoherent sentence: {generated_sentence}")

            if not coherent_sentence:
                self.log(f"Max attempts reached for generating sentence {i + 1}. Adding the incoherent sentence.")
                generated_sentence = self.correct_grammar(generated_sentence)
                generated_sentences.append(generated_sentence)

        final_text = ' '.join(generated_sentences)
        final_text = self.correct_grammar(final_text)
        final_text = self.post_process_text(final_text)
        return final_text

    def advanced_length_adjustment(self, last_sentence, base_length):
        last_words = last_sentence.split()
        last_length = len(last_words)
        clause_count = sum(last_sentence.count(conj) for conj in [',', 'and', 'but', 'or', 'yet']) + 1
        doc = nlp(last_sentence)
        noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
        verb_count = sum(1 for token in doc if token.pos_ == "VERB")
        adjective_count = sum(1 for token in doc if token.pos_ == "ADJ")
        adverb_count = sum(1 for token in doc if token.pos_ == "ADV")
        complexity_factor = ((noun_count + verb_count + adjective_count + adverb_count) +
                            sum(1 for token in doc if token.dep_ in {"conj", "advcl", "relcl"})) // 2
        length_variability = ((last_length - base_length) + complexity_factor) // 3
        adjusted_length = max(5, min(base_length + random.randint(-3, 3) + clause_count + complexity_factor + length_variability, 16))
        return adjusted_length

    def rewrite_ill_formed_sentence(self, sentence):
        doc = nlp(sentence)
        tokens = list(doc)

        has_subject = any(tok.dep_ in ('nsubj', 'nsubjpass', 'expl') for tok in tokens)
        has_verb = any(tok.pos_ in ('VERB', 'AUX') for tok in tokens)

        # Eğer yüklem yoksa basit bir yüklem ekle
        if has_subject and not has_verb:
            sentence += " is."

        # Eğer hem özne hem yüklem yoksa, ya da çok fazla bağlaç varsa:
        num_conj = sum(1 for tok in tokens if tok.dep_ == 'cc')
        if not has_subject or num_conj > 2:
            # Fazla bağlaç varsa: cümleyi parçalayıp ilk anlamlı kısmı tut
            chunks = re.split(r'\band\b|\bor\b|\bif\b|\bbecause\b|\bbut\b', sentence)
            chunks = [chunk.strip() for chunk in chunks if len(chunk.strip().split()) >= 3]
            if chunks:
                sentence = chunks[0].capitalize() + '.'

        # Çok kısa cümlelerde yapısal destek
        if len(tokens) <= 3 and not has_verb:
            sentence += " exists."

        return sentence


    def post_process_text(self, text):
        # === ENCODING düzeltmesi ===
        text = text.replace('â€ ™', "'").replace('â€"', '"').replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Non-ASCII karakterleri sil

        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        cleaned_sentences = []
        buffer_sentence = ""
        min_words_for_sentence = 4

        for sentence in sentences:
            cleaned_sentence = sentence.strip()

            if cleaned_sentence:
                # Baş harf düzeltmesi
                cleaned_sentence = cleaned_sentence[0].upper() + cleaned_sentence[1:]

                # Bozuk boşluk ve noktalama düzeltmesi
                cleaned_sentence = re.sub(r'\s+([,.!?])', r'\1', cleaned_sentence)
                cleaned_sentence = re.sub(r'([,.!?])\s+', r'\1 ', cleaned_sentence)
                cleaned_sentence = re.sub(r'\s{2,}', ' ', cleaned_sentence)
                cleaned_sentence = cleaned_sentence.rstrip(",")
                cleaned_sentence = re.sub(r'([.!?])\1+', r'\1', cleaned_sentence)
                cleaned_sentence = re.sub(r'^(And|But|Or)\b,?\s+', '', cleaned_sentence, flags=re.IGNORECASE)
                cleaned_sentence = re.sub(r'"(.*?)"', r'“\1”', cleaned_sentence)

                # === Cümle sonu bağlaç kontrolü ===
                if re.search(r'\b(and|but|or|if|because)\.$', cleaned_sentence, flags=re.IGNORECASE):
                    cleaned_sentence = re.sub(r'\b(and|but|or|if|because)\.$', '', cleaned_sentence, flags=re.IGNORECASE).strip()
                    if not cleaned_sentence.endswith('.'):
                        cleaned_sentence += '.'

                # Named Entity düzeltmesi
                doc = nlp(cleaned_sentence)
                for entity in doc.ents:
                    if entity.label_ in ["PERSON", "ORG", "GPE"]:
                        cleaned_sentence = re.sub(r'\b' + re.escape(entity.text.lower()) + r'\b', entity.text, cleaned_sentence)

                # === Duplicate sözcükleri kaldırma ===
                words = cleaned_sentence.split()
                deduplicated_words = []
                previous_word = None
                for word in words:
                    if word.lower() != previous_word:
                        deduplicated_words.append(word)
                    previous_word = word.lower()
                cleaned_sentence = ' '.join(deduplicated_words)

                # === Kısa cümleleri birleştirme ===
                word_count = len(cleaned_sentence.split())
                if word_count < min_words_for_sentence:
                    buffer_sentence += " " + cleaned_sentence.lower()
                else:
                    if buffer_sentence.strip():
                        full_sentence = buffer_sentence.strip().capitalize() + " " + cleaned_sentence
                        cleaned_sentences.append(full_sentence.strip())
                        buffer_sentence = ""
                    else:
                        cleaned_sentences.append(cleaned_sentence)

        if buffer_sentence.strip():
            cleaned_sentences.append(buffer_sentence.strip().capitalize())

        # Final düzeltmeler
        final_output = ' '.join(cleaned_sentences).strip()
        if final_output and not final_output.endswith('.'):
            final_output += '.'

        final_output = self.correct_grammar(final_output)
        return final_output



    def is_sentence_coherent(self, sentence, previous_sentences=None):
        if not sentence or len(sentence.split()) < 4 or sentence[-1] not in ['.', '!', '?']:
            return False

        current_doc = nlp(sentence)
        current_embedding = current_doc.vector

        if previous_sentences is None or len(previous_sentences) == 0:
            return True

        previous_embeddings = [nlp(prev).vector for prev in previous_sentences if prev.strip()]
        if not previous_embeddings:
            return True

        # ✨ Son 2 cümleye daha fazla ağırlık ver
        weights = np.linspace(1.5, 1.0, num=len(previous_embeddings))  # Yakın cümleler daha etkili
        similarities = []
        for emb, weight in zip(previous_embeddings, weights):
            sim = np.dot(current_embedding, emb) / (np.linalg.norm(current_embedding) * np.linalg.norm(emb) + 1e-8)
            similarities.append(sim * weight)

        avg_similarity = np.mean(similarities)

        # ✨ Sentence complexity
        noun_count = sum(1 for token in current_doc if token.pos_ == "NOUN")
        verb_count = sum(1 for token in current_doc if token.pos_ == "VERB")
        adj_count = sum(1 for token in current_doc if token.pos_ == "ADJ")
        adv_count = sum(1 for token in current_doc if token.pos_ == "ADV")
        clause_count = sum(1 for token in current_doc if token.dep_ in ("advcl", "relcl", "ccomp", "xcomp"))
        complexity_factor = (noun_count + verb_count + adj_count + adv_count + clause_count) / 4.0

        avg_len = np.mean([len(prev.split()) for prev in previous_sentences])
        var_len = np.var([len(prev.split()) for prev in previous_sentences])

        # ✨ Dinamik threshold
        threshold = 0.55  # biraz daha insaflı başlıyoruz
        if avg_len > 15:
            threshold += 0.05
        elif avg_len < 8:
            threshold -= 0.05
        if var_len > 5:
            threshold += 0.03
        if complexity_factor > 2.5:
            threshold += 0.03
        elif complexity_factor < 1.5:
            threshold -= 0.03

        # ✨ Ekstra ufak düzen: Eğer cümlede çok az özgün lemma varsa thresholdu hafif artır
        unique_lemmas = set(token.lemma_ for token in current_doc if token.pos_ in {"NOUN", "VERB"})
        if len(unique_lemmas) < 3:
            threshold += 0.02

        return avg_similarity > threshold


    def save_model(self, filename, compress=False):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            model_data = (self.model, self.total_counts)
            if compress:
                import gzip
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(model_data, f)
            else:
                with open(filename, 'wb') as f:
                    pickle.dump(model_data, f)
            self.log(f"Model saved successfully to {filename}")
        except IOError as e:
            self.log(f"Error saving model to {filename}: {e}")
        except Exception as e:
            self.log(f"An unexpected error occurred: {e}")

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            model, total_counts = pickle.load(f)
        instance = cls("dummy text")
        instance.model = defaultdict(lambda: defaultdict(int), model)
        instance.total_counts = total_counts
        return instance

def load_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            text = ' '.join(text.split())
            text = text.strip()
            doc = nlp(text)
            cleaned_tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
            cleaned_text = ' '.join(cleaned_tokens)
            return nlp(cleaned_text)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    return None

# --- Ana Akış ---
file_path = text_path
text = load_text_from_file(file_path)
model_file = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\language_model.pkl'

try:
    language_model = EnhancedLanguageModel.load_model(model_file)
    language_model.log("Loaded existing model.")
except (FileNotFoundError, EOFError):
    language_model = EnhancedLanguageModel(text, n=4)
    language_model.save_model(model_file)
    language_model.log("Created and saved new model.")

num_sentences = 5
input_words = "I remember those days.".split()
generated_text = language_model.generate_and_post_process(num_sentences=num_sentences, input_words=input_words, length=10)
language_model.log("Generated Text:\n" + generated_text)
print("Generated Text:\n" + generated_text)
