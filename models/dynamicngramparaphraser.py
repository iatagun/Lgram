import os
import spacy
import random
import pickle
from collections import defaultdict, Counter
from itertools import islice
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import json

# SpaCy modelini yükle
nlp = spacy.load("en_core_web_lg")  # veya "en_core_web_lg"
corrections_file = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\corrections.json"

def generate_ngrams(tokens, n):
    """n-gram oluşturmak için yardımcı fonksiyon."""
    return zip(*[islice(tokens, i, None) for i in range(n)])

def build_ngram_model(text_path, bigram_path, trigram_path, fourgram_path, fivegram_path, sixgram_path, frequency_threshold=1):
    """Metin dosyasından n-gram modelini oluşturup kaydeder, düşük frekanslı n-gramları filtreler."""
    bigram_model = defaultdict(Counter)
    trigram_model = defaultdict(Counter)
    fourgram_model = defaultdict(Counter)
    fivegram_model = defaultdict(Counter)
    sixgram_model = defaultdict(Counter)

    # Metin dosyasını oku ve n-gram modellerini oluştur
    with open(text_path, "r", encoding="utf-8") as file:
        text = file.read()
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_punct]
        
        # Bağlama duyarlı özellikler için kelime türü ve cümle bilgisi
        pos_tags = [token.pos_ for token in doc if not token.is_punct]  # Kelime türlerini al
        sentences = list(doc.sents)  # Cümleleri listele

        # N-gram modellerini oluşturma ve frekans eşiklerini uygulama
        def add_ngram_to_model(ngram_model, ngrams, context=None):
            for ngram in ngrams:
                prefix = ngram[:-1]
                last_token = ngram[-1]
                
                # Eğer bir bağlam varsa, bağlamı da kullanarak n-gram'ı güncelle
                if context:
                    context_key = tuple(context)
                    ngram_model[context_key][last_token] += 1
                else:
                    ngram_model[prefix][last_token] += 1

        # Bigram modeli
        print("Bigram modeli oluşturuluyor...")
        add_ngram_to_model(bigram_model, generate_ngrams(tokens, 2))

        # Trigram modeli
        print("Trigram modeli oluşturuluyor...")
        add_ngram_to_model(trigram_model, generate_ngrams(tokens, 3))

        # Fourgram modeli
        print("Fourgram modeli oluşturuluyor...")
        add_ngram_to_model(fourgram_model, generate_ngrams(tokens, 4))

        # Fivegram modeli
        print("Fivegram modeli oluşturuluyor...")
        add_ngram_to_model(fivegram_model, generate_ngrams(tokens, 5))

        # Sixgram modeli
        print("Sixgram modeli oluşturuluyor...")
        add_ngram_to_model(sixgram_model, generate_ngrams(tokens, 6))

    # Modelleri kaydetmeden önce frekans eşiği uygulayın
    def filter_by_frequency(ngram_model, threshold):
        return {prefix: Counter({word: count for word, count in suffixes.items() if count >= threshold})
                for prefix, suffixes in ngram_model.items() if any(count >= threshold for count in suffixes.values())}

    bigram_model = filter_by_frequency(bigram_model, frequency_threshold)
    trigram_model = filter_by_frequency(trigram_model, frequency_threshold)
    fourgram_model = filter_by_frequency(fourgram_model, frequency_threshold)
    fivegram_model = filter_by_frequency(fivegram_model, frequency_threshold)
    sixgram_model = filter_by_frequency(sixgram_model, frequency_threshold)

    # Modelleri pickle dosyasına kaydet
    with open(bigram_path, "wb") as f:
        pickle.dump(dict(bigram_model), f)
    with open(trigram_path, "wb") as f:
        pickle.dump(dict(trigram_model), f)
    with open(fourgram_path, "wb") as f:
        pickle.dump(dict(fourgram_model), f)
    with open(fivegram_path, "wb") as f:
        pickle.dump(dict(fivegram_model), f)
    with open(sixgram_path, "wb") as f:
        pickle.dump(dict(sixgram_model), f)

    print("N-gram modelleri başarıyla oluşturuldu ve kaydedildi.")


    # Modelleri kaydetmeden önce frekans eşiği uygulayın
    def filter_by_frequency(ngram_model, threshold):
        return {prefix: Counter({word: count for word, count in suffixes.items() if count >= threshold})
                for prefix, suffixes in ngram_model.items() if any(count >= threshold for count in suffixes.values())}

    bigram_model = filter_by_frequency(bigram_model, frequency_threshold)
    trigram_model = filter_by_frequency(trigram_model, frequency_threshold)
    fourgram_model = filter_by_frequency(fourgram_model, frequency_threshold)
    fivegram_model = filter_by_frequency(fivegram_model, frequency_threshold)
    sixgram_model = filter_by_frequency(sixgram_model, frequency_threshold)

    # Modelleri pickle dosyasına kaydet
    with open(bigram_path, "wb") as f:
        pickle.dump(dict(bigram_model), f)
    with open(trigram_path, "wb") as f:
        pickle.dump(dict(trigram_model), f)
    with open(fourgram_path, "wb") as f:
        pickle.dump(dict(fourgram_model), f)
    with open(fivegram_path, "wb") as f:
        pickle.dump(dict(fivegram_model), f)
    with open(sixgram_path, "wb") as f:
        pickle.dump(dict(sixgram_model), f)

    print("N-gram modelleri başarıyla oluşturuldu ve kaydedildi.")

def load_ngram_model(bigram_path, trigram_path, fourgram_path, fivegram_path, sixgram_path):
    """Bigram, trigram, fourgram, fivegram ve sixgram modellerini yükler."""
    with open(bigram_path, "rb") as f:
        bigram_model = pickle.load(f)
    with open(trigram_path, "rb") as f:
        trigram_model = pickle.load(f)
    with open(fourgram_path, "rb") as f:
        fourgram_model = pickle.load(f)
    with open(fivegram_path, "rb") as f:
        fivegram_model = pickle.load(f)
    with open(sixgram_path, "rb") as f:
        sixgram_model = pickle.load(f)
    return bigram_model, trigram_model, fourgram_model, fivegram_model, sixgram_model

def calculate_weighted_similarity(word_token, choice_token, freq, normalization=True):
    """Verilen kelime ve seçim için ağırlıklı benzerliği hesaplar."""
    
    # Cosinus benzerliğini hesaplamak için gerekli vektörler
    word_vector = word_token.vector
    choice_vector = choice_token.vector

    if normalization:
        # Vektörleri normalize et
        word_vector = word_vector / norm(word_vector) if norm(word_vector) != 0 else word_vector
        choice_vector = choice_vector / norm(choice_vector) if norm(choice_vector) != 0 else choice_vector

    # Cosinus benzerliği
    similarity = dot(word_vector, choice_vector)

    # Ağırlıklı benzerliği döndür
    return similarity * (freq ** 0.7)  # Sıklığı kök alma ile hafifçe ağırlıklandır

def select_best_match(word, choices):
    """Orijinal kelimeye en yakın eşleşmeyi seçer."""
    if isinstance(choices, list):  # Eğer `choices` bir listeyse Counter'a çevir
        choices = Counter(choices)

    best_choice = word  # Default: orijinal kelime
    max_similarity = 0
    word_token = nlp(word)[0]

    if not word_token.has_vector:
        print(f"Hata: '{word}' kelimesinin vektörü yok.")
        return best_choice

    for choice, freq in choices.items():
        choice_token = nlp(choice)[0]
        if choice_token.has_vector:  # Eşleşen kelimenin vektörü varsa
            weighted_similarity = calculate_weighted_similarity(word_token, choice_token, freq)
            
            # Benzerlik değerini güncelle
            if weighted_similarity > max_similarity:
                max_similarity = weighted_similarity
                best_choice = choice
        else:
            print(f"Uyarı: '{choice}' kelimesinin vektörü yok, bu nedenle dikkate alınmıyor.")

    return best_choice  # En iyi eşleşmeyi döner

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

def correct_grammar(sentence):
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


def generate_paraphrase(text, bigram_model, trigram_model, fourgram_model, fivegram_model, sixgram_model):
    """Bağımlılık ilişkilerini kullanarak n-gram modelleri ile paraphrase oluşturur."""
    doc = nlp(text)
    paraphrased_sentences = []

    # Her cümleyi işle
    for sentence in doc.sents:
        paraphrased_text = []
        
        for token in sentence:
            word = token.text
            lemma = token.lemma_

            # Bağımlılık ilişkisine göre en iyi n-gram seçimi
            dep_context = (token.head.text, token.dep_, token.pos_)  # Bağlı olduğu kelime, bağımlılık türü ve sözcük türü

            # Bağımlılık bağlamına göre n-gram seçimi
            if len(paraphrased_text) >= 5 and dep_context in sixgram_model:
                choices = sixgram_model[dep_context]
                paraphrased_text.append(select_best_match(word, choices))
            elif len(paraphrased_text) >= 4 and dep_context in fivegram_model:
                choices = fivegram_model[dep_context]
                paraphrased_text.append(select_best_match(word, choices))
            elif len(paraphrased_text) >= 3 and dep_context in fourgram_model:
                choices = fourgram_model[dep_context]
                paraphrased_text.append(select_best_match(word, choices))
            elif len(paraphrased_text) >= 2 and dep_context in trigram_model:
                choices = trigram_model[dep_context]
                paraphrased_text.append(select_best_match(word, choices))
            elif (lemma,) in bigram_model:
                choices = bigram_model[(lemma,)]
                paraphrased_text.append(select_best_match(word, choices))
            else:
                paraphrased_text.append(word)

        # Cümleyi birleştir ve biçimlendir
        final_sentence = " ".join(paraphrased_text).strip()
        if final_sentence:
            final_sentence = final_sentence[0].upper() + final_sentence[1:]
            if not final_sentence.endswith('.'):
                final_sentence += '.'
            paraphrased_sentences.append(final_sentence)
    paraphrased_sentences = " ".join(paraphrased_sentences)
    paraphrased_sentences = correct_grammar(paraphrased_sentences)
    return paraphrased_sentences


# N-gram modellerini kontrol et ve yükle
text_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\text_data.txt"
bigram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\bigram_model.pkl"
trigram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\trigram_model.pkl"
fourgram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\fourgram_model.pkl"
fivegram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\fivegram_model.pkl"
sixgram_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\sixgram_model.pkl"

build_ngram_model(text_path, bigram_path, trigram_path, fourgram_path, fivegram_path, sixgram_path)