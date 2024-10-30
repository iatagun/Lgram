import os
import spacy
import random
import pickle
from collections import defaultdict, Counter
from itertools import islice
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm

# SpaCy modelini yükle
nlp = spacy.load("en_core_web_lg")  # veya "en_core_web_lg"


def generate_ngrams(tokens, n):
    """n-gram oluşturmak için yardımcı fonksiyon."""
    return zip(*[islice(tokens, i, None) for i in range(n)])

def build_ngram_model(text_path, bigram_path, trigram_path, fourgram_path, fivegram_path, sixgram_path):
    """Metin dosyasından bigram, trigram, fourgram, fivegram ve sixgram modelini oluşturup kaydeder."""
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

        # Bigram oluştur
        print("Bigram modeli oluşturuluyor...")
        for bigram in tqdm(generate_ngrams(tokens, 2), total=len(tokens)-1):
            bigram_model[bigram[:1]][bigram[1]] += 1

        # Trigram oluştur
        print("Trigram modeli oluşturuluyor...")
        for trigram in tqdm(generate_ngrams(tokens, 3), total=len(tokens)-2):
            trigram_model[trigram[:2]][trigram[2]] += 1

        # Fourgram oluştur
        print("Fourgram modeli oluşturuluyor...")
        for fourgram in tqdm(generate_ngrams(tokens, 4), total=len(tokens)-3):
            fourgram_model[fourgram[:3]][fourgram[3]] += 1

        # Fivegram oluştur
        print("Fivegram modeli oluşturuluyor...")
        for fivegram in tqdm(generate_ngrams(tokens, 5), total=len(tokens)-4):
            fivegram_model[fivegram[:4]][fivegram[4]] += 1

        # Sixgram oluştur
        print("Sixgram modeli oluşturuluyor...")
        for sixgram in tqdm(generate_ngrams(tokens, 6), total=len(tokens)-5):
            sixgram_model[sixgram[:5]][sixgram[5]] += 1

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
    return similarity * (freq ** 0.5)  # Sıklığı kök alma ile hafifçe ağırlıklandır

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

    # Basic reordering strategy: Subject-Verb-Object
    if subjects and verbs:
        reordered_tokens.append(random.choice(subjects))  # Select one subject
        if verbs:
            reordered_tokens.append(random.choice(verbs))  # Select one verb
        if objects:
            reordered_tokens.extend(objects)  # Add all objects
        reordered_tokens.extend(modifiers)  # Add all modifiers

    return " ".join([token.text for token in reordered_tokens])

def generate_paraphrase(text, bigram_model, trigram_model, fourgram_model, fivegram_model, sixgram_model):
    """n-gram modellerini kullanarak paraphrase oluşturur."""
    doc = nlp(text)
    paraphrased_sentences = []

    # Metni cümlelere ayır
    sentences = list(doc.sents)

    for sentence in sentences:
        paraphrased_text = []
        for token in sentence:
            word = token.text
            lemma = token.lemma_

            # Bağlama uygun n-gram seçenekleri belirleme
            context = tuple(paraphrased_text[-5:]) if len(paraphrased_text) >= 5 else None
            if context in sixgram_model:
                choices = sixgram_model[context]
                paraphrased_text.append(select_best_match(word, choices))
            elif len(paraphrased_text) >= 4 and tuple(paraphrased_text[-4:]) in fivegram_model:
                context = tuple(paraphrased_text[-4:])
                choices = fivegram_model[context]
                paraphrased_text.append(select_best_match(word, choices))
            elif len(paraphrased_text) >= 3 and tuple(paraphrased_text[-3:]) in fourgram_model:
                context = tuple(paraphrased_text[-3:])
                choices = fourgram_model[context]
                paraphrased_text.append(select_best_match(word, choices))
            elif len(paraphrased_text) >= 2 and tuple(paraphrased_text[-2:]) in trigram_model:
                context = tuple(paraphrased_text[-2:])
                choices = trigram_model[context]
                paraphrased_text.append(select_best_match(word, choices))
            elif (lemma,) in bigram_model:
                # Bigram ile bağlama uygun bir kelime seçimi
                choices = bigram_model[(lemma,)]
                paraphrased_text.append(select_best_match(word, choices))
            else:
                # Hiçbir eşleşme yoksa kelimeyi doğrudan ekle
                paraphrased_text.append(word)

        # Birleştir ve cümle sonunu nokta ile bitir, büyük harfle başlat
        final_sentence = " ".join(paraphrased_text).strip()
        if final_sentence:  # Eğer cümle boş değilse
            final_sentence = final_sentence[0].upper() + final_sentence[1:]  # İlk harfi büyük yap
            if not final_sentence.endswith('.'):  # Eğer cümle noktada bitmiyorsa nokta ekle
                final_sentence += '.'
            paraphrased_sentences.append(final_sentence)

    return " ".join(paraphrased_sentences)


