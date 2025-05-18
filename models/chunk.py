import random
import pickle
from collections import defaultdict, Counter
import spacy
import numpy as np
from tqdm import tqdm
import sys
import time
import os
import re
import json
import datetime
from transition_analyzer import TransitionAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from functools import cache
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. Model ve tokenizer'ı başlat
tokenizer = AutoTokenizer.from_pretrained("pszemraj/flan-t5-large-grammar-synthesis")
model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/flan-t5-large-grammar-synthesis")
model.eval()

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
with open(corrections_file, encoding="utf-8") as f:
    CORRECTIONS = json.load(f)



class EnhancedLanguageModel:
    def __init__(self, text, n=2, colloc_path="collocations.pkl"):
        self.n = n
        self.model, self.total_counts = self.build_model(text)
        self.load_ngram_models()
        self.collocations = self._load_collocations(colloc_path)

    @cache
    def _load_collocations(self, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            # Hata yönetimi: dosya yoksa boş dict dön
            return {}

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

    def choose_next_word_dynamically(self, prefix):
        model_priority = ['sixgram_model', 'fivegram_model', 'fourgram_model', 'trigram_model', 'bigram_model']
        candidates = []

        # Toplu aday listesi oluştur
        for model_attr in model_priority:
            model = getattr(self, model_attr, None)
            if model:
                next_words = model.get(prefix)
                if next_words:
                    candidates.append((model_attr, next_words))

        if not candidates:
            return None

        weighted_candidates = []
        for model_attr, words_dict in candidates:
            weight = model_priority.index(model_attr) + 1
            for word, prob in words_dict.items():
                adjusted_prob = prob / weight

                # 🔥 Semantic relation kontrolü
                if self.is_semantically_related(prefix, word):
                    adjusted_prob *= 1.7  # %70 ekstra ağırlık

                # 🤝 Collocation bonusu (PMI tabanlı)
                last = prefix[-1] if prefix else None
                if last and hasattr(self, 'collocations'):
                    bonus = self.collocations.get(last, {}).get(word, 0)
                    # PMI genellikle 0–2 arası değerler aldığı için (1+bonus) formülü uygundur
                    adjusted_prob *= (1 + bonus)

                weighted_candidates.append((word, adjusted_prob))

        if not weighted_candidates:
            return None

        # Kelime ve ağırlıkları ayır
        words, scores = zip(*weighted_candidates)
        # Random seçim
        chosen_word = random.choices(words, weights=scores, k=1)[0]

        return chosen_word

    def is_semantically_related(self, prefix, word):
        """
        Prefix'in son kelimesi ile candidate word arasında semantic similarity ölçer.
        """
        if not prefix:
            return False

        try:
            last_word = prefix[-1]
            token1 = nlp(last_word)[0]
            token2 = nlp(word)[0]

            if token1.has_vector and token2.has_vector:
                similarity = 1 - cosine(token1.vector, token2.vector)
                if similarity > 0.7:  # Threshold seçiyoruz
                    return True
        except:
            pass

        return False


    def generate_sentence(self, start_words=None, length=15):
        # Generate a raw sentence without context-based adjustments
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
            raw_next_word = self.choose_next_word_dynamically(prefix)
            if not raw_next_word:
                break
            current_words.append(raw_next_word)
            sentence.append(raw_next_word)

        sentence_text = " ".join(sentence)
        return sentence_text

    def correct_grammar(self, text: str) -> str:
        """
        corrections.json içindeki yanlış-doğru eşleştirmelerine
        dayalı basit kural tabanlı gramer düzeltme.
        """
        for wrong, right in CORRECTIONS.items():
            # Kelime bütünlüğünü koruyarak değiştir
            pattern = re.compile(rf"\b{re.escape(wrong)}\b", flags=re.IGNORECASE)
            text = pattern.sub(right, text)
        return text

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


    def get_center_from_sentence(self, prev_sentence, current_sentence, transition_analyzer, p_alt=0.03):
        def compute_Fc(sent):
            doc = nlp(sent)
            sal = []
            for tok in doc:
                if tok.dep_ in ('nsubj', 'nsubjpass'):
                    sal.append((tok.text, 3))
                elif tok.pos_ == 'PRON':
                    sal.append((tok.text, 2))
                elif tok.dep_ in ('dobj', 'pobj', 'attr', 'oprd'):
                    sal.append((tok.text, 1))
            sal_sorted = sorted(sal, key=lambda x: (-x[1], sent.find(x[0])))
            return [w for w, _ in sal_sorted]

        Cf_prev = compute_Fc(prev_sentence)
        Cf_curr = compute_Fc(current_sentence)
        Cb = Cf_prev[0] if Cf_prev and Cf_prev[0] in Cf_curr else None

        # Transition analizini al
        results = transition_analyzer.analyze()
        for res in results:
            tr = res.get('transition')
            if tr == "Center Continuation (CON)":
                # Bazen ikinci merkeze geç
                if Cf_curr and random.random() < p_alt and len(Cf_curr) > 1:
                    return Cf_curr[1]
                return Cb
            if tr in ("Smooth Shift (SSH)", "Rough Shift (RSH)"):
                # Smooth Shift’te de arada bir farklı shift
                if Cf_curr and random.random() < p_alt and len(Cf_curr) > 1:
                    return Cf_curr[1]
                return Cf_curr[0] if Cf_curr else None

        # Fallback
        if Cb:
            return Cb if random.random() > p_alt else (Cf_curr[1] if len(Cf_curr) > 1 else Cb)
        # Tamamen shift modu: bazen ikinci, bazen ilk
        return (Cf_curr[1] if len(Cf_curr) > 1 and random.random() < p_alt else
                (Cf_curr[0] if Cf_curr else None))

    

    def choose_word_with_context(
        self,
        next_words,
        context_word=None,
        semantic_threshold=0.9,
        position_index=0,
        structure_template=None,
        prev_pos=None,
        pos_bigrams=None
    ):
        if not next_words:
            return None

        # 1) Temel olasılıklar
        words = list(next_words.keys())
        probs = np.array(list(next_words.values()), dtype=float)
        probs = np.clip(probs, 0, None)
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones_like(probs) / len(probs)

        # 2) POS ve vektörleri bir kerede çıkar
        docs = [nlp(w) for w in words]
        pos_tags = [doc[0].pos_ if doc and doc[0].pos_ else None for doc in docs]
        vector_dim = nlp.vocab.vectors_length
        vectors = np.array([
            doc[0].vector if doc and doc[0].has_vector else np.zeros((vector_dim,))
            for doc in docs
        ])

        # 3) Şablon bazlı filtreleme (structure_template)
        if structure_template:
            target_pos = structure_template[position_index % len(structure_template)]
            mask = [pos == target_pos for pos in pos_tags]
            if any(mask):
                words      = [w for w, m in zip(words, mask) if m]
                probs      = np.array([p for p, m in zip(probs, mask) if m])
                vectors    = np.array([v for v, m in zip(vectors, mask) if m])
                pos_tags   = [pos for pos, m in zip(pos_tags, mask) if m]
            else:
                self.log(f"[WARN] No words match POS '{target_pos}', skipping template filter.")

        # 4) Anlamsal benzerlik
        if context_word:
            ctx_doc = nlp(str(context_word))
            if ctx_doc and ctx_doc[0].has_vector:
                ctx_vec = ctx_doc[0].vector.reshape(1, -1)
                sim_scores = cosine_similarity(ctx_vec, vectors).flatten()
                sim_scores = np.clip(sim_scores, 0, None)
                sim_scores[sim_scores < semantic_threshold] = 0
            else:
                sim_scores = np.ones_like(probs)
        else:
            sim_scores = np.ones_like(probs)

        # 5) POS-bigram geçiş skoru
        if prev_pos and pos_bigrams:
            trans_scores = np.array([
                pos_bigrams.get((prev_pos, pos), 0.01)
                for pos in pos_tags
            ])
        else:
            trans_scores = np.ones_like(probs)

        # 6) Collocation bonus
        if context_word and hasattr(self, 'collocations'):
            coll_map    = self.collocations.get(str(context_word), {})
            coll_bonus  = np.array([coll_map.get(w, 0.0) for w in words])
            coll_factor = 1 + coll_bonus
        else:
            coll_factor = np.ones_like(probs)

        # 7) Puanları birleştir, normalize et
        combined = probs * sim_scores * trans_scores * coll_factor
        total = combined.sum()
        if total <= 0:
            self.log("[WARN] Combined scores zero; using uniform distribution.")
            combined = np.ones_like(combined) / len(combined)
        else:
            combined /= total

        # 8) Seçim
        choice = np.random.choice(words, p=combined).item()
        chosen_pos = nlp(choice)[0].pos_
        self.log(f"[LOG] ➡️ Chosen: '{choice}' | POS: {chosen_pos}")
        return choice




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
        text = re.sub(r'([.,!?;])(\s)', r'\1', text)
        return text

    def reorder_sentence(self, sentence):
        """Reorder the sentence structure based on dependency parsing: Subject-Verb-Object(+Modifiers)."""
        # 1. Hızlı parse: sadece parser’ı çalıştır, NER’i kapat
        doc = nlp(sentence, disable=["ner"])

        # 2. Gerçek zamanlı (streaming) kategori toplama
        subjects, verbs, objects, modifiers = [], [], [], []
        for tok in doc:
            if tok.dep_ in ('nsubj', 'nsubjpass'):
                subjects.append(tok)
            elif tok.pos_ == 'VERB' or tok.dep_ == 'ROOT':
                verbs.append(tok)
            elif tok.dep_ in ('dobj', 'pobj', 'iobj'):
                objects.append(tok)
            elif tok.dep_ in ('amod', 'advmod', 'npadvmod', 'acomp'):
                modifiers.append(tok)

        # 3. Orijinal SVO mantığı
        reordered_tokens = []
        if subjects and verbs:
            # a) Özne: en büyük subtree’a sahip olanı al
            subj = max(subjects, key=lambda x: len(list(x.subtree)))
            reordered_tokens.extend(sorted(subj.subtree, key=lambda x: x.i))

            # b) Uyumlu yüklem
            sel_verb = None
            for verb in verbs:
                # basit özne-yüklem uyumsuzluklarını es geç
                if (subj.tag_ == 'NNS' and verb.tag_ == 'VBZ') or \
                (subj.tag_ == 'NN' and verb.tag_ == 'VBP'):
                    continue
                sel_verb = verb
                reordered_tokens.append(verb)
                break

            if sel_verb:
                # c) Yardımcı fiiller
                auxs = [t for t in doc if t.dep_ in ('aux', 'auxpass') and t.head == sel_verb]
                reordered_tokens.extend(sorted(auxs, key=lambda x: x.i))

                # d) Nesne cümlecikleri
                for obj in objects:
                    if obj.head == sel_verb:
                        reordered_tokens.extend(sorted(obj.subtree, key=lambda x: x.i))

                # e) Modifiers (özne veya yükleme bağlı)
                for mod in modifiers:
                    if mod.head in (subj, sel_verb):
                        reordered_tokens.append(mod)

        # 4. Yeterli öğe yoksa orijinaline dön
        key_toks = set(subjects + verbs + objects)
        if len([t for t in reordered_tokens if t in key_toks]) < 3:
            return sentence

        # 5. Tekilleştir ve orijinal sıraya göre sırala
        seen = set(); unique = []
        for t in reordered_tokens:
            if t.i not in seen:
                seen.add(t.i); unique.append(t)
        unique = sorted(unique, key=lambda x: x.i)

        # 6. Metni kur ve noktalama ekle
        out = " ".join(t.text for t in unique)
        if out and out[-1] not in '.!?':
            out += sentence.strip()[-1] if sentence.strip()[-1] in '.!?' else '.'
        return out
    
    def get_dynamic_pronoun(self, token):
        """
        Token'ın türüne, sayısına ve morfolojik özelliklerine göre uygun zamiri döndürür.
        Erkek/kadın/şirket/nesne/çoğul ayrımı otomatik!
        """
        # Çoğulsa "they"
        if token.tag_ in ("NNS", "NNPS") or token.morph.get("Number") == ["Plur"]:
            return "they"
        
        # İnsan mı? (PERSON entity veya muhtemel isim, özel isim)
        if token.ent_type_ == "PERSON" or (token.pos_ == "PROPN" and token.morph.get("Gender")):
            gender = token.morph.get("Gender")
            if "Masc" in gender:
                return "he"
            elif "Fem" in gender:
                return "she"
            else:
                return "they"  # cinsiyetsiz için
        
        # Kurum/organizasyon mu?
        if token.ent_type_ == "ORG":
            return "it"
        
        # Hayvan ya da cinsiyeti olmayan nesne, soyut kavram
        if token.ent_type_ in ("PRODUCT", "GPE", "LOC", "EVENT") or token.pos_ == "NOUN":
            return "it"
        
        # Hiçbiri değilse de default olarak "it"
        return "it"

    @cache
    def generate_and_post_process(self, num_sentences=10, input_words=None, length=15):
        generated_sentences = []
        max_attempts = 5
        context_word = None  # İlk center

        for i in tqdm(range(num_sentences), desc="Generating sentences", position=1, leave=False, dynamic_ncols=True, mininterval=0.05, maxinterval=0.3):
            attempts = 0
            coherent_sentence = False
            corrected_sentence = ""
            prev_sentence = generated_sentences[-1] if generated_sentences else None

            # 1) İlk cümlede input_words kullan, diğerlerinde context_word (center)
            if i == 0 and input_words:
                start_words = input_words
            elif context_word:
                start_words = [context_word]
            else:
                start_words = None

            # 2) Cümleyi üret, center ile başlamasını ZORUNLU tut
            while attempts < max_attempts and not coherent_sentence:
                raw_sentence = self.generate_sentence(start_words=start_words, length=length)
                # Eğer center ile başlamıyorsa brute-force başa ekle
                if start_words and not raw_sentence.lower().startswith(str(start_words[0]).lower()):
                    raw_sentence = str(start_words[0]) + " " + raw_sentence

                corrected_sentence = self.correct_grammar(raw_sentence)
                # Noktalama zorunluluğu
                if corrected_sentence and not corrected_sentence.endswith(('.', '!', '?')):
                    corrected_sentence += '.'

                # (İstersen coherence check ekleyebilirsin)
                #if self.is_sentence_coherent(corrected_sentence, generated_sentences):
                coherent_sentence = True
                attempts += 1

            # 3) Cümleyi ekle
            generated_sentences.append(corrected_sentence)

            # 4) Yeni center'ı bul (yeni cümleyle analiz)
            if prev_sentence:
                analyzer = TransitionAnalyzer(prev_sentence + " " + corrected_sentence)
                context_word = self.get_center_from_sentence(prev_sentence, corrected_sentence, analyzer)
                # Sonraki cümlelerde zamirleştir
                if last_entity_token is not None:
                    context_word = self.get_dynamic_pronoun(last_entity_token)
            else:
                # İlk cümlede context_word = ilk özne olsun
                doc = nlp(corrected_sentence)
                for tok in doc:
                    if tok.dep_ in ('nsubj', 'nsubjpass', 'expl'):
                        context_word = tok.text
                        last_entity_token = tok  # Entityyi sakla!
                        break

        # 5) Sonuçları birleştir, post-process et
        final_text = " ".join(generated_sentences)
        return self.post_process_text(final_text)



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
        adjusted_length = max(5, min(base_length + random.randint(-3, 3) + clause_count + complexity_factor + length_variability, 13))
        return adjusted_length

    def rewrite_ill_formed_sentence(self, sentence):
        """
        Basitçe biçimsiz cümleleri iyileştirir: eksik yüklem ekler,
        uzun ve karışık cümleleri bölerek ilk anlamlı kısmı döner.
        """
        import re

        # 1. İlk analiz
        doc = nlp(sentence)
        tokens = list(doc)
        has_subject = any(tok.dep_ in ('nsubj', 'nsubjpass', 'expl') for tok in tokens)
        has_verb = any(tok.pos_ in ('VERB', 'AUX') for tok in tokens)

        # 2. Yüklem eksikse ve özne varsa ekle
        if has_subject and not has_verb:
            sentence = sentence.rstrip('.!?') + ' is.'

        # 3. Yeniden analiz (güncellenmiş cümle üzerinde)
        doc = nlp(sentence)
        tokens = list(doc)
        has_subject = any(tok.dep_ in ('nsubj', 'nsubjpass', 'expl') for tok in tokens)
        has_verb = any(tok.pos_ in ('VERB', 'AUX') for tok in tokens)
        num_conj = sum(1 for tok in tokens if tok.dep_ == 'cc')

        # 4. Çok fazla bağlaç veya özne-yüklem yoksa ilk anlamlı kısmı al
        if not has_subject or num_conj > 2:
            split_pattern = r"\b(?:and|or|if|because|but|so|although|though|since|when|while)\b"
            parts = re.split(split_pattern, sentence, flags=re.IGNORECASE)
            # En az 3 kelime barındıran ilk parçayı seç
            valid = [p.strip() for p in parts if len(p.strip().split()) >= 3]
            if valid:
                first = valid[0].strip()
                first = first[0].upper() + first[1:].rstrip(' .!?') + '.'
                return first
            # Fallback: tek kelime varsa eklemeli bir cümle kur
            if tokens:
                return tokens[0].text.capitalize() + ' exists.'
            return 'Something exists.'

        # 5. Çok kısa ve yüklemsizse destek ekle
        if len(tokens) <= 3 and not has_verb:
            core = sentence.rstrip('.!?')
            return core + ' exists.'

        # 6. Zaten iyi formdaysa olduğu gibi dön
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

        # final_output = self.correct_grammar(final_output)
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
        weights = np.linspace(1.7, 1.2, num=len(previous_embeddings))  # Yakın cümleler daha etkili
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
        threshold = 0.85  # biraz daha insaflı başlıyoruz
        if avg_len > 15:
            threshold += 0.075
        elif avg_len < 8:
            threshold -= 0.075
        if var_len > 5:
            threshold += 0.065
        if complexity_factor > 2.5:
            threshold += 0.065
        elif complexity_factor < 1.5:
            threshold -= 0.065

        # ✨ Ekstra ufak düzen: Eğer cümlede çok az özgün lemma varsa thresholdu hafif artır
        unique_lemmas = set(token.lemma_ for token in current_doc if token.pos_ in {"NOUN", "VERB"})
        if len(unique_lemmas) < 3:
            threshold += 0.05

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
    language_model = EnhancedLanguageModel(text, n=2)
    language_model.save_model(model_file)
    language_model.log("Created and saved new model.")

num_sentences = 5
# I am going to kill you too.
input_sentence = "The victim "
input_words = tuple(token.lower() for token in input_sentence.split())
generated_text = language_model.generate_and_post_process(num_sentences=num_sentences, input_words=input_words, length=13)
language_model.log("Generated Text:\n" + generated_text)
print("Generated Text:\n" + generated_text)
def correct_grammar_t5(text: str) -> str:
    """
    FLAN-T5 ile:
      • Gramer ve noktalama düzeltmesi
      • Sadece düzeltilmiş metni döndürme
    """
    # 1. Çok net bir talimat + delimiter
    prompt = (
        "Proofread the text between the triple quotes and fix clarity, continuity and correct all grammar "
        "and punctuation errors. Do NOT include the original text or any "
        "commentary—output ONLY the corrected text.\n"
        '"""\n'
        f"{text}\n"
        '"""\n'
    )

    # 2. Tokenize et
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # 3. Beam search (sampling kapalı)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],

        max_new_tokens=500,
        num_beams=5,               # yeterli beam genişliği
        no_repeat_ngram_size=2,
        repetition_penalty=1.1,
        early_stopping=True,

        do_sample=False,           # sampling kapalı
        use_cache=True
    )

    # 4. Decode ve trim
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 5. Eğer delimter’li bir echo kaldıysa, sonrasında gelen kısmı al
    if '"""' in generated:
        corrected = generated.split('"""')[-1].strip()
    else:
        corrected = generated

    return corrected if corrected else text


corrected_text = correct_grammar_t5(generated_text)
language_model.log("\nCorrected Text:\n" + corrected_text)
print("\nCorrected Text:\n" + corrected_text)