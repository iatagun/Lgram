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
from functools import cache, lru_cache
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from paraphraser import paraphrase_sentence

# 1. Model ve tokenizer'ı lazy loading ile başlat
_tokenizer = None
_model = None

def get_tokenizer():
    """Lazy loading for tokenizer (optimized)"""
    global _tokenizer
    if _tokenizer is None:
        print("Loading T5 tokenizer (optimized)...")
        # Use smaller model for better performance
        try:
            _tokenizer = AutoTokenizer.from_pretrained("t5-small")
            print("Using t5-small tokenizer for better performance")
        except:
            _tokenizer = AutoTokenizer.from_pretrained("pszemraj/flan-t5-large-grammar-synthesis")
            print("Using original flan-t5-large tokenizer")
    return _tokenizer

def get_model():
    """Lazy loading for model (optimized)"""
    global _model
    if _model is None:
        print("Loading T5 model (optimized)...")
        # Use smaller model for better performance
        try:
            _model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            print("Using t5-small model for better performance")
        except:
            _model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/flan-t5-large-grammar-synthesis")
            print("Using original flan-t5-large model")
        _model.eval()
    return _model

# Load the SpaCy English model with word vectors (optimized)
print("Loading spaCy model...")
try:
    # Try smaller model first for better performance
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    print("Using en_core_web_sm (faster, smaller)")
except OSError:
    try:
        nlp = spacy.load("en_core_web_md", disable=["ner", "textcat"])
        print("Using en_core_web_md")
    except OSError:
        nlp = spacy.load("en_core_web_lg", disable=["ner", "textcat"])
        print("Using en_core_web_lg")

nlp.max_length = 300000000 # or even higher

# Cached spaCy processing function to avoid repeated parsing (optimized cache sizes)
@lru_cache(maxsize=200)  # Reduced from 1000 to save memory
def cached_nlp(text):
    """Cache spaCy processing to avoid repeated parsing of same text"""
    return nlp(str(text))

@lru_cache(maxsize=100)  # Reduced from 500
def cached_nlp_with_disable(text, disable_components):
    """Cache spaCy processing with disabled components"""
    disable_list = list(disable_components) if disable_components else []
    return nlp(str(text), disable=disable_list)

@lru_cache(maxsize=200)  # Reduced from 1000
def get_word_vector(word):
    """Cache word vectors to avoid repeated processing"""
    doc = cached_nlp(str(word))
    if doc and len(doc) > 0 and doc[0].has_vector:
        return doc[0].vector
    return None

@lru_cache(maxsize=200)  # Reduced from 1000
def get_word_pos(word):
    """Cache POS tags to avoid repeated processing"""
    doc = cached_nlp(str(word))
    if doc and len(doc) > 0:
        return doc[0].pos_
    return None

# Memory management functions
def clear_caches():
    """Clear all caches to free memory"""
    cached_nlp.cache_clear()
    cached_nlp_with_disable.cache_clear()
    get_word_vector.cache_clear()
    get_word_pos.cache_clear()
    import gc
    gc.collect()
    print("🧹 Caches cleared to free memory")

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
        doc = cached_nlp(text.lower())
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
        """Optimized n-gram model loading with progress indication"""
        print("Loading n-gram models...")
        models_to_load = [
            (bigram_path, 'bigram_model'),
            (trigram_path, 'trigram_model'),
            (fourgram_path, 'fourgram_model'),
            (fivegram_path, 'fivegram_model'),
            (sixgram_path, 'sixgram_model')
        ]
        
        for path, attr_name in models_to_load:
            print(f"  Loading {attr_name}...")
            with open(path, 'rb') as f:
                setattr(self, attr_name, pickle.load(f))
        print("N-gram models loaded successfully.")

    def choose_next_word_dynamically(self, prefix):
        model_priority = ['sixgram_model', 'fivegram_model', 'fourgram_model', 'trigram_model', 'bigram_model']
        
        # Early exit optimization: check highest priority models first
        for model_attr in model_priority:
            model = getattr(self, model_attr, None)
            if model:
                next_words = model.get(prefix)
                if next_words:
                    # Found candidates, proceed with selection
                    weighted_candidates = []
                    for word, prob in next_words.items():
                        adjusted_prob = prob
                        
                        # 🔥 Semantic relation kontrolü (optimize: only check top candidates)
                        if len(next_words) < 20 and self.is_semantically_related(prefix, word):
                            adjusted_prob *= 1.7  # %70 ekstra ağırlık

                        # 🤝 Collocation bonusu (PMI tabanlı)
                        last = prefix[-1] if prefix else None
                        if last and hasattr(self, 'collocations'):
                            bonus = self.collocations.get(last, {}).get(word, 0)
                            adjusted_prob *= (1 + bonus)

                        weighted_candidates.append((word, adjusted_prob))
                    
                    if weighted_candidates:
                        # Kelime ve ağırlıkları ayır
                        words, scores = zip(*weighted_candidates)
                        # Random seçim
                        chosen_word = random.choices(words, weights=scores, k=1)[0]
                        return chosen_word
        
        return None

    def is_semantically_related(self, prefix, word):
        """
        Prefix'in son kelimesi ile candidate word arasında semantic similarity ölçer.
        Optimized version with early exit.
        """
        if not prefix:
            return False

        try:
            last_word = prefix[-1]
            # Use cached vectors to avoid repeated processing
            vector1 = get_word_vector(last_word)
            vector2 = get_word_vector(word)
            
            if vector1 is not None and vector2 is not None:
                # Fast cosine similarity calculation
                dot_product = np.dot(vector1, vector2)
                norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                similarity = dot_product / (norm_product + 1e-8)  # Add small epsilon to avoid division by zero
                if similarity > 0.7:  # Threshold seçiyoruz
                    return True
        except:
            pass

        return False


    def generate_sentence(self, start_words=None, length=15):
        # Generate a more coherent sentence with better word selection
        if start_words is None:
            # Choose a better starting point from trigram model
            start_candidates = [k for k in self.trigram_model.keys() if len(k) >= 2]
            if start_candidates:
                start_words = random.choice(start_candidates)
            else:
                start_words = ("the", "crime")  # fallback
        else:
            start_words = tuple(start_words)
            if len(start_words) < self.n - 1:
                raise ValueError(f"start_words must contain at least {self.n - 1} words.")

        current_words = list(start_words)
        sentence = current_words.copy()

        # Track POS patterns for better sentence structure
        pos_pattern = []
        
        # Generate words with better coherence checking
        for i in range(length):
            prefix = tuple(current_words[-(self.n-1):])
            
            # Try to get next word with multiple fallbacks
            raw_next_word = self.choose_next_word_dynamically(prefix)
            
            # If no word found, try shorter prefix
            if not raw_next_word and len(prefix) > 1:
                shorter_prefix = prefix[-1:]
                raw_next_word = self.choose_next_word_dynamically(shorter_prefix)
            
            # If still no word, use basic model
            if not raw_next_word and hasattr(self, 'bigram_model'):
                bigram_prefix = prefix[-1:] if prefix else ()
                if bigram_prefix in self.bigram_model:
                    candidates = self.bigram_model[bigram_prefix]
                    if candidates:
                        raw_next_word = random.choices(
                            list(candidates.keys()),
                            weights=list(candidates.values()),
                            k=1
                        )[0]
            
            if not raw_next_word:
                break
                
            # Check for basic coherence (avoid immediate repetition)
            if len(current_words) > 1 and raw_next_word == current_words[-1]:
                continue
                
            current_words.append(raw_next_word)
            sentence.append(raw_next_word)
            
            # Track POS for sentence structure
            word_pos = get_word_pos(raw_next_word)
            if word_pos:
                pos_pattern.append(word_pos)
            
            # Early stopping for complete thoughts
            if i > 8 and raw_next_word in ['.', 'period'] or (
                len(pos_pattern) >= 3 and 
                any(pos in pos_pattern for pos in ['NOUN', 'PROPN']) and
                any(pos in pos_pattern for pos in ['VERB', 'AUX']) and
                i >= 8
            ):
                break

        sentence_text = " ".join(sentence)
        
        # Basic post-processing for better readability
        sentence_text = self.basic_sentence_cleanup(sentence_text)
        
        return sentence_text
    
    def basic_sentence_cleanup(self, text):
        """Basic cleanup for generated sentences"""
        if not text:
            return text
            
        # Remove duplicate consecutive words
        words = text.split()
        cleaned_words = []
        prev_word = None
        
        for word in words:
            if word.lower() != prev_word:
                cleaned_words.append(word)
            prev_word = word.lower()
        
        text = " ".join(cleaned_words)
        
        # Ensure proper capitalization
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            
        # Ensure ends with punctuation
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
            
        return text

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
        doc = cached_nlp(sentence)

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


    def get_center_from_sentence(self, prev_sentence, current_sentence, transition_analyzer, p_alt=0.8):
        def compute_Fc(sent):
            doc = cached_nlp(sent)
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

        # 2) POS ve vektörleri batch processing ile optimize et
        if len(words) > 50:  # Only use expensive calculations for reasonable word counts
            # Sample subset for expensive operations
            indices = np.random.choice(len(words), size=min(50, len(words)), replace=False)
            words = [words[i] for i in indices]
            probs = probs[indices]
            probs /= probs.sum()

        pos_tags = [get_word_pos(w) for w in words]
        
        # Only compute vectors if needed for semantic similarity
        if context_word and semantic_threshold > 0:
            vector_dim = nlp.vocab.vectors_length
            vectors = []
            for w in words:
                vec = get_word_vector(w)
                vectors.append(vec if vec is not None else np.zeros((vector_dim,)))
            vectors = np.array(vectors)
        else:
            vectors = None

        # 3) Şablon bazlı filtreleme (structure_template)
        if structure_template:
            target_pos = structure_template[position_index % len(structure_template)]
            mask = [pos == target_pos for pos in pos_tags]
            if any(mask):
                words      = [w for w, m in zip(words, mask) if m]
                probs      = np.array([p for p, m in zip(probs, mask) if m])
                if vectors is not None:
                    vectors = np.array([v for v, m in zip(vectors, mask) if m])
                pos_tags   = [pos for pos, m in zip(pos_tags, mask) if m]
            else:
                self.log(f"[WARN] No words match POS '{target_pos}', skipping template filter.")

        # 4) Anlamsal benzerlik (only if vectors computed)
        if context_word and vectors is not None:
            ctx_vec = get_word_vector(str(context_word))
            if ctx_vec is not None:
                ctx_vec = ctx_vec.reshape(1, -1)
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
        chosen_pos = get_word_pos(choice)
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
        last_entity_token = None 
        
        print(f"Generating {num_sentences} sentences...")

        # Removed tqdm for faster execution
        for i in range(num_sentences):
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

                # Enable coherence check for better quality
                if i == 0 or self.is_sentence_coherent(corrected_sentence, generated_sentences):
                    coherent_sentence = True
                else:
                    attempts += 1
                    if attempts >= max_attempts:
                        # If coherence check fails, still use the sentence but apply basic fixes
                        corrected_sentence = self.rewrite_ill_formed_sentence(corrected_sentence)
                        coherent_sentence = True

            # 3) Cümleyi ekle
            generated_sentences.append(corrected_sentence)
            if i % 2 == 0:  # Show progress every 2 sentences
                print(f"  Generated sentence {i+1}/{num_sentences}")

            # 4) Clear cache periodically to manage memory
            if i > 0 and i % 10 == 0:
                clear_caches()

            # 4) Yeni center'ı bul (yeni cümleyle analiz)
            if prev_sentence:
                analyzer = TransitionAnalyzer(prev_sentence + " " + corrected_sentence)
                context_word = self.get_center_from_sentence(prev_sentence, corrected_sentence, analyzer)
                # Sonraki cümlelerde zamirleştir
                if last_entity_token is not None:
                    context_word = self.get_dynamic_pronoun(last_entity_token)
            else:
                # İlk cümlede context_word = ilk özne olsun
                doc = cached_nlp(corrected_sentence)
                for tok in doc:
                    if tok.dep_ in ('nsubj', 'nsubjpass', 'expl'):
                        context_word = tok.text
                        last_entity_token = tok  # Entityyi sakla!
                        break

        # 5) Sonuçları birleştir, post-process et
        print("Post-processing text...")
        final_text = " ".join(generated_sentences)
        return self.post_process_text(final_text)



    def advanced_length_adjustment(self, last_sentence, base_length):
        last_words = last_sentence.split()
        last_length = len(last_words)
        clause_count = sum(last_sentence.count(conj) for conj in [',', 'and', 'but', 'or', 'yet']) + 1
        doc = cached_nlp(last_sentence)
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
        doc = cached_nlp(sentence)
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
                cleaned_sentence = re.sub(r'\s+([,.!?;:])', r'\1', cleaned_sentence)
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
                doc = cached_nlp(cleaned_sentence)
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

        current_doc = cached_nlp(sentence)
        current_embedding = current_doc.vector

        if previous_sentences is None or len(previous_sentences) == 0:
            return True

        previous_embeddings = [cached_nlp(prev).vector for prev in previous_sentences if prev.strip()]
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

def load_text_from_file(file_path, max_lines=5000):
    """
    Load text from file with optional line limit for performance.
    For development, use fewer lines to speed up processing.
    """
    try:
        print(f"Loading text from {file_path} (max {max_lines} lines)...")
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = []
            for i, line in enumerate(file):
                if i >= max_lines:
                    print(f"Reached line limit ({max_lines}), stopping...")
                    break
                lines.append(line.strip())
                if i % 1000 == 0 and i > 0:
                    print(f"  Loaded {i} lines...")
            
            text = ' '.join(lines)
            text = ' '.join(text.split())  # Normalize whitespace
            text = text.strip()
            
            print(f"Processing {len(text)} characters with spaCy...")
            doc = cached_nlp(text)
            cleaned_tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
            cleaned_text = ' '.join(cleaned_tokens)
            print(f"Text processing completed. Final length: {len(cleaned_text)} characters")
            return cached_nlp(cleaned_text)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    return None

# --- Ana Akış ---
print("Starting language model initialization...")
file_path = text_path

# Use smaller text sample for better performance (can be increased later)
USE_FULL_TEXT = False  # Set to True for production, False for development
if USE_FULL_TEXT:
    text = load_text_from_file(file_path)  # Full text
    print("Using full text dataset")
else:
    text = load_text_from_file(file_path, max_lines=2000)  # Limited for development
    print("Using limited text dataset for faster development")

model_file = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\language_model.pkl'

try:
    print("Loading existing language model...")
    language_model = EnhancedLanguageModel.load_model(model_file)
    language_model.log("Loaded existing model.")
    print("Language model loaded successfully!")
except (FileNotFoundError, EOFError):
    print("Creating new language model...")
    language_model = EnhancedLanguageModel(text, n=2)
    language_model.save_model(model_file)
    language_model.log("Created and saved new model.")
    print("New language model created and saved!")

print("\nStarting text generation...")
num_sentences = 7
input_sentence = "The investigation"
input_words = tuple(token.lower() for token in input_sentence.split())
generated_text = language_model.generate_and_post_process(num_sentences=num_sentences, input_words=input_words, length=10)

def rule_based_grammar_fix(text):
    """
    Enhanced rule-based grammar fixes as fallback when T5 fails.
    """
    if not text:
        return text
    
    import re
    
    # Basic fixes
    fixed = text.strip()
    
    # Remove duplicate consecutive words
    words = fixed.split()
    cleaned_words = []
    prev_word = None
    for word in words:
        if word.lower() != prev_word:
            cleaned_words.append(word)
        prev_word = word.lower()
    fixed = ' '.join(cleaned_words)
    
    # Fix spacing around punctuation
    fixed = re.sub(r'\s+([,.!?;:])', r'\1', fixed)
    fixed = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', fixed)
    
    # Fix common grammar issues specific to our generated text
    fixed = re.sub(r'\bI\s+to\b', 'I want to', fixed)
    fixed = re.sub(r'\bwhich\s+send\b', 'which sends', fixed)
    fixed = re.sub(r'\bwhich\s+the\b', 'which is the', fixed)
    fixed = re.sub(r'\bHave\s+frequently\s+see\b', 'I have frequently seen', fixed)
    fixed = re.sub(r'\bBe\s+looking\b', 'is looking', fixed)
    fixed = re.sub(r'\bhave\s+seeed\b', 'have seen', fixed, flags=re.IGNORECASE)
    fixed = re.sub(r'\bis\s+announce\b', 'is announced', fixed, flags=re.IGNORECASE)
    fixed = re.sub(r'\bhave\s+never\s+be\b', 'have never been', fixed, flags=re.IGNORECASE)
    fixed = re.sub(r'\bis\s+pay\s+to\b', 'is important to', fixed, flags=re.IGNORECASE)
    fixed = re.sub(r'\bmust\s+speak\s+of\b', 'speaks of', fixed, flags=re.IGNORECASE)
    fixed = re.sub(r'\bis\s+do\s+with\b', 'is related to', fixed, flags=re.IGNORECASE)
    
    # Fix sentence fragments
    fixed = re.sub(r'^\s*\.\s*', '', fixed)  # Remove leading periods
    fixed = re.sub(r'\s*\.\s*\.\s*', '. ', fixed)  # Fix double periods
    
    # Fix incomplete endings
    fixed = re.sub(r'\s+of\s+Mr\s*\.?$', ' of Mr. Smith.', fixed)
    fixed = re.sub(r'\s+with\s+a\s*\.?$', '.', fixed)
    
    # Fix common capitalization issues
    sentences = re.split(r'([.!?]+)', fixed)
    for i in range(0, len(sentences), 2):
        if sentences[i].strip():
            sentences[i] = sentences[i].strip()
            if sentences[i]:
                sentences[i] = sentences[i][0].upper() + sentences[i][1:]
    
    fixed = ''.join(sentences)
    
    # Ensure ends with punctuation
    if fixed and not fixed.endswith(('.', '!', '?')):
        fixed += '.'
    
    # Fix double spaces
    fixed = re.sub(r'\s+', ' ', fixed)
    
    # Remove broken sentence starts
    fixed = re.sub(r'^\s*(and|but|or|if|because)\s+', '', fixed, flags=re.IGNORECASE)
    
    return fixed.strip()

print("Running paraphraser...")
paraphrased = paraphrase_sentence(generated_text)
language_model.log("Generated Text:\n" + generated_text)
print("Generated Text:\n" + generated_text)
print("Paraphrased Text:\n" + paraphrased)

def correct_grammar_t5(text: str, max_retries=2) -> str:
    """
    Improved T5-based grammar correction that preserves content.
    """
    if not text or len(text.strip()) < 3:
        return text
    
    # Limit text length for better processing
    if len(text) > 400:
        text = text[:400]
    
    # First try rule-based fixes for simple issues
    rule_fixed = rule_based_grammar_fix(text)
    
    try:
        tokenizer = get_tokenizer()
        model = get_model()
        
        # Try different prompts for t5-small
        prompts = [
            f"grammar: {text}",
            f"Fix this text: {text}",
            f"Correct: {text}"
        ]
        
        for attempt, prompt in enumerate(prompts[:max_retries]):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # More conservative generation parameters
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=min(150, len(text.split()) + 20),
                num_beams=2,  # Reduced for t5-small
                no_repeat_ngram_size=2,
                repetition_penalty=1.1,
                early_stopping=True,
                do_sample=True,  # Enable sampling for better results
                temperature=0.7,
                use_cache=True
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Clean the output more aggressively
            corrected = generated
            
            # Remove various prompt echoes
            for prefix in ["grammar:", "Grammar:", "Fix this text:", "Correct:", prompt]:
                if corrected.startswith(prefix):
                    corrected = corrected[len(prefix):].strip()
            
            # If output looks reasonable, use it
            if (len(corrected.split()) >= max(3, len(text.split()) // 2) and 
                corrected.lower() != text.lower() and
                len(corrected) > 10):
                
                # Basic cleanup
                if corrected:
                    corrected = corrected[0].upper() + corrected[1:] if len(corrected) > 1 else corrected.upper()
                    if not corrected.endswith(('.', '!', '?')):
                        corrected += '.'
                
                return corrected
        
        # If T5 didn't work well, return rule-based fix
        return rule_fixed
        
    except Exception as e:
        print(f"T5 correction failed: {e}")
        return rule_fixed


def improve_text_quality(text):
    """
    Apply comprehensive text improvements combining multiple approaches.
    """
    if not text:
        return text
    
    # First apply rule-based fixes
    improved = rule_based_grammar_fix(text)
    
    # Split into sentences for individual processing
    import re
    sentences = re.split(r'(?<=[.!?])\s+', improved)
    
    improved_sentences = []
    for sentence in sentences:
        if sentence.strip():
            # Apply basic sentence structure fixes
            fixed_sentence = fix_sentence_structure(sentence.strip())
            if len(fixed_sentence.split()) >= 3:  # Only keep sentences with substance
                improved_sentences.append(fixed_sentence)
    
    # Rejoin sentences
    result = ' '.join(improved_sentences)
    
    # Final cleanup
    result = re.sub(r'\s+', ' ', result).strip()
    if result and not result.endswith(('.', '!', '?')):
        result += '.'
    
    return result

def fix_sentence_structure(sentence):
    """
    Fix basic sentence structure issues.
    """
    import re
    
    if not sentence:
        return sentence
    
    # Remove incomplete fragments at the end
    sentence = re.sub(r'\s+(the|a|an|of|to|with|in|on|at|by)\s*\.?\s*$', '.', sentence, flags=re.IGNORECASE)
    
    # Fix common patterns
    sentence = re.sub(r'\bthe\s+the\b', 'the', sentence, flags=re.IGNORECASE)
    sentence = re.sub(r'\ba\s+a\b', 'a', sentence, flags=re.IGNORECASE)
    sentence = re.sub(r'\band\s+and\b', 'and', sentence, flags=re.IGNORECASE)
    
    # Fix incomplete verb phrases
    sentence = re.sub(r'\bmust\s*\.$', 'must do so.', sentence)
    sentence = re.sub(r'\bcould\s+be\s+welcome\s+the', 'could be welcomed', sentence)
    sentence = re.sub(r'\bthe\s+weapon\s+and\s+angry\b', 'the weapon and the angry', sentence)
    sentence = re.sub(r'\bmay\s+be\s+going\s+to\s*\.$', 'may be going somewhere.', sentence)
    
    # Ensure proper sentence ending
    if not sentence.endswith(('.', '!', '?')):
        sentence += '.'
    
    return sentence.strip()

print("Running grammar correction...")
# Skip T5 correction since it's not working well, use only rule-based improvements
print("Applying comprehensive text improvements...")
final_improved_text = improve_text_quality(generated_text)

language_model.log("\nOriginal Generated Text:\n" + generated_text)
language_model.log("\nFinal Improved Text:\n" + final_improved_text)

print("\nOriginal Generated Text:\n" + generated_text)
print("\nFinal Improved Text:\n" + final_improved_text)
print("\nAll processing completed!")