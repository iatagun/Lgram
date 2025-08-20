def create_default_language_model() -> 'EnhancedLanguageModel':
    """
    Paketle gelen varsayılan text_data.txt dosyasını kullanarak model oluşturur.
    """
    return create_language_model(
        model_file=Config.BIGRAM_PATH,
        text_file=Config.TEXT_PATH
    )
class TextLoader:
    @staticmethod
    def load_text_from_file(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading text from {file_path}: {e}")
            return ""

import os
import re
import json
import time
import pickle
import random
import shutil
import datetime
from collections import defaultdict, Counter
from functools import cache, lru_cache
from typing import Optional, List, Tuple, Dict, Any
import logging

import numpy as np
import spacy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from tqdm import tqdm



class TransitionAnalyzer:
    def __init__(self, *args, **kwargs):
        pass
    def analyze(self):
        return []

logger = logging.getLogger(__name__)

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    NGRAMS_DIR = os.path.join(BASE_DIR, "ngrams")
    TEXT_PATH = os.path.join(NGRAMS_DIR, "text_data.txt")
    BIGRAM_PATH = os.path.join(NGRAMS_DIR, "bigram_model.pkl")
    TRIGRAM_PATH = os.path.join(NGRAMS_DIR, "trigram_model.pkl")
    FOURGRAM_PATH = os.path.join(NGRAMS_DIR, "fourgram_model.pkl")
    FIVEGRAM_PATH = os.path.join(NGRAMS_DIR, "fivegram_model.pkl")
    SIXGRAM_PATH = os.path.join(NGRAMS_DIR, "sixgram_model.pkl")
    CORRECTIONS_FILE = os.path.join(NGRAMS_DIR, "corrections.json")
    MODEL_FILE = os.path.join(NGRAMS_DIR, "language_model.pkl")
    COLLOCATIONS_PATH = os.path.join(NGRAMS_DIR, "collocations.pkl")
    T5_MODEL_NAME = "pszemraj/flan-t5-large-grammar-synthesis"
    SPACY_MODEL_NAME = "en_core_web_sm"
    SPACY_MAX_LENGTH = 1000000
    DEFAULT_NUM_SENTENCES = 5
    DEFAULT_SENTENCE_LENGTH = 13
    MIN_SENTENCE_LENGTH = 5
    MAX_ATTEMPTS = 5
    SEMANTIC_THRESHOLD = 0.65

class ModelInitializer:
    @staticmethod
    def initialize_t5_model():
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                Config.T5_MODEL_NAME,
                use_fast=True,
                padding_side="left"
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                Config.T5_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            model.eval()
            model = torch.compile(model)
            logger.info("T5 model initialized successfully")
            return tokenizer, model
        except Exception as e:
            logger.error(f"Failed to initialize T5 model: {e}")
            return None, None
    @staticmethod
    def initialize_spacy_model():
        try:
            nlp = spacy.load(
                Config.SPACY_MODEL_NAME, 
                disable=["ner", "textcat", "lemmatizer"]
            )
            nlp.max_length = Config.SPACY_MAX_LENGTH
            logger.info("SpaCy model initialized successfully")
            return nlp
        except Exception as e:
            logger.error(f"Failed to initialize SpaCy model: {e}")
            return None
    @staticmethod
    def load_corrections():
        try:
            with open(Config.CORRECTIONS_FILE, encoding="utf-8") as f:
                corrections = json.load(f)
            logger.info("Corrections loaded successfully")
            return corrections
        except Exception as e:
            logger.error(f"Failed to load corrections: {e}")
            return {}

try:
    TOKENIZER, T5_MODEL = ModelInitializer.initialize_t5_model()
    NLP = ModelInitializer.initialize_spacy_model()
    CORRECTIONS = ModelInitializer.load_corrections()
except Exception as e:
    logger.error(f"Model initialization failed: {e}")
    TOKENIZER, T5_MODEL, NLP, CORRECTIONS = None, None, None, {}

class EnhancedLanguageModel:
    def __init__(self, text: Optional[str] = None, n: int = 2, 
                 colloc_path: str = Config.COLLOCATIONS_PATH):
        self.n = n
        self.collocations = self._load_collocations(colloc_path)

    def _load_collocations(self, path: str) -> dict:
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"Collocations file not found: {path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading collocations: {e}")
            return {}
    def _get_log_file(self) -> str:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        logs_dir = os.path.join(Config.BASE_DIR, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        return os.path.join(logs_dir, f"daily_log_{today}.txt")
    def log(self, message: str) -> None:
        try:
            log_file = self._get_log_file()
            with open(log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            logger.error(f"Logging failed: {e}")
    def _load_ngram_models(self) -> None:
        model_paths = {
            'bigram_model': Config.BIGRAM_PATH,
            'trigram_model': Config.TRIGRAM_PATH,
            'fourgram_model': Config.FOURGRAM_PATH,
            'fivegram_model': Config.FIVEGRAM_PATH,
            'sixgram_model': Config.SIXGRAM_PATH
        }
        for model_name, path in model_paths.items():
            try:
                with open(path, 'rb') as f:
                    setattr(self, model_name, pickle.load(f))
                logger.debug(f"Loaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                setattr(self, model_name, {})
    def is_semantically_related(self, prefix: Tuple[str], word: str) -> bool:
        if not prefix or not NLP:
            return False
        try:
            last_word = prefix[-1]
            token1 = NLP(last_word)[0]
            token2 = NLP(word)[0]
            if token1.has_vector and token2.has_vector:
                similarity = 1 - cosine(token1.vector, token2.vector)
                pos_bonus = self._get_pos_bonus(token1, token2)
                dep_bonus = self._get_dependency_bonus(token1, token2)
                theme_bonus = self._get_thematic_bonus(last_word, word)
                final_score = similarity + pos_bonus + dep_bonus + theme_bonus
                return final_score > Config.SEMANTIC_THRESHOLD
        except Exception as e:
            logger.debug(f"Semantic relation check failed: {e}")
        return False
    def _get_pos_bonus(self, token1, token2) -> float:
        if ((token1.pos_ == 'NOUN' and token2.pos_ == 'VERB') or
            (token1.pos_ == 'ADJ' and token2.pos_ == 'NOUN') or
            (token1.pos_ == 'VERB' and token2.pos_ in ['ADV', 'NOUN'])):
            return 0.1
        return 0
    def _get_dependency_bonus(self, token1, token2) -> float:
        if ((token1.dep_ in ['nsubj', 'ROOT'] and token2.pos_ == 'VERB') or
            (token1.pos_ == 'VERB' and token2.dep_ in ['dobj', 'pobj'])):
            return 0.15
        return 0
    def _get_thematic_bonus(self, word1: str, word2: str) -> float:
        themes = {
            'technology': ['computer', 'internet', 'digital', 'software', 'data', 'system'],
            'nature': ['tree', 'forest', 'river', 'mountain', 'animal', 'plant'],
            'emotion': ['happy', 'sad', 'angry', 'love', 'fear', 'joy'],
            'time': ['yesterday', 'tomorrow', 'morning', 'night', 'hour', 'minute'],
            'business': ['company', 'market', 'profit', 'customer', 'business', 'work']
        }
        for theme, words in themes.items():
            if word1.lower() in words and word2.lower() in words:
                return 0.2
        return 0
    def choose_next_word_dynamically(self, prefix: Tuple[str]) -> Optional[str]:
        model_priority = ['sixgram_model', 'fivegram_model', 'fourgram_model', 
                        'trigram_model', 'bigram_model']
        candidates = []
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
                if self.is_semantically_related(prefix, word):
                    adjusted_prob *= 1.7
                last = prefix[-1] if prefix else None
                if last and self.collocations:
                    bonus = self.collocations.get(last, {}).get(word, 0)
                    adjusted_prob *= (1 + bonus)
                weighted_candidates.append((word, adjusted_prob))
        if not weighted_candidates:
            return None
        words, scores = zip(*weighted_candidates)
        chosen_word = random.choices(words, weights=scores, k=1)[0]
        return chosen_word
    def generate_sentence(self, start_words: Optional[List[str]] = None, 
                        base_length: int = Config.DEFAULT_SENTENCE_LENGTH) -> str:
        target_length = self._get_dynamic_sentence_length(base_length)
        if start_words is None:
            start_words = random.choice(list(self.trigram_model.keys()))
        else:
            start_words = tuple(start_words)
        current_words = list(start_words)
        sentence = current_words.copy()
        min_length = Config.MIN_SENTENCE_LENGTH
        for i in range(target_length):
            prefix = tuple(current_words[-(self.n-1):])
            raw_next_word = self.choose_next_word_dynamically(prefix)
            if not raw_next_word:
                break
            current_words.append(raw_next_word)
            sentence.append(raw_next_word)
            if (len(sentence) >= min_length and 
                len(sentence) >= target_length * 0.7 and
                self._should_end_sentence(sentence)):
                break
        return " ".join(sentence)
    def _should_end_sentence(self, sentence_words: List[str]) -> bool:
        if len(sentence_words) < Config.MIN_SENTENCE_LENGTH:
            return False
        if not NLP:
            return random.random() < 0.4
        last_word = sentence_words[-1]
        doc = NLP(last_word)
        if doc and doc[0].pos_ in ['NOUN', 'PRON', 'ADJ']:
            return random.random() < 0.4
        return False
    def _get_dynamic_sentence_length(self, base_length: int, 
                                   context_sentences: Optional[List[str]] = None) -> int:
        variations = [
            (0.25, range(5, 8)),
            (0.55, range(8, 13)),
            (0.80, range(13, 17)),
            (1.0, range(17, 22))
        ]
        rand = random.random()
        for prob, length_range in variations:
            if rand <= prob:
                return random.choice(length_range)
        return base_length
    def correct_grammar(self, text: str) -> str:
        if not CORRECTIONS:
            return text
        for wrong, right in CORRECTIONS.items():
            pattern = re.compile(rf"\b{re.escape(wrong)}\b", flags=re.IGNORECASE)
            text = pattern.sub(right, text)
        return text
    def correct_grammar_t5(self, text: str) -> str:
        if not all([TOKENIZER, T5_MODEL]):
            logger.warning("T5 model not available, using rule-based correction")
            return self.correct_grammar(text)
        try:
            prompt = f"grammar, coherence, ambiguity, story, novel: {text}"
            inputs = TOKENIZER(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            outputs = T5_MODEL.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=min(200, len(text.split()) * 2),
                num_beams=2,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True,
                do_sample=False,
                temperature=0.1,
                use_cache=True
            )
            generated = TOKENIZER.decode(outputs[0], skip_special_tokens=True).strip()
            corrected = self._clean_t5_output(generated, text, prompt)
            return corrected
        except Exception as e:
            logger.error(f"T5 grammar correction failed: {e}")
            return self.correct_grammar(text)
    def _clean_t5_output(self, generated: str, original: str, prompt: str) -> str:
        corrected = generated
        prompt_prefixes = [
            "grammar, coherence, ambiguity:",
            "grammar, coherence, ambiguity, story:",
            "grammar, coherence, ambiguity, story, novel:",
            "grammar:",
            "Grammar:",
            "correct:",
            "Correct:",
            prompt,
            original
        ]
        for prefix in prompt_prefixes:
            if corrected.startswith(prefix):
                corrected = corrected[len(prefix):].strip()
        corrected = corrected.replace('"""', '').strip()
        corrected = re.sub(r'^[\"\'\`\n\r\s]+', '', corrected)
        corrected = re.sub(r'[\"\'\`\n\r\s]+$', '', corrected)
        prompt_words = ["grammar", "punctuation", "correct", "coherence", "ambiguity"]
        for word in prompt_words:
            corrected = re.sub(rf'^\b{word}\b\s*:?', '', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\n+', ' ', corrected)
        corrected = re.sub(r'\s+', ' ', corrected)
        corrected = corrected.strip()
        if (not corrected or 
            len(corrected.split()) < max(2, len(original.split()) // 3) or
            corrected.lower() == original.lower()):
            return original
        if corrected and not corrected.endswith(('.', '!', '?')):
            corrected += '.'
        if corrected:
            corrected = corrected[0].upper() + corrected[1:] if len(corrected) > 1 else corrected.upper()
        return corrected
    def generate_text(self, num_sentences: int = Config.DEFAULT_NUM_SENTENCES,
                    input_words: Optional[List[str]] = None,
                    length: int = Config.DEFAULT_SENTENCE_LENGTH,
                    use_progress_bar: bool = False) -> str:
        generated_sentences = []
        context_word = None
        last_entity_token = None
        pbar = range(num_sentences)
        if use_progress_bar:
            try:
                terminal_width = shutil.get_terminal_size().columns
                bar_width = min(terminal_width - 20, 120)
                pbar = tqdm(
                    range(num_sentences),
                    desc="✨ Text Generation",
                    unit=" sent",
                    ncols=bar_width,
                    colour='cyan'
                )
            except Exception:
                pbar = range(num_sentences)
        for _ in pbar:
            # Burada örnek bir cümle üretimi yapılabilir, örnek olarak:
            raw_sentence = self.generate_sentence(
                start_words=input_words, 
                base_length=length
            )
            corrected_sentence = self.correct_grammar(raw_sentence)
            if corrected_sentence and not corrected_sentence.endswith(('.', '!', '?')):
                corrected_sentence += '.'
            generated_sentences.append(corrected_sentence)
        final_text = " ".join(generated_sentences)
        return self._post_process_text(final_text)
    def generate_text_with_centering(self, num_sentences: int = 5,
                                    input_words: Optional[List[str]] = None,
                                    length: int = 13) -> str:
        """
        Centering Theory tabanlı metin üretimi. Her cümle için bir önceki cümleye göre merkez belirler.
        """
        generated_sentences = []
        prev_sentence = None
        for i in range(num_sentences):
            if i == 0 and input_words:
                sentence = self.generate_sentence(input_words, length)
            else:
                sentence = self.generate_sentence(base_length=length)
            # Centering işlemleri burada yapılabilir (örnek, bir transition_analyzer ile)
            generated_sentences.append(sentence)
            prev_sentence = sentence
        final_text = " ".join(generated_sentences)
        return self._post_process_text(final_text)
    def _get_dynamic_pronoun(self, token) -> str:
        if not hasattr(token, 'tag_'):
            return "it"
        if token.tag_ in ("NNS", "NNPS") or (hasattr(token, 'morph') and token.morph.get("Number") == ["Plur"]):
            return "they"
        if (hasattr(token, 'ent_type_') and token.ent_type_ == "PERSON") or \
           (token.pos_ == "PROPN" and hasattr(token, 'morph') and token.morph.get("Gender")):
            gender = token.morph.get("Gender") if hasattr(token, 'morph') else []
            if "Masc" in gender:
                return "he"
            elif "Fem" in gender:
                return "she"
            else:
                return "they"
        if hasattr(token, 'ent_type_') and token.ent_type_ == "ORG":
            return "it"
        return "it"

    def _post_process_text(self, text: str) -> str:
        if not text:
            return ""
        text = (text.replace('â€ ™', "'")
                   .replace('â€"', '"')
                   .replace('â€™', "'")
                   .replace('â€œ', '"')
                   .replace('â€', '"'))
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        cleaned_sentences = []
        buffer_sentence = ""
        min_words_for_sentence = 4
        for sentence in sentences:
            cleaned_sentence = sentence.strip()
            if cleaned_sentence:
                cleaned_sentence = cleaned_sentence[0].upper() + cleaned_sentence[1:]
                cleaned_sentence = re.sub(r'\s+([,.!?])', r'\1', cleaned_sentence)
                cleaned_sentence = re.sub(r'([,.!?])\s+', r'\1 ', cleaned_sentence)
                cleaned_sentence = re.sub(r'\s{2,}', ' ', cleaned_sentence)
                cleaned_sentence = cleaned_sentence.rstrip(",")
                cleaned_sentence = re.sub(r'([.!?])\1+', r'\1', cleaned_sentence)
                cleaned_sentence = re.sub(r'^(And|But|Or)\b,?\s+', '', cleaned_sentence, flags=re.IGNORECASE)
                cleaned_sentence = re.sub(r'"(.*?)"', r'"\1"', cleaned_sentence)
                if re.search(r'\b(and|but|or|if|because)\.$', cleaned_sentence, flags=re.IGNORECASE):
                    cleaned_sentence = re.sub(r'\b(and|but|or|if|because)\.$', '', cleaned_sentence, flags=re.IGNORECASE).strip()
                    if not cleaned_sentence.endswith('.'):
                        cleaned_sentence += '.'
                if NLP:
                    doc = NLP(cleaned_sentence)
                    for entity in doc.ents:
                        if entity.label_ in ["PERSON", "ORG", "GPE"]:
                            cleaned_sentence = re.sub(r'\b' + re.escape(entity.text.lower()) + r'\b', entity.text, cleaned_sentence)
                words = cleaned_sentence.split()
                deduplicated_words = []
                previous_word = None
                for word in words:
                    if word.lower() != previous_word:
                        deduplicated_words.append(word)
                    previous_word = word.lower()
                cleaned_sentence = ' '.join(deduplicated_words)
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
        final_output = ' '.join(cleaned_sentences).strip()
        if final_output and not final_output.endswith('.'):
            final_output += '.'
        return final_output

    def save_model(self, filename: str, compress: bool = False) -> None:
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
            logger.info(f"Model saved successfully to {filename}")
        except Exception as e:
            logger.error(f"Error saving model to {filename}: {e}")

    @classmethod
    def load_model(cls, filename: str, compress: bool = False) -> 'EnhancedLanguageModel':
        try:
            if compress:
                import gzip
                with gzip.open(filename, 'rb') as f:
                    model, total_counts = pickle.load(f)
            else:
                with open(filename, 'rb') as f:
                    model, total_counts = pickle.load(f)
            instance = cls("")
            instance.model = defaultdict(lambda: defaultdict(int), model)
            instance.total_counts = total_counts
            logger.info(f"Model loaded successfully from {filename}")
            return instance
        except Exception as e:
            logger.error(f"Error loading model from {filename}: {e}")
            return None

# --- create_language_model fonksiyonu modül seviyesinde, doğru girintiyle tanımlanıyor ---
def create_language_model(model_file: str = Config.BIGRAM_PATH, text_file: str = Config.TEXT_PATH) -> 'EnhancedLanguageModel':
    """
    Model dosyası varsa yükler, yoksa text_file'dan yeni model oluşturur ve kaydeder.
    """
    try:
        language_model = EnhancedLanguageModel.load_model(model_file)
        if language_model:
            logger.info("Loaded existing model")
            return language_model
    except (FileNotFoundError, EOFError, AttributeError):
        pass
    text = TextLoader.load_text_from_file(text_file)
    if not text:
        raise RuntimeError(f"Could not load text from {text_file}")
    language_model = EnhancedLanguageModel(text, n=2)
    language_model.save_model(model_file)
    logger.info("Created and saved new model")
    return language_model


__all__ = [
    'EnhancedLanguageModel',
    'Config',
    'ModelInitializer',
    'TextLoader',
    'create_language_model',
    'create_default_language_model',
]
