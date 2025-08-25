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

# Try relative import first, fallback to absolute import
try:
    from .centering_theory import EnhancedCenteringTheory, TransitionType
    from .transition_analyzer import TransitionAnalyzer
except ImportError:
    try:
        from centering_theory import EnhancedCenteringTheory, TransitionType
        from transition_analyzer import TransitionAnalyzer
    except ImportError:
        # Create dummy classes if not available
        class TransitionType:
            pass
        class EnhancedCenteringTheory:
            def __init__(self, *args, **kwargs):
                pass
        class TransitionAnalyzer:
            def __init__(self, *args, **kwargs):
                pass
            def analyze(self):
                return []

# Configure logging
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for Enhanced Language Model"""
    
    # Model paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # ngrams klasörü proje kökünde
    NGRAMS_DIR = os.path.abspath(os.path.join(BASE_DIR, 'ngrams'))

    TEXT_PATH = os.path.join(NGRAMS_DIR, "text_data.txt")
    BIGRAM_PATH = os.path.join(NGRAMS_DIR, "bigram_model.pkl")
    TRIGRAM_PATH = os.path.join(NGRAMS_DIR, "trigram_model.pkl")
    FOURGRAM_PATH = os.path.join(NGRAMS_DIR, "fourgram_model.pkl")
    FIVEGRAM_PATH = os.path.join(NGRAMS_DIR, "fivegram_model.pkl")
    SIXGRAM_PATH = os.path.join(NGRAMS_DIR, "sixgram_model.pkl")
    CORRECTIONS_FILE = os.path.join(NGRAMS_DIR, "corrections.json")
    MODEL_FILE = os.path.join(NGRAMS_DIR, "language_model.pkl")
    COLLOCATIONS_PATH = os.path.join(NGRAMS_DIR, "collocations.pkl")
    
    # Model parameters
    T5_MODEL_NAME = "pszemraj/flan-t5-large-grammar-synthesis"
    SPACY_MODEL_NAME = "en_core_web_sm"
    SPACY_MAX_LENGTH = 1000000
    
    # Generation parameters
    DEFAULT_NUM_SENTENCES = 5
    DEFAULT_SENTENCE_LENGTH = 13
    MIN_SENTENCE_LENGTH = 5
    MAX_ATTEMPTS = 5
    SEMANTIC_THRESHOLD = 0.65


class ModelInitializer:
    """Handles model initialization and loading"""
    
    @staticmethod
    def initialize_t5_model():
        """Initialize T5 model for grammar correction"""
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
        """Initialize SpaCy model"""
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
        """Load corrections dictionary"""
        try:
            with open(Config.CORRECTIONS_FILE, encoding="utf-8") as f:
                corrections = json.load(f)
            logger.info("Corrections loaded successfully")
            return corrections
        except Exception as e:
            logger.error(f"Failed to load corrections: {e}")
            return {}


# Initialize models globally (Django compatible)
try:
    TOKENIZER, T5_MODEL = ModelInitializer.initialize_t5_model()
    NLP = ModelInitializer.initialize_spacy_model()
    CORRECTIONS = ModelInitializer.load_corrections()
except Exception as e:
    logger.error(f"Model initialization failed: {e}")
    TOKENIZER, T5_MODEL, NLP, CORRECTIONS = None, None, None, {}


class EnhancedLanguageModel:
    """Enhanced Language Model for text generation with coherence and grammar correction"""
    
    def __init__(self, text: Optional[str] = None, n: int = 2, 
                 colloc_path: str = Config.COLLOCATIONS_PATH):
        """
        Initialize the Enhanced Language Model
        
        Args:
            text: Training text (optional)
            n: N-gram size
            colloc_path: Path to collocations file
        """
        self.n = n
        self.collocations = self._load_collocations(colloc_path)
        
        # Initialize T5 correction cache
        self._correction_cache = {}
        self._cache_max_size = 1000
        
        if text:
            self.model, self.total_counts = self.build_model(text)
        else:
            self.model, self.total_counts = {}, {}
        
        self._load_ngram_models()
        
        # Initialize centering theory
        if NLP:
            self.centering = EnhancedCenteringTheory(NLP)
        else:
            self.centering = None
        
        # Ensure models are available
        if not all([NLP, TOKENIZER, T5_MODEL]):
            raise RuntimeError("Required models not initialized")
    
    @cache
    def _load_collocations(self, path: str) -> Dict:
        """Load collocations from file with caching"""
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
        """Get log file path for current date"""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        logs_dir = os.path.join(Config.BASE_DIR, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        return os.path.join(logs_dir, f"daily_log_{today}.txt")
    
    def log(self, message: str) -> None:
        """Log message to file"""
        try:
            log_file = self._get_log_file()
            with open(log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            logger.error(f"Logging failed: {e}")
    
    def build_model(self, text: str) -> Tuple[Dict, Dict]:
        """Build n-gram model from text"""
        if not NLP:
            raise RuntimeError("SpaCy model not available")
        
        model = defaultdict(lambda: defaultdict(int))
        doc = NLP(text.lower())
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
    
    def _load_ngram_models(self) -> None:
        """Load all n-gram models"""
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
        """Check semantic relationship between prefix and word"""
        if not prefix or not NLP:
            return False
        
        try:
            last_word = prefix[-1]
            token1 = NLP(last_word)[0]
            token2 = NLP(word)[0]
            
            if token1.has_vector and token2.has_vector:
                similarity = 1 - cosine(token1.vector, token2.vector)
                
                # POS compatibility bonus
                pos_bonus = self._get_pos_bonus(token1, token2)
                
                # Dependency relation bonus
                dep_bonus = self._get_dependency_bonus(token1, token2)
                
                # Thematic consistency bonus
                theme_bonus = self._get_thematic_bonus(last_word, word)
                
                final_score = similarity + pos_bonus + dep_bonus + theme_bonus
                
                return final_score > Config.SEMANTIC_THRESHOLD
                
        except Exception as e:
            logger.debug(f"Semantic relation check failed: {e}")
        
        return False
    
    def _get_pos_bonus(self, token1, token2) -> float:
        """Calculate POS compatibility bonus"""
        if ((token1.pos_ == 'NOUN' and token2.pos_ == 'VERB') or
            (token1.pos_ == 'ADJ' and token2.pos_ == 'NOUN') or
            (token1.pos_ == 'VERB' and token2.pos_ in ['ADV', 'NOUN'])):
            return 0.1
        return 0
    
    def _get_dependency_bonus(self, token1, token2) -> float:
        """Calculate dependency relation bonus"""
        if ((token1.dep_ in ['nsubj', 'ROOT'] and token2.pos_ == 'VERB') or
            (token1.pos_ == 'VERB' and token2.dep_ in ['dobj', 'pobj'])):
            return 0.15
        return 0
    
    def _get_thematic_bonus(self, word1: str, word2: str) -> float:
        """Calculate thematic consistency bonus"""
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
        """Choose next word based on multiple n-gram models"""
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
                
                # Semantic relation bonus
                if self.is_semantically_related(prefix, word):
                    adjusted_prob *= 1.7
                
                # Collocation bonus
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
        """Generate a single sentence"""
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
        """Determine if sentence should end"""
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
        """Get dynamic sentence length with variation"""
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
        """Apply rule-based grammar corrections"""
        if not CORRECTIONS:
            return text
        
        for wrong, right in CORRECTIONS.items():
            pattern = re.compile(rf"\b{re.escape(wrong)}\b", flags=re.IGNORECASE)
            text = pattern.sub(right, text)
        
        return text
    
    def correct_grammar_t5(self, text: str) -> str:
        """Correct grammar using T5 model with optimized parameters and caching"""
        if not all([TOKENIZER, T5_MODEL]):
            logger.warning("T5 model not available, using rule-based correction")
            return self.correct_grammar(text)
        
        # Early return for very short or empty text
        if not text or len(text.strip()) < 5:
            return text
            
        # Check cache first
        text_hash = hash(text.strip().lower())
        if text_hash in self._correction_cache:
            logger.debug("Using cached T5 correction")
            return self._correction_cache[text_hash]
        
        try:
            prompt = f"fluency, coherence, semantic accuracy, clarity, sentence completeness, cohesion, style control, contextual relevance, respectful, appropriate, ethical: {text}"
            
            inputs = TOKENIZER(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Optimized generation parameters for better quality
            with torch.no_grad():
                outputs = T5_MODEL.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=min(300, len(text.split()) * 3),  # Increased for better output
                    num_beams=4,  # Increased beam size for better quality
                    no_repeat_ngram_size=2,  # Reduced for more natural text
                    repetition_penalty=1.1,  # Reduced for better flow
                    early_stopping=True,
                    do_sample=True,  # Enable sampling for variety
                    temperature=0.7,  # Slightly increased for more natural output
                    top_k=50,  # Add top-k filtering
                    top_p=0.95,  # Add nucleus sampling
                    use_cache=True,
                    pad_token_id=TOKENIZER.pad_token_id
                )
            
            generated = TOKENIZER.decode(outputs[0], skip_special_tokens=True).strip()
            corrected = self._clean_t5_output(generated, text, prompt)
            
            # Additional validation and fallback
            if self._is_valid_correction(corrected, text):
                # Cache the successful correction
                self._cache_correction(text_hash, corrected)
                return corrected
            else:
                logger.warning("T5 correction validation failed, using rule-based correction")
                fallback = self.correct_grammar(text)
                self._cache_correction(text_hash, fallback)
                return fallback
            
        except Exception as e:
            logger.error(f"T5 grammar correction failed: {e}")
            fallback = self.correct_grammar(text)
            self._cache_correction(text_hash, fallback)
            return fallback
    
    def _cache_correction(self, text_hash: int, correction: str) -> None:
        """Cache correction result with size management"""
        if len(self._correction_cache) >= self._cache_max_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._correction_cache))
            del self._correction_cache[oldest_key]
        
        self._correction_cache[text_hash] = correction
    
    def _is_valid_correction(self, corrected: str, original: str) -> bool:
        """Validate if the correction is acceptable"""
        if not corrected or corrected.strip() == "":
            return False
            
        # Check minimum word count (should be at least 1/3 of original)
        original_words = len(original.split())
        corrected_words = len(corrected.split())
        
        if corrected_words < max(3, original_words // 3):
            return False
            
        # Check if output is too similar to input (no improvement)
        if corrected.lower().strip() == original.lower().strip():
            return False
            
        # Check for repeated patterns that indicate model failure
        words = corrected.split()
        if len(words) > 4:
            # Check for excessive repetition
            for i in range(len(words) - 2):
                if words[i] == words[i+1] == words[i+2]:
                    return False
                    
        # Check if correction contains prompt remnants
        prompt_indicators = [
            "fluency", "coherence", "semantic", "accuracy", 
            "clarity", "completeness", "cohesion", "style",
            "contextual", "relevance", "respectful", "appropriate", "ethical"
        ]
        
        corrected_lower = corrected.lower()
        prompt_word_count = sum(1 for word in prompt_indicators if word in corrected_lower)
        
        # If more than 2 prompt words appear, it's likely contaminated
        if prompt_word_count > 2:
            return False
            
        return True
    
    def _clean_t5_output(self, generated: str, original: str, prompt: str) -> str:
        """Clean T5 model output with enhanced cleaning"""
        corrected = generated.strip()
        
        # Step 1: Remove the full prompt if it appears at the beginning
        if corrected.startswith(prompt):
            corrected = corrected[len(prompt):].strip()
        
        # Step 2: Remove prompt patterns with more aggressive matching
        prompt_patterns = [
            r'^.*?fluency[^:]*:',
            r'^.*?coherence[^:]*:',
            r'^.*?semantic[^:]*:',
            r'^.*?clarity[^:]*:',
            r'^.*?sentence[^:]*:',
            r'^.*?cohesion[^:]*:',
            r'^.*?style[^:]*:',
            r'^.*?contextual?[^:]*:',
            r'^.*?respectful[^:]*:',
            r'^.*?appropriate[^:]*:',
            r'^.*?ethical[^:]*:',
            r'^.*?grammar[^:]*:',
            r'^.*?correct[^:]*:',
        ]
        
        for pattern in prompt_patterns:
            corrected = re.sub(pattern, '', corrected, flags=re.IGNORECASE | re.DOTALL).strip()
        
        # Step 3: Remove any colon-separated prefixes more aggressively
        corrected = re.sub(r'^[^.!?]*?:', '', corrected).strip()
        
        # Step 4: Remove the exact long prompt prefix
        long_prefix = "fluency, coherence, semantic accuracy, clarity, sentence completeness, cohesion, style control, contextual relevance, respectful, appropriate, ethical"
        if corrected.lower().startswith(long_prefix.lower()):
            corrected = corrected[len(long_prefix):].strip()
            if corrected.startswith(':'):
                corrected = corrected[1:].strip()
        
        # Step 5: Clean delimiters and unwanted characters
        corrected = corrected.replace('"""', '').replace('``', '').replace('`', '')
        corrected = re.sub(r'^[\"\'\`\n\r\s\-\—\–]+', '', corrected)
        corrected = re.sub(r'[\"\'\`\n\r\s]+$', '', corrected)
        
        # Step 6: Remove individual prompt words at the beginning (more comprehensive)
        prompt_words = [
            "fluency", "coherence", "semantic", "accuracy", "clarity", 
            "sentence", "completeness", "cohesion", "style", "control", 
            "contextual", "relevance", "respectful", "appropriate", 
            "ethical", "grammar", "punctuation", "correct"
        ]
        
        # Remove prompt words iteratively until none remain
        changed = True
        while changed:
            old_corrected = corrected
            for word in prompt_words:
                corrected = re.sub(rf'^\b{word}\b\s*[,:;]?\s*', '', corrected, flags=re.IGNORECASE)
            changed = (old_corrected != corrected)
        
        # Step 7: Clean whitespace and formatting
        corrected = re.sub(r'\n+', ' ', corrected)
        corrected = re.sub(r'\s{2,}', ' ', corrected)
        corrected = corrected.strip()
        
        # Step 8: Final cleanup - remove leading punctuation and comma sequences
        corrected = re.sub(r'^[,\s\-\—\–:;]+', '', corrected).strip()
        corrected = re.sub(r'^(and|but|or|so|then|also|additionally|furthermore|moreover|however|therefore|thus|hence|consequently)\b[,\s]*', '', corrected, flags=re.IGNORECASE).strip()
        
        # Step 9: Handle incomplete sentences at the start
        if corrected and not corrected[0].isupper():
            corrected = corrected[0].upper() + corrected[1:] if len(corrected) > 1 else corrected.upper()
            
        # Step 10: Final validation and formatting
        if not corrected or len(corrected.split()) < 2:
            return original
            
        # Add proper ending punctuation if missing
        if corrected and not corrected.endswith(('.', '!', '?')):
            # Check if it's a complete sentence
            if len(corrected.split()) >= 3:
                corrected += '.'
        
        return corrected
    
    def generate_text(self, num_sentences: int = Config.DEFAULT_NUM_SENTENCES,
                     input_words: Optional[List[str]] = None,
                     length: int = Config.DEFAULT_SENTENCE_LENGTH,
                     use_progress_bar: bool = False) -> str:
        """
        Generate coherent text with multiple sentences
        
        Args:
            num_sentences: Number of sentences to generate
            input_words: Starting words for first sentence
            length: Base length for sentences
            use_progress_bar: Whether to show progress bar
            
        Returns:
            Generated and post-processed text
        """
        generated_sentences = []
        context_word = None
        last_entity_token = None
        
        # Progress bar setup
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
            except:
                pbar = range(num_sentences)
        else:
            pbar = range(num_sentences)
        
        for i in pbar:
            attempts = 0
            coherent_sentence = False
            corrected_sentence = ""
            prev_sentence = generated_sentences[-1] if generated_sentences else None
            
            # Determine starting words
            if i == 0 and input_words:
                start_words = input_words
            elif context_word:
                start_words = [context_word]
            else:
                start_words = None
            
            # Generate sentence with retry logic
            while attempts < Config.MAX_ATTEMPTS and not coherent_sentence:
                attempts += 1
                
                raw_sentence = self.generate_sentence(
                    start_words=start_words, 
                    base_length=length
                )
                
                # Ensure sentence starts with context word
                if start_words and not raw_sentence.lower().startswith(str(start_words[0]).lower()):
                    raw_sentence = str(start_words[0]) + " " + raw_sentence
                
                corrected_sentence = self.correct_grammar(raw_sentence)
                
                if corrected_sentence and not corrected_sentence.endswith(('.', '!', '?')):
                    corrected_sentence += '.'
                
                coherent_sentence = True
            
            generated_sentences.append(corrected_sentence)
            
            # Find next context word
            if prev_sentence and NLP:
                try:
                    analyzer = TransitionAnalyzer(prev_sentence + " " + corrected_sentence)
                    context_word = self._get_center_from_sentence(
                        prev_sentence, corrected_sentence, analyzer
                    )
                    
                    if last_entity_token:
                        context_word = self._get_dynamic_pronoun(last_entity_token)
                        
                except Exception as e:
                    logger.debug(f"Context analysis failed: {e}")
                    context_word = None
            else:
                # First sentence - find subject
                if NLP and corrected_sentence:
                    doc = NLP(corrected_sentence)
                    for tok in doc:
                        if tok.dep_ in ('nsubj', 'nsubjpass', 'expl'):
                            context_word = tok.text
                            last_entity_token = tok
                            break
        
        final_text = " ".join(generated_sentences)
        return self._post_process_text(final_text)
    
    def generate_text_with_centering(self, num_sentences: int = 5,
                                   input_words: Optional[List[str]] = None,
                                   length: int = 13) -> str:
        """Generate text using enhanced centering theory"""
        generated_sentences = []
        
        for i in range(num_sentences):
            if i == 0 and input_words:
                # First sentence with input words
                sentence = self.generate_sentence(input_words, length)
            else:
                # Use centering theory for subsequent sentences
                center = self.centering.get_coherent_next_center() if self.centering else None
                start_words = [center] if center else None
                sentence = self.generate_sentence(start_words, length)
            
            # Update centering state
            if self.centering:
                self.centering.update_discourse(sentence)
            
            generated_sentences.append(sentence)
        
        final_text = " ".join(generated_sentences)
        
        # Evaluate coherence
        if self.centering:
            coherence_info = self.centering.evaluate_coherence(generated_sentences)
            logger.info(f"Coherence score: {coherence_info['coherence_score']:.3f}")
            logger.info(f"Transitions: {coherence_info['transition_distribution']}")
        
        return self._post_process_text(final_text)
    
    def _get_center_from_sentence(self, prev_sentence: str, current_sentence: str, 
                                 transition_analyzer, p_alt: float = 0.8) -> Optional[str]:
        """Get center word from sentence using transition analysis"""
        if not NLP:
            return None
        
        def compute_Fc(sent):
            doc = NLP(sent)
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
        
        try:
            Cf_prev = compute_Fc(prev_sentence)
            Cf_curr = compute_Fc(current_sentence)
            Cb = Cf_prev[0] if Cf_prev and Cf_prev[0] in Cf_curr else None
            
            results = transition_analyzer.analyze()
            for res in results:
                tr = res.get('transition')
                if tr == "Center Continuation (CON)":
                    if Cf_curr and random.random() < p_alt and len(Cf_curr) > 1:
                        return Cf_curr[1]
                    return Cb
                if tr in ("Smooth Shift (SSH)", "Rough Shift (RSH)"):
                    if Cf_curr and random.random() < p_alt and len(Cf_curr) > 1:
                        return Cf_curr[1]
                    return Cf_curr[0] if Cf_curr else None
            
            # Fallback
            if Cb:
                return Cb if random.random() > p_alt else (Cf_curr[1] if len(Cf_curr) > 1 else Cb)
            
            return (Cf_curr[1] if len(Cf_curr) > 1 and random.random() < p_alt else
                   (Cf_curr[0] if Cf_curr else None))
               
        except Exception as e:
            logger.debug(f"Center finding failed: {e}")
            return None
    
    def _get_dynamic_pronoun(self, token) -> str:
        """Get appropriate pronoun for token"""
        if not hasattr(token, 'tag_'):
            return "it"
        
        if token.tag_ in ("NNS", "NNPS") or (hasattr(token, 'morph') and 
                                           token.morph.get("Number") == ["Plur"]):
            return "they"
        
        if (hasattr(token, 'ent_type_') and token.ent_type_ == "PERSON" or 
            (token.pos_ == "PROPN" and hasattr(token, 'morph') and token.morph.get("Gender"))):
            
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
        """Post-process generated text for better quality"""
        if not text:
            return ""
        
        # Fix encoding issues
        text = (text.replace('â€ ™', "'")
                   .replace('â€"', '"')
                   .replace('â€™', "'")
                   .replace('â€œ', '"')
                   .replace('â€', '"'))
        
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        cleaned_sentences = []
        buffer_sentence = ""
        min_words_for_sentence = 4
        
        for sentence in sentences:
            cleaned_sentence = sentence.strip()
            
            if cleaned_sentence:
                # Capitalize first letter
                cleaned_sentence = cleaned_sentence[0].upper() + cleaned_sentence[1:]
                
                # Fix spacing and punctuation
                cleaned_sentence = re.sub(r'\s+([,.!?])', r'\1', cleaned_sentence)
                cleaned_sentence = re.sub(r'([,.!?])\s+', r'\1 ', cleaned_sentence)
                cleaned_sentence = re.sub(r'\s{2,}', ' ', cleaned_sentence)
                cleaned_sentence = cleaned_sentence.rstrip(",")
                cleaned_sentence = re.sub(r'([.!?])\1+', r'\1', cleaned_sentence)
                
                # Remove conjunctions at beginning
                cleaned_sentence = re.sub(r'^(And|But|Or)\b,?\s+', '', 
                                        cleaned_sentence, flags=re.IGNORECASE)
                
                # Fix quotes
                cleaned_sentence = re.sub(r'"(.*?)"', r'"\1"', cleaned_sentence)
                
                # Handle sentence-ending conjunctions
                if re.search(r'\b(and|but|or|if|because)\.$', cleaned_sentence, flags=re.IGNORECASE):
                    cleaned_sentence = re.sub(r'\b(and|but|or|if|because)\.$', '', 
                                            cleaned_sentence, flags=re.IGNORECASE).strip()
                    if not cleaned_sentence.endswith('.'):
                        cleaned_sentence += '.'
                
                # Named entity fixing with SpaCy
                if NLP:
                    doc = NLP(cleaned_sentence)
                    for entity in doc.ents:
                        if entity.label_ in ["PERSON", "ORG", "GPE"]:
                            cleaned_sentence = re.sub(r'\b' + re.escape(entity.text.lower()) + r'\b',
                                                    entity.text, cleaned_sentence)
                
                # Remove duplicate words
                words = cleaned_sentence.split()
                deduplicated_words = []
                previous_word = None
                
                for word in words:
                    if word.lower() != previous_word:
                        deduplicated_words.append(word)
                    previous_word = word.lower()
                
                cleaned_sentence = ' '.join(deduplicated_words)
                
                # Handle short sentences
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
        
        # Final output
        final_output = ' '.join(cleaned_sentences).strip()
        if final_output and not final_output.endswith('.'):
            final_output += '.'
        
        return final_output
    
    def save_model(self, filename: str, compress: bool = False) -> None:
        """Save model to file"""
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
            self.log(f"Model saved successfully to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving model to {filename}: {e}")
            self.log(f"Error saving model to {filename}: {e}")
    
    @classmethod
    def load_model(cls, filename: str) -> 'EnhancedLanguageModel':
        """Load model from file"""
        try:
            with open(filename, 'rb') as f:
                model, total_counts = pickle.load(f)
            
            instance = cls()
            instance.model = defaultdict(lambda: defaultdict(int), model)
            instance.total_counts = total_counts
            
            logger.info(f"Model loaded successfully from {filename}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model from {filename}: {e}")
            raise


class TextLoader:
    """Utility class for loading and preprocessing text data"""
    
    @staticmethod
    def load_text_from_file(file_path: str) -> Optional[str]:
        """Load and preprocess text from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                text = ' '.join(text.split()).strip()
                
                if NLP:
                    doc = NLP(text)
                    cleaned_tokens = [token.text for token in doc 
                                    if not token.is_punct and not token.is_space]
                    return ' '.join(cleaned_tokens)
                
                return text
                
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None


# Django-compatible factory function
def create_language_model(model_file: str = Config.MODEL_FILE, 
                         text_file: str = Config.TEXT_PATH) -> EnhancedLanguageModel:
    """
    Factory function to create or load language model
    Django-compatible initialization
    """
    try:
        # Try to load existing model
        language_model = EnhancedLanguageModel.load_model(model_file)
        logger.info("Loaded existing model")
        
    except (FileNotFoundError, EOFError):
        # Create new model if not exists
        text = TextLoader.load_text_from_file(text_file)
        
        if not text:
            raise RuntimeError(f"Could not load text from {text_file}")
        
        language_model = EnhancedLanguageModel(text, n=2)
        language_model.save_model(model_file)
        logger.info("Created and saved new model")
    
    return language_model


# Example usage (can be removed in production)
if __name__ == "__main__":
    # This block won't run when imported as a module
    try:
        model = create_language_model()
        
        # Generate text
        input_sentence = "The truth "
        input_words = input_sentence.strip().rstrip('.').split()
        
        generated_text = model.generate_text(
            num_sentences=5,
            input_words=input_words,
            length=13,
            use_progress_bar=True
        )
        
        print("Generated Text:")
        print(generated_text)
        
        # Apply T5 correction
        corrected_text = model.correct_grammar_t5(generated_text)
        print("\nCorrected Text:")
        print(corrected_text)
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        print(f"Error: {e}")