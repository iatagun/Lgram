import os
import pickle
import json
from pathlib import Path
from functools import lru_cache
import sys
import os

# Bu dosyanın bulunduğu klasörün bir üst klasörünü sys.path'e ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models.chunk import EnhancedLanguageModel  # Modelin tanımı bu dosyadaysa path’i buna göre ayarla

# === Parametreler ===
model_file = Path("C:/Users/user/OneDrive/Belgeler/GitHub/Lgram/ngrams/language_model.pkl")
text_path = Path("C:/Users/user/OneDrive/Belgeler/GitHub/Lgram/ngrams/text_data.txt")
N_GRAM_SIZE = 6

# === Text'i oku ===
def load_text(text_path: Path) -> str:
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# === MODEL CACHE ===
@lru_cache(maxsize=1)
def get_language_model(model_file: str, n: int = 6) -> EnhancedLanguageModel:
    if os.path.exists(model_file):
        model = EnhancedLanguageModel.load_model(model_file)
        model.log("✅ Cached model loaded from disk.")
    else:
        text_data = load_text(text_path)  # sadece model yoksa oku
        model = EnhancedLanguageModel(text_data, n=n)
        model.save_model(model_file)
        model.log("🚀 New model created and saved.")
    return model


# === METİN ÜRETİMİ ===
def generate_text(input_words=("i", "know"), num_sentences=5, length=20):
    text_data = load_text(text_path)
    model = get_language_model(str(model_file), n=N_GRAM_SIZE)
    result = model.generate_and_post_process(
        num_sentences=num_sentences,
        input_words=input_words,
        length=length
    )
    model.log("📄 Generated Text:\n" + result)
    return result

# === Kullanım ===
if __name__ == "__main__":
    # The suspect saw a clue in the alley early in the morning.
    input_words = ("The", "suspect", "saw", "a", "clue")  # Buraya istediğin başlangıç kelimelerini yazabilirsin
    generated = generate_text(input_words=input_words, num_sentences=5, length=20)
    print("\n🔹 Generated Text 🔹\n")
    print(generated)
    
