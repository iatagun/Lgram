import spacy
import re
import json
import os
import argparse
from tqdm import tqdm

# SpaCy İngilizce modelini yükle
nlp = spacy.load("en_core_web_sm")  # Küçük model yeterli şimdilik

# Varsayılan yollar (Argümanlarla değiştirilebilir)
DEFAULT_LOGS_FOLDER = r"C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\logs"
DEFAULT_CORRECTIONS_FILE = r"C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\corrections.json"

# Çıkarılan otomatik düzeltmeleri burada toplayacağız
auto_corrections = {}


def detect_errors_and_propose_fixes(text):
    """
    Verilen paragrafta sık rastlanan İngilizce kullanım hatalarını tespit eder
    ve düzeltme önerileri auto_corrections sözlüğüne ekler.
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]

    for i in range(len(tokens) - 1):
        window = tokens[i:i+3]
        window_pos = pos_tags[i:i+3]

        # 1. 'has/have + base form' hatası
        if window_pos[0] in ("AUX", "VERB") and window_pos[1] == "VERB":
            if window[0].lower() in ("has", "have") and not re.search(r"(ed|en)$", window[1]):
                wrong = f"{window[0]} {window[1]}"
                right = f"{window[0]} {window[1]}ed"
                auto_corrections.setdefault(wrong, right)

        # 2. 'be + base verb' hatası
        if window_pos[0] == "AUX" and window_pos[1] == "VERB":
            if window[0].lower() in ("is", "are", "was", "were", "be") and not window[1].endswith("ing"):
                wrong = f"{window[0]} {window[1]}"
                right = f"{window[0]} {window[1]}ing"
                auto_corrections.setdefault(wrong, right)

        # 3. Eksik article ('a', 'the') hataları (çoğul yap)
        if window_pos[0] == "NUM" and window_pos[1] == "NOUN":
            wrong = f"{window[0]} {window[1]}"
            right = f"{window[0]} {window[1]}s"
            auto_corrections.setdefault(wrong, right)

        # 4. Yalnızca 'be' fiilinin eksik çekimi
        if window_pos[0] == "NOUN" and window_pos[1] == "AUX" and window[1].lower() == "be":
            wrong = f"{window[0]} be"
            right = f"{window[0]} is"
            auto_corrections.setdefault(wrong, right)

        # 5. Tekrarlanan kelimeler (duplicate word)
        if window[0].lower() == window[1].lower():
            wrong = f"{window[0]} {window[1]}"
            right = window[0]
            auto_corrections.setdefault(wrong, right)

    # 6. Giriş ifadesi sonrası virgül eksikliği
    intro_match = re.match(r'^(However|Moreover|Therefore|Meanwhile|Consequently)\s+(\w+)', text)
    if intro_match:
        first = intro_match.group(1)
        rest = text[len(first):].lstrip()
        wrong = text.strip()
        right = f"{first}, {rest}"
        auto_corrections.setdefault(wrong, right)


def main(logs_folder, corrections_file):
    # Mevcut düzeltmeleri yükle
    if os.path.exists(corrections_file):
        with open(corrections_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    # Log dosyalarını tara
    files = [f for f in os.listdir(logs_folder) if f.startswith("daily_log_") and f.endswith(".txt")]
    for filename in tqdm(files, desc="Processing logs"):
        path = os.path.join(logs_folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = re.split(r'\n\n+', text)
        for para in paragraphs:
            detect_errors_and_propose_fixes(para)
        print(f"{filename}: {len(auto_corrections)} total errs detected so far.")

    # Yeni bulunanları birleştir ve kaydet
    merged = {**existing, **auto_corrections}
    with open(corrections_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)

    print(f"\u2705 Toplam {len(auto_corrections)} yeni correction bulundu ve '{corrections_file}' dosyasına yazıldı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic English correction finder")
    parser.add_argument('--logs', default=DEFAULT_LOGS_FOLDER, help='Logs folder path')
    parser.add_argument('--output', default=DEFAULT_CORRECTIONS_FILE, help='Corrections JSON file')
    args = parser.parse_args()
    main(args.logs, args.output)

