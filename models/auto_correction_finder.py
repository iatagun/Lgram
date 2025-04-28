import spacy
import re
import json
import os

# SpaCy İngilizce modelini yükle
nlp = spacy.load("en_core_web_sm")  # Küçük model yeterli şimdilik

# Hatalı üretimlerin olduğu klasör
logs_folder = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\logs\\"

# Çıkarılan otomatik düzeltmeleri burada toplayacağız
auto_corrections = {}

def detect_errors_and_propose_fixes(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]

    for i in range(len(tokens) - 1):
        window = tokens[i:i+3]
        window_pos = pos_tags[i:i+3]

        # 1. 'has/have + base form' hatası
        if window_pos[0] in ("AUX", "VERB") and window_pos[1] == "VERB":
            if window[0].lower() in ("has", "have") and not window[1].endswith("ed") and not window[1].endswith("en"):
                wrong = f"{window[0]} {window[1]}"
                right = f"{window[0]} {window[1]}ed"
                auto_corrections[wrong] = right

        # 2. 'be + base verb' hatası
        if window_pos[0] == "AUX" and window_pos[1] == "VERB":
            if window[0].lower() in ("is", "are", "was", "were", "be") and not window[1].endswith("ing"):
                wrong = f"{window[0]} {window[1]}"
                right = f"{window[0]} {window[1]}ing"
                auto_corrections[wrong] = right

        # 3. Eksik article ('a', 'the') hataları
        if window_pos[0] == "NUM" and window_pos[1] == "NOUN":
            wrong = f"{window[0]} {window[1]}"
            right = f"{window[0]} {window[1]}s"  # Çoğul yap
            auto_corrections[wrong] = right

        # 4. Yalnızca "be" fiilinin eksik çekimi
        if window_pos[0] == "NOUN" and window_pos[1] == "AUX":
            if window[1] == "be":
                wrong = f"{window[0]} be"
                right = f"{window[0]} is"
                auto_corrections[wrong] = right

corrections_file = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\corrections.json"

def main():
    # logs klasöründeki tüm daily_log dosyalarını bul
    for filename in os.listdir(logs_folder):
        if filename.startswith("daily_log_") and filename.endswith(".txt"):
            file_path = os.path.join(logs_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            paragraphs = re.split(r'\n\n+', text)
            
            for para in paragraphs:
                detect_errors_and_propose_fixes(para)

    # corrections.json dosyasını yükle (varsa)
    if os.path.exists(corrections_file):
        with open(corrections_file, "r", encoding="utf-8") as f:
            existing_corrections = json.load(f)
    else:
        existing_corrections = {}

    # Yeni bulunanları eski correctionlarla birleştir
    merged_corrections = {**existing_corrections, **auto_corrections}

    # corrections.json dosyasını güncelle
    with open(corrections_file, "w", encoding="utf-8") as f:
        json.dump(merged_corrections, f, indent=4, ensure_ascii=False)

    print(f"✅ {len(auto_corrections)} yeni correction bulundu ve corrections.json dosyasına eklendi.")


if __name__ == "__main__":
    main()
