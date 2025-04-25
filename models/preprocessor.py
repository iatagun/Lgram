import spacy
import random

nlp = spacy.load("en_core_web_lg")
nlp.max_length = 2_000_000  # SpaCy'yi daha toleranslı yapıyoruz
text_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\text_data.txt"

def smart_format_text_into_paragraphs_safe(text, min_sentences=3, max_sentences=5, chunk_size=500_000):
    """
    Büyük metni güvenli bir şekilde SpaCy ile analiz ederek paragraflara ayırır.
    SpaCy'nin max_length limitine takılmaması için metin bloklara bölünür.
    """
    paragraphs = []

    # Metni chunk'lara ayır (karakter bazlı)
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    for chunk in chunks:
        doc = nlp(chunk.strip())
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

        current_paragraph = []

        for i, sentence in enumerate(sentences):
            sent_doc = nlp(sentence)
            has_subject = any(tok.dep_ in ("nsubj", "nsubjpass", "expl") for tok in sent_doc)
            has_verb = any(tok.pos_ == "VERB" for tok in sent_doc)
            starts_with_connector = sent_doc[0].text.lower() in {
                "but", "however", "although", "though", "meanwhile", "because", "yet", "nevertheless", "nonetheless"
            }

            current_paragraph.append(sentence)

            if (
                len(current_paragraph) >= random.randint(min_sentences, max_sentences)
                or starts_with_connector
                or (not has_subject and not has_verb)
            ):
                paragraphs.append(' '.join(current_paragraph).strip())
                current_paragraph = []

        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph).strip())

    return '\n\n'.join(paragraphs)



with open(text_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

formatted_text = smart_format_text_into_paragraphs_safe(raw_text)

with open("formatted_text_safe.txt", "w", encoding="utf-8") as f:
    f.write(formatted_text)