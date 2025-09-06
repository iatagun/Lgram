import spacy

# İngilizce NLP modelini yükleyin
nlp = spacy.load("en_core_web_lg")

class TransitionAnalyzer:
    def __init__(self, text):
        """
        TransitionAnalyzer sınıfının başlangıç fonksiyonu.
        
        Parametreler:
        text (str): Analiz edilecek metin.
        """
        self.text = text

    def preprocess(self):
        """
        Metni cümlelere ayırır ve cümle çiftlerini oluşturur.

        Dönüş:
        List: Cümle çiftleri listesi.
        """
        doc = nlp(self.text)
        sentences = list(doc.sents)
        sentence_pairs = [(sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)]
        return sentence_pairs

    def extract_nps(self, sentence):
        """
        Verilen cümledeki isim tamlamalarını çıkarır.

        Parametreler:
        sentence (spacy.tokens.Span): Analiz edilecek cümle.

        Dönüş:
        List: İsim tamlamalarının listesi.
        """
        return [chunk.text for chunk in sentence.noun_chunks]

    def classify_transition(self, current_entities, next_entities, prev_cb):
        """
        Cümleler arasındaki geçişi sınıflandırır.

        Parametreler:
        current_entities (set): Mevcut cümledeki varlıklar.
        next_entities (set): Sonraki cümledeki varlıklar.
        prev_cb (set): Önceki merkez varlıklar.

        Dönüş:
        str: Geçiş türü.
        """
        cb = prev_cb
        cp = next_entities if next_entities else set()

        if not current_entities and not cp:
            return "Center Establishment (EST)"
        if cb == current_entities:
            if cb == cp:
                return "Center Continuation (CON)"
            else:
                return "Center Retaining (RET)"
        elif cb != current_entities:
            if cb & cp:
                return "Smooth Shift (SSH)"
            else:
                proper_nouns = {e for e in current_entities if e.istitle()}
                pronouns = {e for e in next_entities if e in ["He", "She", "They", "It"]}
                
                if proper_nouns and pronouns:
                    return "Center Continuation (CON)"
                else:
                    return "Rough Shift (RSH)"
        else:
            return "New Topic Transition"

    def annotate_anaphoric_relations(self, current_entities, next_entities):
        """
        Anaforik ilişkileri etiketler.

        Parametreler:
        current_entities (set): Mevcut cümledeki varlıklar.
        next_entities (set): Sonraki cümledeki varlıklar.

        Dönüş:
        dict: Anaforik bilgiler.
        """
        anaphoric_info = {}
        for entity in current_entities:
            if entity.lower() in (e.lower() for e in next_entities):
                anaphoric_info[entity] = {
                    "type": "identity",
                    "antecedent": entity
                }
        return anaphoric_info

    def analyze(self):
        """
        Metni analiz eder ve geçiş bilgilerini döndürür.

        Dönüş:
        List: Analiz sonuçları listesi.
        """
        sentence_pairs = self.preprocess()
        results = []
        prev_cb = set()

        for current_sentence, next_sentence in sentence_pairs:
            current_nps = self.extract_nps(current_sentence)
            next_nps = self.extract_nps(next_sentence)

            combined_current_nps = set(current_nps)
            combined_next_nps = set(next_nps)

            transition = self.classify_transition(combined_current_nps, combined_next_nps, prev_cb)
            anaphoric_relations = self.annotate_anaphoric_relations(combined_current_nps, combined_next_nps)

            prev_cb = combined_current_nps  # Sonraki iterasyon için CB'yi güncelle

            results.append({
                'current_sentences': current_sentence.text,
                'next_sentences': next_sentence.text,
                'transition': transition,
                'current_nps': combined_current_nps,
                'next_nps': combined_next_nps,
                'anaphoric_relations': anaphoric_relations
            })

        return results
