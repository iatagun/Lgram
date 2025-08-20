from .transition_analyzer import TransitionAnalyzer

class TransitionScorer:
    def __init__(self, weights=None):
        # Ağırlıkları dışarıdan da alabilirsin, varsayılan ağırlıkları ayarlayabilirsin.
        if weights is None:
            self.weights = {
                'CON': 3,
                'RET': 2,
                'SSH': 2,
                'RSH': 1,
                'NTT': 0,
                'EST': 1
            }
        else:
            self.weights = weights

    def score_transition(self, transition_type):
        # Her geçiş türü için skoru döner
        return self.weights.get(transition_type, 0)

class CenteringModel:
    def __init__(self, text, weights=None):
        self.analyzer = TransitionAnalyzer(text)
        self.scorer = TransitionScorer(weights)
        self.scores = []

    def score_transitions(self):
        results = self.analyzer.analyze()
        total_score = 0

        for result in results:
            transition = result['transition']
            score = self.calculate_transition_score(transition)
            total_score += score
            self.scores.append({
                'current_sentences': result['current_sentences'],
                'next_sentences': result['next_sentences'],
                'transition': transition,
                'score': score
            })

        # Hem toplam skoru hem de detaylı skorlama sonuçlarını döndür
        return total_score, self.scores

    def calculate_transition_score(self, transition_type):
        # Geçiş türünü kısa bir anahtarla eşleştir
        transition_mapping = {
            "Center Continuation (CON)": "CON",
            "Center Retaining (RET)": "RET",
            "Smooth Shift (SSH)": "SSH",
            "Rough Shift (RSH)": "RSH",
            "New Topic Transition": "NTT",
            "Center Establishment (EST)": "EST"
        }

        # Geçiş türünü bul ve skoru al, eşleşme yoksa varsayılan olarak 'NTT' kullan
        mapped_type = transition_mapping.get(transition_type, "NTT")
        return self.scorer.score_transition(mapped_type)

