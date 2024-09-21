from transition_analyzer import TransitionAnalyzer

class CenteringModel:
    def __init__(self, text):
        self.analyzer = TransitionAnalyzer(text)
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

        return total_score

    def calculate_transition_score(self, transition_type):
        if transition_type == "Center Continuation (CON)":
            return 3
        elif transition_type == "Center Retaining (RET)":
            return 2
        elif transition_type == "Smooth Shift (SSH)":
            return 2
        elif transition_type == "Rough Shift (RSH)":
            return 1
        elif transition_type == "New Topic Transition":
            return 0
        else:
            return 0
