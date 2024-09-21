from centering_model import CenteringModel
import pandas as pd

def analyze_transitions(text):
    # Initialize the model with the given text
    model = CenteringModel(text)
    # Score the transitions
    model.score_transitions()
    
    # Data structure to hold the transition information
    data = []

    # Gather transition information
    for score in model.scores:
        transition_info = {
            "current_sentence": score['current_sentences'],
            "next_sentence": score['next_sentences'],
            "transition_type": score['transition'],
            "score": score['score']
        }
        data.append(transition_info)
    
    return pd.DataFrame(data)  # Return as a DataFrame for easier manipulation
