from centering_model import CenteringModel
import pandas as pd

def analyze_transitions(text):
    # Initialize the model with the given text
    model = CenteringModel(text)
    # Score the transitions and retrieve total score and detailed scores
    total_score, scores = model.score_transitions()
    
    # Data structure to hold the transition information
    data = []

    # Gather transition information
    for score in scores:
        transition_info = {
            "current_sentence": score['current_sentences'],
            "next_sentence": score['next_sentences'],
            "transition_type": score['transition'],
            "score": score['score']
        }
        data.append(transition_info)
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(data)
    df['total_score'] = total_score  # Add total score as a new column

    return df
