from text_generation import SentenceGenerator, DynamicNGram, load_text_data
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def evaluate_fluency(text):
    """Evaluate the fluency of generated text."""
    sentences = nltk.sent_tokenize(text)
    avg_sentence_length = np.mean([len(nltk.word_tokenize(sent)) for sent in sentences])
    fluency_score = avg_sentence_length / 20  # Normalize to a scale of 0-1 for example
    return fluency_score

def evaluate_accuracy(generated_text, reference_texts):
    """Evaluate the accuracy of generated text against reference texts."""
    vectorizer = CountVectorizer().fit_transform([generated_text] + reference_texts)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)

    # Get the similarity scores between the generated text and each reference text
    similarity_scores = cosine_matrix[0][1:]  # First row is the generated text
    avg_similarity = np.mean(similarity_scores)  # Average similarity
    return avg_similarity

def evaluate_transitions(text):
    """Evaluate the transitions between sentences."""
    sentences = nltk.sent_tokenize(text)
    transition_scores = []
    
    for i in range(1, len(sentences)):
        prev_sent = sentences[i - 1]
        curr_sent = sentences[i]
        
        # Calculate the cosine similarity between the previous and current sentence
        vectorizer = CountVectorizer().fit_transform([prev_sent, curr_sent])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)[0][1]  # Similarity score between two sentences
        transition_scores.append(cosine_sim)
    
    # Calculate average transition score
    avg_transition_score = np.mean(transition_scores) if transition_scores else 0
    return avg_transition_score

def evaluate_consistency(text):
    """Evaluate the consistency of the generated text."""
    sentences = nltk.sent_tokenize(text)
    consistency_scores = []
    
    for i in range(1, len(sentences)):
        curr_sent = sentences[i]
        prev_sent = sentences[i - 1]
        
        # Measure similarity (cosine) between consecutive sentences
        vectorizer = CountVectorizer().fit_transform([curr_sent, prev_sent])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)[0][1]  # Get the similarity score
        consistency_scores.append(cosine_sim)
    
    avg_consistency_score = np.mean(consistency_scores) if consistency_scores else 0
    return avg_consistency_score

def evaluate_thematic_coherence(text):
    """Evaluate the thematic coherence of the generated text."""
    sentences = nltk.sent_tokenize(text)
    thematic_scores = []

    # Use the CountVectorizer to convert sentences to vectors
    vectorizer = CountVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences).toarray()
    
    for i in range(len(sentence_vectors) - 1):
        # Calculate the cosine similarity between consecutive sentence vectors
        sim = cosine_similarity([sentence_vectors[i]], [sentence_vectors[i + 1]])[0][0]
        thematic_scores.append(sim)

    avg_thematic_coherence = np.mean(thematic_scores) if thematic_scores else 0
    return avg_thematic_coherence

def evaluate_generated_text(generated_text, reference_texts):
    """Evaluate generated text for fluency, accuracy, transitions, consistency, and thematic coherence."""
    fluency_score = evaluate_fluency(generated_text)
    accuracy_score = evaluate_accuracy(generated_text, reference_texts)
    transition_score = evaluate_transitions(generated_text)
    consistency_score = evaluate_consistency(generated_text)
    thematic_coherence_score = evaluate_thematic_coherence(generated_text)

    return fluency_score, accuracy_score, transition_score, consistency_score, thematic_coherence_score

# Load text data
text_file_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_gen_data.txt"
transition_model_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\best_transition_model.keras"

# Load text data
text = load_text_data(text_file_path)

# Create a SentenceGenerator instance
sentence_generator = SentenceGenerator(text, transition_model_path)

# Generate text starting from an initial sentence
initial_sentence = "As the spirit began to fade, it gifted each of them a glowing seed."
generated_text = sentence_generator.generate_text(initial_sentence, num_sentences=30)

# Output the generated text
print("\nGenerated Text:")
print(generated_text)

# Load reference sentences from a text file for comparison
reference_file_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_gen_data.txt"
with open(reference_file_path, 'r', encoding='utf-8') as file:
    reference_texts = file.readlines()

# Evaluate the generated text
(fl_score, acc_score, trans_score, cons_score, them_score) = evaluate_generated_text(generated_text, reference_texts)

# Output the evaluation results
print(f"\nFluency Score: {fl_score * 100:.2f}%")
print(f"Accuracy Score: {acc_score * 100:.2f}%")
print(f"Transition Score: {trans_score * 100:.2f}%")
print(f"Consistency Score: {cons_score * 100:.2f}%")
print(f"Thematic Coherence Score: {them_score * 100:.2f}%")
