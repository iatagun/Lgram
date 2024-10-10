import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors

# Your custom imports for the sentence generation
from text_generation import SentenceGenerator, load_text_data

def evaluate_fluency(text):
    sentences = nltk.sent_tokenize(text)
    avg_sentence_length = np.mean([len(nltk.word_tokenize(sent)) for sent in sentences])
    num_complex_sentences = sum(1 for sent in sentences if len(nltk.sent_tokenize(sent)) > 1)
    
    fluency_score = (avg_sentence_length / 20) + (num_complex_sentences / len(sentences))
    fluency_score = min(fluency_score, 1.0)  # Ensure score does not exceed 1.0
    return fluency_score

def evaluate_accuracy(generated_text, reference_texts):
    vectorizer = CountVectorizer().fit_transform([generated_text] + reference_texts)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)

    # Get the similarity scores between the generated text and each reference text
    similarity_scores = cosine_matrix[0][1:]  # First row is the generated text
    avg_similarity = np.mean(similarity_scores)  # Average similarity
    return avg_similarity

def load_word2vec_model():
    # Load the Word2Vec model from pre-trained vectors
    model = KeyedVectors.load_word2vec_format("C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\GoogleNews-vectors-negative300.bin", binary=True)
    return model

def evaluate_transitions(text):
    """Evaluate transitions between sentences using Word2Vec-based semantic similarity."""
    model = load_word2vec_model()  # Load the Word2Vec model
    sentences = nltk.sent_tokenize(text)
    transition_scores = []
    
    for i in range(1, len(sentences)):
        prev_sent = sentences[i - 1]
        curr_sent = sentences[i]
        
        prev_tokens = nltk.word_tokenize(prev_sent)
        curr_tokens = nltk.word_tokenize(curr_sent)
        
        # Filter tokens in the model's vocabulary
        prev_vectors = [model[token] for token in prev_tokens if token in model]
        curr_vectors = [model[token] for token in curr_tokens if token in model]
        
        if prev_vectors and curr_vectors:
            prev_vector_avg = np.mean(prev_vectors, axis=0)
            curr_vector_avg = np.mean(curr_vectors, axis=0)
            cosine_sim = cosine_similarity([prev_vector_avg], [curr_vector_avg])[0][0]
            transition_scores.append(cosine_sim)
    
    avg_transition_score = np.mean(transition_scores) if transition_scores else 0
    return avg_transition_score

def evaluate_consistency(text):
    sentences = nltk.sent_tokenize(text)
    consistency_scores = []
    
    for i in range(1, len(sentences)):
        curr_sent = sentences[i]
        prev_sent = sentences[i - 1]
        
        vectorizer = CountVectorizer().fit_transform([curr_sent, prev_sent])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)[0][1]
        consistency_scores.append(cosine_sim)
    
    avg_consistency_score = np.mean(consistency_scores) if consistency_scores else 0
    return avg_consistency_score

def evaluate_thematic_coherence(text):
    sentences = nltk.sent_tokenize(text)
    thematic_scores = []

    vectorizer = CountVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences).toarray()
    
    for i in range(len(sentence_vectors) - 1):
        sim = cosine_similarity([sentence_vectors[i]], [sentence_vectors[i + 1]])[0][0]
        thematic_scores.append(sim)

    avg_thematic_coherence = np.mean(thematic_scores) if thematic_scores else 0
    return avg_thematic_coherence

def evaluate_generated_text(generated_text, reference_texts):
    fluency_score = evaluate_fluency(generated_text)
    accuracy_score = evaluate_accuracy(generated_text, reference_texts)
    transition_score = evaluate_transitions(generated_text)
    consistency_score = evaluate_consistency(generated_text)
    thematic_coherence_score = evaluate_thematic_coherence(generated_text)

    return fluency_score, accuracy_score, transition_score, consistency_score, thematic_coherence_score

# Load text data for generating sentences
text_file_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_gen_data.txt"
transition_model_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\best_transition_model.keras"
input_text = load_text_data(text_file_path)

# Create a SentenceGenerator instance and generate text
sentence_generator = SentenceGenerator(input_text, transition_model_path)
initial_sentence = "As the spirit began to fade, it gifted each of them a glowing seed."
num_sentences_to_generate = 20

# Generate text using the sentence generator
generated_text = sentence_generator.generate_text(initial_sentence, num_sentences_to_generate)
print("Generated Text:")
print(generated_text)

# Load reference texts for evaluation
reference_file_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_gen_data.txt"
with open(reference_file_path, 'r', encoding='utf-8') as file:
    reference_texts = file.readlines()

# Evaluate the generated text
fl_score, acc_score, trans_score, cons_score, them_score = evaluate_generated_text(generated_text, reference_texts)

# Output the evaluation results
print(f"\nFluency Score: {fl_score * 100:.2f}%")
print(f"Accuracy Score: {acc_score * 100:.2f}%")
print(f"Transition Score: {trans_score * 100:.2f}%")
print(f"Consistency Score: {cons_score * 100:.2f}%")
print(f"Thematic Coherence Score: {them_score * 100:.2f}%")
