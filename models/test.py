import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import language_tool_python

# Your custom imports for the sentence generation
from text_generation import SentenceGenerator, load_text_data

tool = language_tool_python.LanguageTool('en-US')

def evaluate_fluency(text):
    sentences = nltk.sent_tokenize(text)
    avg_sentence_length = np.mean([len(nltk.word_tokenize(sent)) for sent in sentences]) if sentences else 0
    num_complex_sentences = sum(1 for sent in sentences if len(nltk.word_tokenize(sent)) > 20)  # Adjusted threshold for complexity
    
    # Calculate fluency score considering no division by zero
    if len(sentences) > 0:
        fluency_score = (avg_sentence_length / 20) + (num_complex_sentences / len(sentences))
        fluency_score = min(fluency_score, 1.0)  # Ensure score does not exceed 1.0
    else:
        fluency_score = 0
    
    return fluency_score

def evaluate_accuracy(generated_text, reference_texts):
    vectorizer = CountVectorizer().fit_transform([generated_text] + reference_texts)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)

    # Get the similarity scores between the generated text and each reference text
    similarity_scores = cosine_matrix[0][1:]  # First row is the generated text
    avg_similarity = np.mean(similarity_scores)  # Average similarity
    return avg_similarity

def evaluate_transitions(text):
    """Evaluate transitions between sentences using traditional cosine similarity."""
    sentences = nltk.sent_tokenize(text)
    transition_scores = []
    
    # Use more advanced methods or thresholding to improve the transition evaluation
    for i in range(1, len(sentences)):
        prev_sent = sentences[i - 1]
        curr_sent = sentences[i]
        
        # Combine sentences for context
        combined = prev_sent + " " + curr_sent
        
        vectorizer = CountVectorizer().fit_transform([combined])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)[0][0]  # Similarity of combined sentences
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
initial_sentence = "This is one of those miserable thoroughfares which intervene between the Rue Richelieu and the Rue St. Roch."
num_sentences_to_generate = 10

# Generate text using the sentence generator
generated_text = sentence_generator.generate_text(initial_sentence, num_sentences_to_generate)

print('generated_text: ')
print(generated_text)

matches = tool.check(generated_text)

# Metindeki hataları düzelten bir fonksiyon
def apply_corrections(text, matches):
    corrected_text = text
    # Her hatayı düzeltmek için ters sırayla ilerle
    for match in reversed(matches):
        start = match.offset  # Hatanın başladığı nokta
        end = match.offset + match.errorLength  # Hatanın bittiği nokta
        replacement = match.replacements[0] if match.replacements else match.context  # Önerilen düzeltme
        # Metni düzelt
        corrected_text = corrected_text[:start] + replacement + corrected_text[end:]
    return corrected_text

# Düzeltmeleri uygula
corrected_text = apply_corrections(generated_text, matches)

# Düzeltilmiş metni yazdır
print("Düzeltilmiş metin:", corrected_text)

# Load reference texts for evaluation
# reference_file_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_gen_data.txt"
# with open(reference_file_path, 'r', encoding='utf-8') as file:
    # reference_texts = file.readlines()

# Evaluate the generated text
# fl_score, acc_score, trans_score, cons_score, them_score = evaluate_generated_text(corrected_text, reference_texts)

# Output the evaluation results
# print(f"\nFluency Score: {fl_score * 100:.2f}%")
# print(f"Accuracy Score: {acc_score * 100:.2f}%")
# print(f"Transition Score: {trans_score * 100:.2f}%")
# print(f"Consistency Score: {cons_score * 100:.2f}%")
# print(f"Thematic Coherence Score: {them_score * 100:.2f}%")