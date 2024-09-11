# -*- coding: utf-8 -*-
import numpy as np
import string
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from itertools import chain
from keras.preprocessing.text import Tokenizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class Centering:
    """Centering theory is a framework for theorizing about local coherence
    by focusing on the relationships between discourse entities in sentences."""

    def __init__(self, sent, n):
        self.sent = sent
        self.n = n
        self.tokenizer = Tokenizer()

    def preprocess(self):
        """Tokenize sentences and words, perform POS tagging."""
        tokens = sent_tokenize(self.sent)  # Tokenize sentences
        output = []
        for i in range(len(tokens) - self.n + 1):
            output.append(tokens[i:i + self.n])  # Create sliding window for n sentences
        
        df = pd.DataFrame(output)
        print("Sentence Windows:\n", df)  # Display sentence windows
        
        return output

    def extract_entities(self, output):
        """Extract proper nouns (NNP) and pronouns (PRP) from each sentence."""
        data = []
        for sentence_group in output:
            data2 = []
            for sentence in sentence_group:
                words = word_tokenize(sentence)  # Tokenize words in each sentence
                tagged = pos_tag(words)  # Perform POS tagging
                for word, tag in tagged:
                    if tag in ['NNP', 'PRP']:  # Focus on proper nouns and pronouns
                        data2.append(word)
            data.append(data2)
        
        return data

    def clean_data(self, data):
        """Flatten the list of entities and remove punctuation."""
        flatten_list = list(chain.from_iterable(data))  # Flatten nested lists
        flatten_list = [word for word in flatten_list if word not in string.punctuation]  # Remove punctuation
        unique_words = list(set(flatten_list))  # Keep only unique words
        
        print("Unique Entities:", unique_words)
        return unique_words

    def create_sequences(self, unique_words):
        """Create sequences for RNN model training."""
        self.tokenizer.fit_on_texts([unique_words])
        encoded_data = self.tokenizer.texts_to_sequences([unique_words])[0]  # Encode words to integers
        
        if len(encoded_data) % 2 != 0:  # Ensure we have an even number of elements
            encoded_data = encoded_data[:-1]
        
        sequence = np.reshape(encoded_data, (-1, 2))  # Reshape into pairs for sequence modeling
        print("Encoded Sequences:\n", sequence)
        return sequence

    def anaphora(self):
        """Main function to run the Centering model."""
        output = self.preprocess()  # Tokenize and prepare sentence windows
        entities = self.extract_entities(output)  # Extract proper nouns and pronouns
        cleaned_data = self.clean_data(entities)  # Clean and prepare entity list
        sequences = self.create_sequences(cleaned_data)  # Create sequences for RNN input
        return sequences


# Örnek kullanım:
centering = Centering("Alice went to the park. She saw a cat. The cat was sleeping.", 2)
centering.anaphora()
