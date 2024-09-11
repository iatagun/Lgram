# -*- coding: utf-8 -*-
import numpy as np
import string
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from itertools import chain
from nltk import FreqDist

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class Centering:
    def __init__(self, sent, n):
        self.sent = sent
        self.n = n

    def preprocess(self):
        tokens = nltk.sent_tokenize(self.sent)
        output = [tokens[i:i + self.n] for i in range(len(tokens) - self.n + 1)]
        df = pd.DataFrame(output)
        print("Sentence Windows:\n", df)
        return output

    def extract_entities(self, output):
        data = []
        for sentence_group in output:
            data2 = []
            for sentence in sentence_group:
                words = word_tokenize(sentence)
                tagged = pos_tag(words)
                for word, tag in tagged:
                    if tag in ['NNP', 'PRP']:
                        data2.append(word)
            data.append(data2)
        return data

    def clean_data(self, data):
        flatten_list = list(chain.from_iterable(data))
        flatten_list = [word for word in flatten_list if word not in string.punctuation]
        unique_words = list(set(flatten_list))
        print("Unique Entities:", unique_words)
        return unique_words

    def create_sequences(self, unique_words):
        fdist = FreqDist(unique_words)
        # Create a mapping of words to indices
        word_index = {word: i for i, word in enumerate(fdist.keys())}
        encoded_data = [word_index[word] for word in unique_words]
        if len(encoded_data) % 2 != 0:
            encoded_data = encoded_data[:-1]
        sequence = np.reshape(encoded_data, (-1, 2))
        print("Encoded Sequences:\n", sequence)
        return sequence

    def anaphora(self):
        output = self.preprocess()
        entities = self.extract_entities(output)
        cleaned_data = self.clean_data(entities)
        sequences = self.create_sequences(cleaned_data)
        return sequences

# Örnek kullanım:
centering = Centering("Alice went to the park. She saw a cat. The cat was sleeping.", 2)
centering.anaphora()



