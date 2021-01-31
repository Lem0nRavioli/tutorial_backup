""" Recurrent neural networks """
""" One-hot encoding toy examples (Word-level or character-level) """

import numpy as np
import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']


""" Word level """
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1  # not attributing the 0 index
max_length = 10
# last shape value +1 explained: if there is 2 word, max(token_index.values() == 2
# then you get np.zeros(2, 10, 2), here adding +1 result in shape (2, 10, 3), i guess it's keeping 0 index for unknown
results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1


""" Character level """

characters = string.printable
token_index_char = dict(zip(range(1, len(characters) + 1), characters))
max_length_char = 50
results_char = np.zeros((len(samples), max_length_char, max(token_index_char.keys()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index_char.get(character)
        results_char[i, j, index] = 1

# print(results_char.shape)
# print(results)

""" One hot word with hashing trick """
# see p206
dimensionality = 1000
max_length = 10
results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1


import silence_tensorflow.auto
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)  # Build the word index
sequences = tokenizer.texts_to_sequences(samples)  # turn strings into list of int indices
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')  # turn string into one_hot binary
word_index = tokenizer.word_index  # dict of words
print('Found %s unique tokens.' % len(word_index))
print(one_hot_results.shape)
print(sequences)

# p207
