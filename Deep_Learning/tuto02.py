import silence_tensorflow.auto
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.datasets import imdb
import numpy as np


# decoder
def decode_review(index_review):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # i - 3 because 0, 1, 2 are reserved indices for "padding", "start of sequence" and "unknown"
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in index_review])


# imdb ds, label 0 is neg, 1 is pos
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# np seems to allow np.array[i, []] = 1 type of assign, cool !
print(len(train_data))
print(np.shape(train_data))
for i, sequence in enumerate([train_data[0]]):
    print(sequence)