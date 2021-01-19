""" Sentiment analysis """

import silence_tensorflow.auto
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
from tuto_utils import util_func


# decoder
def decode_review(index_review):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # i - 3 because 0, 1, 2 are reserved indices for "padding", "start of sequence" and "unknown"
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in index_review])


# imdb ds, label 0 is neg, 1 is pos
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# np seems to allow np.array[i, []] = 1 type of assign, cool !
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train, x_test = vectorize_sequences(train_data), vectorize_sequences(test_data)
y_train, y_test = np.asarray(train_labels).astype('float32'), np.asarray(test_labels).astype('float32')
epochs = 5

model = models.Sequential()
model.add(layers.Dense(16, input_shape=(10000,), activation='relu',
                       kernel_regularizer=regularizers.l2(.001)))
model.add(layers.Dense(16, activation='relu',
                       kernel_regularizer=regularizers.l2(.001)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
x_val, y_val = x_train[:10000], y_train[:10000]
partial_x_train, partial_y_train = x_train[10000:], y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=epochs, batch_size=512, validation_data=(x_val, y_val))
history_dict = history.history
print(history_dict.keys())
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

util_func.plot_history(loss_values, val_loss_values, epochs, metric='loss')
