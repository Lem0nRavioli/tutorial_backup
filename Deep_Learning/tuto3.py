""" Multilabel classification with words """

import silence_tensorflow.auto
from tensorflow.keras.datasets import reuters
from tensorflow.keras import models, layers
import numpy as np
from tuto_utils import util_func

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def decode_review(index_review):
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # i - 3 because 0, 1, 2 are reserved indices for "padding", "start of sequence" and "unknown"
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in index_review])


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# Can also be done with tensorflow.keras.utils.np_utils.to_categorical
# one_hot_label = to_categorical(label)
# if not using one_hot_labels, loss function must be "sparse_categorical_crossentropy"
def to_one_hot(labels):
    num_cat = np.max(labels) + 1
    results = np.zeros((len(labels), num_cat))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


x_train, x_test = vectorize_sequences(train_data), vectorize_sequences(test_data)
y_train_onehot, y_test_onehot = to_one_hot(train_labels), to_one_hot(test_labels)
x_val, partial_x_train = x_train[:1000], x_train[1000:]
y_val_onehot, partial_y_train_onehot = y_train_onehot[:1000], y_train_onehot[1000:]

epochs = 9

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(.3))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(.2))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(partial_x_train, partial_y_train_onehot,
                    epochs=epochs, batch_size=256, validation_data=(x_val, y_val_onehot))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

util_func.plot_history(loss_values, val_loss_values, epochs, metric='loss')
util_func.plot_history(acc_values, val_acc_values, epochs, metric='accuracy')


# p 108