""" RNN """
import silence_tensorflow.auto
from tensorflow.keras.layers import Embedding, Flatten, Dense, RNN, SimpleRNN, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras import preprocessing
from tensorflow.keras.datasets import imdb
import os
import numpy as np
from util_func import plot_acc_loss


def rnn_toy_example():
    timesteps = 100
    input_features = 32
    output_features = 64

    inputs = np.random.random((timesteps, input_features))
    state_t = np.zeros((output_features,))

    W = np.random.random((output_features, input_features))
    U = np.random.random((output_features, output_features))
    b = np.random.random((output_features,))

    successive_outputs = []
    for input_t in inputs:
        output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
        successive_outputs.append(output_t)
        state_t = output_t

    final_output_sequence = np.concatenate(successive_outputs, axis=0)
    return final_output_sequence


def sample_architecture():
    """ return_sequence=True return every output at timestep t
        Necessary for stacking multiple RNN that will require a time factor to compute """
    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(SimpleRNN(32))
    model.summary()

    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32))
    model.summary()


max_features = 10000
maxlen = 500
epochs = 10

print('loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = preprocessing.sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = preprocessing.sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


def train_plot_simple_RNN():
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=.2)

    plot_acc_loss(history, epochs)


def train_plot_LSTM():
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=.2)

    plot_acc_loss(history, epochs)


# p230
# p244 for reverse ordering


def try_reverse_order():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = [x[::-1] for x in x_train]
    x_test = [x[::-1] for x in x_test]

    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=.2)

    plot_acc_loss(history, 10)


def try_bidirectionnal_layer():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = [x[::-1] for x in x_train]
    x_test = [x[::-1] for x in x_test]

    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=.2)

    plot_acc_loss(history, 10)


# train_plot_LSTM()
try_bidirectionnal_layer()