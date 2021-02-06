""" Learning Keras functional API """

import silence_tensorflow.auto
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Sequential, Model
import numpy as np


def compare_seq_vs_api():
    seq_model = Sequential()
    seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
    seq_model.add(layers.Dense(32, activation='relu'))
    seq_model.add(layers.Dense(10, activation='softmax'))

    # ==

    input_tensor = Input(shape=(64,))
    x = layers.Dense(32, activation='relu')(input_tensor)
    x = layers.Dense(32, activation='relu')(x)
    ouput_tensor = layers.Dense(10, activation='softmax')(x)

    model = Model(input_tensor, ouput_tensor)

    seq_model.summary()
    model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    x_train = np.random.random((1000, 64))
    y_train = np.random.random((1000, 10))

    model.fit(x_train, y_train, epochs=10, batch_size=128)
    model.evaluate(x_train, y_train)


compare_seq_vs_api()
# p260