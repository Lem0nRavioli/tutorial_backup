""" TensorBoard / Help visualize what happen in the network
    p277, need to use that in a virtual env, conflict between dependencies
    Command line is : "tensorboard --logdir=my_log_dir" then go to localhost:6006
    Windows is fucking annoying, everything is coded for linux
    plot_model also is broken in it"""

import silence_tensorflow.auto
import pydotplus
import pydot
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, callbacks
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import plot_model

max_features = 2000
max_len = 500

(x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_val = sequence.pad_sequences(x_val, maxlen=max_len)


model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

callback = [callbacks.TensorBoard(log_dir='tensor_board_logs', histogram_freq=1, embeddings_freq=1)]
# history = model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

plot_model(model, to_file='model.png')