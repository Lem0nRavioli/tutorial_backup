""" Comparing CNN to LSTM for timeseries related data """

import silence_tensorflow.auto
from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tuto_utils.util_func import plot_acc_loss, plot_history
import tuto11
from tuto11 import generator


max_features = 10000
maxlen = 500
epochs = 10

print('loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
print('input_train shape:', x_train.shape)
print('input_test shape:', x_test.shape)


def try_conv1d_imdb():
    model = Sequential()
    model.add(layers.Embedding(max_features, 128, input_length=maxlen))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))

    model.summary()

    model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=8, batch_size=128, validation_split=.2)
    plot_acc_loss(history, 8)

    model.evaluate(x_test, y_test)  # .85 acc, .43 loss


# loss .4, naive implementation was .29
def try_conv1d_weather():
    train_gen = tuto11.train_gen
    val_gen = tuto11.val_gen
    train_steps = tuto11.train_steps
    val_steps = tuto11.val_steps

    model = Sequential()
    model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, tuto11.float_data.shape[-1])))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))

    model.summary()

    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit(train_gen, steps_per_epoch=train_steps, epochs=20,
                        validation_data=val_gen, validation_steps=val_steps)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_history(loss, val_loss, 20, 'loss')


# loss .27, slightly better than naive, not by much
def try_conv1d_preprocessing():
    float_data = tuto11.float_data
    step = 3
    lookback = 720
    delay = 144
    batch_size = 128

    train_gen = generator(float_data, lookback, delay, 0, 200000, shuffle=True, step=step, batch_size=batch_size)
    val_gen = generator(float_data, lookback, delay, 200001, 300000, step=step, batch_size=batch_size)
    test_gen = generator(float_data, lookback, delay, 300001, None, step=step, batch_size=batch_size)

    train_steps = int((200000 - lookback) / batch_size)
    val_steps = int((300000 - 200001 - lookback) / batch_size)
    test_steps = int((len(float_data) - 300001 - lookback) / batch_size)

    print('float data shape[-1]:', float_data.shape[-1])

    model = Sequential()
    model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.GRU(32, dropout=.1, recurrent_dropout=.5))
    model.add(layers.Dense(1))

    model.summary()
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit(train_gen, steps_per_epoch=train_steps, epochs=20,
                        validation_data=val_gen, validation_steps=val_steps)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plot_history(loss, val_loss, 20, 'loss')


# P255