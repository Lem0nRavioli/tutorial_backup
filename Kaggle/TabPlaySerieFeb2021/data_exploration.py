import pandas as pd
import numpy as np
import silence_tensorflow.auto
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tuto_utils.util_func import plot_history

data = pd.read_csv('../../DataSets/TPSfeb2021/train.csv')
# print(data.nunique())
# print(data.describe())

cols = list(data.columns)
data_cat = data[cols[1:11]]
data_cat = pd.get_dummies(data_cat)
data_num = data[cols[11:-1]]
target = data['target']
data = data_cat.join(data_num)
print(data.shape)
x_train, y_train = data[:200000], target[:200000]
x_valid, y_valid = data[200000:250000], target[200000:250000]
x_test, y_test = data[250000:], target[250000:]

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(70,)))
model.add(Dropout(.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse')


def test_model():
    history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_valid, y_valid))

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_history(loss, val_loss, 20, 'loss')
    # model.evaluate(x_test, y_test)


# model.fit(data, target, epochs=20, batch_size=128)
test_model()

# .7461 128/.5/64/.5/32/1