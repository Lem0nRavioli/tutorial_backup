import pandas as pd
import silence_tensorflow.auto
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tuto_utils.util_func import plot_history

# no missing value
df = pd.read_csv('../../DataSets/TPSjan2021/train.csv')
df_test = pd.read_csv('../../DataSets/TPSjan2021/test.csv')
cols = df.columns
data = df[cols[1:-1]]
target = df['target']

x_train, y_train = data[:200000], target[:200000]
x_valid, y_valid = data[200000:250000], target[200000:250000]
x_test, y_test = data[250000:], target[250000:]


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(14,)))
model.add(Dropout(.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')
history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_valid, y_valid))

loss = history.history['loss']
val_loss = history.history['val_loss']

plot_history(loss, val_loss, 20, 'loss')

data_test = df_test[cols[1:-1]]
test_submit = df_test[['id']].copy()
test_submit['target'] = model.predict(data_test)
test_submit.to_csv('test_submit_01.csv', index=False)