import pandas as pd
import silence_tensorflow.auto
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tuto_utils.util_func import plot_history

train = pd.read_csv('../../DataSets/TPSfeb2021/train.csv')
test = pd.read_csv('../../DataSets/TPSfeb2021/test.csv')


def process_data(train, test):
    cols = list(train.columns)
    data_cat = train[cols[1:11]]
    data_cat = pd.get_dummies(data_cat)
    print(data_cat.shape)
    data_num = train[cols[11:-1]]
    y_train = train['target']
    x_train = data_cat.join(data_num)

    cols = list(test.columns)
    data_cat = test[cols[1:11]]
    data_cat = pd.get_dummies(data_cat)
    print(data_cat.shape)
    data_num = test[cols[11:]]
    x_test = data_cat.join(data_num)

    return x_train, y_train, x_test


x_train, y_train, x_test = process_data(train, test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

def run_prediction():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(70,)))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(x_train, y_train, epochs=20, batch_size=128)

    prediction = model.predict(x_test)
    test_prediction = test[['id']].join(prediction)
    test_prediction.to_csv('test_prediction.csv')