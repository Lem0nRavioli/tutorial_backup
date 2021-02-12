import pandas as pd
import silence_tensorflow.auto
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tuto_utils.util_func import plot_history

train = pd.read_csv('../../DataSets/TPSfeb2021/train.csv')
test = pd.read_csv('../../DataSets/TPSfeb2021/test.csv')


def process_data(train, test, fillmissing=True):
    cols = list(train.columns)
    data_cat = train[cols[1:11]]
    data_cat = pd.get_dummies(data_cat)
    data_num = train[cols[11:-1]]
    y_train = train['target']
    x_train = data_cat.join(data_num)

    cols = list(test.columns)
    data_cat = test[cols[1:11]]
    data_cat = pd.get_dummies(data_cat)
    data_num = test[cols[11:]]
    x_test = data_cat.join(data_num)
    # missing some G category in cat6 testing data, adding it manually
    if fillmissing:
        x_test.insert(23, "cat6_G", 0)
    else:
        x_train.drop(['cat6_G'], axis=1, inplace=True)
    return x_train, y_train, x_test


x_train, y_train, x_test = process_data(train, test, fillmissing=False)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
# print(train.cat6.unique())
# print(test.cat6.unique())
# print(x_train.columns)
# print(x_test.cat6_G.unique())


def run_prediction(earlystop=False):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(69,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    if earlystop:
        callbacks_list = [keras.callbacks.EarlyStopping(monitor='mse', patience=5),
                          keras.callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='val_loss', save_best_only=True)]

        model.compile(optimizer='rmsprop', loss='mse')
        model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=.2, callbacks=callbacks_list)
    else:
        model.compile(optimizer='rmsprop', loss='mse')
        model.fit(x_train, y_train, epochs=80, batch_size=128)

    prediction = model.predict(x_test)
    test_prediction = test[['id']].copy()
    test_prediction['target'] = prediction
    test_prediction.to_csv('test_prediction_dropmissing_5.csv', index=False)


run_prediction()