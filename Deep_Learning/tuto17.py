""" Hyperas (hyperopt for keras) : Automate hyperparameter optimization """

import silence_tensorflow.auto
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform, choice

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

import pandas as pd
import numpy as np


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
    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test)


def data():
    train = pd.read_csv('../DataSets/TPSfeb2021/train.csv')
    cols = list(train.columns)
    data_cat = train[cols[1:11]]
    data_cat = pd.get_dummies(data_cat)
    data_num = train[cols[11:-1]]
    x_train = data_cat.join(data_num)
    x_train.drop(['cat6_G'], axis=1, inplace=True)
    y = np.asarray(train['target'], dtype='float64')
    X = np.asarray(x_train, dtype='float64')

    x_train, y_train = X[:200000], y[:200000]
    x_valid, y_valid = X[200000:], y[200000:]
    return x_train, y_train, x_valid, y_valid


def create_model(x_train, y_train, x_valid, y_valid):
    model = Sequential()
    model.add(Dense(512, input_shape=(69,), activation='relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}, activation={{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense({{choice([128, 256])}}, activation={{choice(['relu', 'sigmoid'])}}))

    model.add(Dense(1))

    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, loss='mse')

    history = model.fit(x_train, y_train, batch_size={{choice([64, 128])}},
                        epochs={{choice([10, 15, 20])}}, validation_split=.1)

    validation_loss = np.amax(history.history['val_loss'])
    print('Best validation loss of epochs:', validation_loss)
    return {'loss': -validation_loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest,
                                          max_evals=10, trials=Trials())
    best_model.save('tpsfeb2021_hyperas01.h5')

    print("Best pertforming model chosen hyper-parameters:")
    print(best_run)
    X_train, y_train, X_valid, y_valid = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(X_valid, y_valid))


    train = pd.read_csv('../DataSets/TPSfeb2021/train.csv')
    test = pd.read_csv('../DataSets/TPSfeb2021/test.csv')

    X, y, x_test = process_data(train, test)

    prediction = best_model.predict(x_test)
    test_prediction = test[['id']].copy()
    test_prediction['target'] = prediction
    test_prediction.to_csv('test_prediction_dropmissing_7.csv', index=False)