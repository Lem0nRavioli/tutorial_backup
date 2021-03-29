import pandas as pd
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tuto_utils.util_func import plot_acc_loss

data_train = pd.read_csv('../../DataSets/TPSmar2021/train.csv')

# print(data.nunique())
# print(data.describe())


def pipeline(data_csv, train=True):
    cols = list(data_csv.columns)
    data_cat = data_csv[cols[1:20]].drop(['cat10'], axis=1)
    data_cat = pd.get_dummies(data_cat)
    if train:
        data_num = data_csv[cols[20:-1]]
        target = data_csv['target']
    else:
        data_num = data_csv[cols[20:]]
    data = data_cat.join(data_num).astype('float64')
    # print(data.shape)
    if train:
        return data, target
    return data


data, target = pipeline(data_train)
x_train, y_train = data[:200000], target[:200000]
x_valid, y_valid = data[200000:250000], target[200000:250000]
x_test, y_test = data[250000:],  target[250000:]


def model_0():
    model = Sequential([
        Dense(128, activation='relu', input_shape=data.shape[1:],
              kernel_regularizer=tf.keras.regularizers.L1(.001),
              activity_regularizer=tf.keras.regularizers.L2(.001)),
        BatchNormalization(),
        Dropout(.5),
        Dense(64, activation='relu'),
        Dropout(.5),
        Dense(32, activation='relu'),
        Dropout(.5),
        Dense(1, activation='sigmoid')
    ])

    return model


def evaluate_model(model):
    history = model.fit(x_train, y_train, epochs=50, batch_size=256, validation_data=(x_valid, y_valid))
    plot_acc_loss(history)


model = model_0()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# evaluate_model(model)

data_test = pd.read_csv('../../DataSets/TPSmar2021/test.csv')
x_test = pipeline(data_test, train=False)


def predict():
    model.fit(data, target, epochs=10, batch_size=256)
    prediction = model.predict(x_test)
    csv_prediction = data_test[['id']].copy()
    csv_prediction['target'] = prediction
    csv_prediction.to_csv('predic_01.csv', index=False)


predict()