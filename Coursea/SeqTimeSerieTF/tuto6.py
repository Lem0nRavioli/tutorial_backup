""" import from csv, process dataset into timeseries, use lstm """

import silence_tensorflow.auto
import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
from tuto_utils.util_func import windowed_dataset, show_me_lr, graph_evaluate


path = "../../DataSets/melbourne_temp/daily-min-temperatures.csv"

temp_dataset = []

with open(path, 'r') as file:
    data = csv.reader(file)
    next(data)
    for row in data:
        temp_dataset.append(float(row[1]))

series = np.array(temp_dataset)
time = np.array(range(len(series)))

# plt.figure(figsize=(10, 6))
# plot_series(time[:730], series[:730])
# plt.show()

split_time = 3285
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 30
batch_size = 128
shuffle_buffer_size = False

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
dataset_valid = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)


def basic_evaluate(model, epoch=100):
    model.fit(dataset, epochs=epoch)

    model.evaluate(dataset_valid)


def model_0():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    return model


def model_1():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='causal',
                               activation='relu'),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 200)
    ])

    return model


def model_2():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='causal',
                               activation='relu'),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 200)
    ])

    return model


def model_3():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='causal',
                               activation='relu'),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 200)
    ])

    return model


model = model_3()

""" gave us 1e-4 """
# show_me_lr(model, dataset, epochs=150, loss='mse', verbose=1, pltax=(1e-8, 1, 0, 150))
optimizer = tf.keras.optimizers.SGD(lr=1e-6, momentum=.9)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# basic_evaluate(model, epoch=150)

history = model.fit(dataset, epochs=200)
graph_evaluate(model, history, series, split_time, window_size)