""" sunspot dataset + lstm + conv1D """

import silence_tensorflow.auto
import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
from tuto_utils.util_func import windowed_dataset, plot_series, show_me_lr, model_forecast

path = '../../DataSets/sunspot/Sunspots.csv'

sunspots = []

with open(path, 'r') as csvfile:
    csv_data = csv.reader(csvfile, delimiter=',')
    next(csv_data)
    for row in csv_data:
        sunspots.append(float(row[2]))

series = np.array(sunspots)
time = range(len(series))
# plt.figure(figsize=(10, 6))
# plot_series(time, series)
# plt.show()

split_time = 3000 # 2550
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 30  # 20
batch_size = 64
shuffle_buffer_size = False

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)


def graph_evaluate(model, epoch=100, verbose=0, show_me_shape=False):
    history = model.fit(dataset, epochs=epoch, verbose=verbose)
    forecast = []
    forecast = model_forecast(model, series, window_size)
    if show_me_shape:
        print(forecast.shape)
        print(forecast[split_time - window_size: -1, -1, 0].shape)
        print(forecast[0])
        print("boop 0 0")
        print(forecast[0, 0])
        print("boop 0 0 0")
        print(forecast[0, 0, 0])
    # for time in range(len(series) - window_size):
    #     forecast.append(model.predict(series[time: time + window_size][np.newaxis]))

    forecast = forecast[split_time - window_size:-1, -1, 0]

    print("mae:", tf.keras.metrics.mean_absolute_error(x_valid, forecast).numpy())

    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plot_series(time_valid, forecast)
    plt.show()

    loss = history.history['loss']
    loss = loss[10: ]
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.show()


def basic_evaluate(model, epoch=100):
    model.fit(dataset, epochs=100)

    model.evaluate(windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size))


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
        tf.keras.layers.Lambda(lambda x: x * 400)
    ])

    return model


model = model_3()
optimizer = tf.keras.optimizers.SGD(lr=1e-7, momentum=.9)
# model.compile(loss=tf.keras.losses.Huber(), optimizer='adam', metrics=['mae'])
model.compile(loss='mse', optimizer=optimizer)
graph_evaluate(model, 300, 1)
