import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = "../DataSets/jena_climate"
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values


# preprocess data
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]  # index 1 is temp in C
        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback, delay, 0, 200000, shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback, delay, 200001, 300000, shuffle=True, step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback, delay, 300001, None, shuffle=True, step=step, batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)


def show_me_data_shape():
    sample, target = next(train_gen)
    print(sample)
    print(target)
    print(sample.shape)
    print(target.shape)
    print(sample[:, -1, 1])


def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        if step % 10000 == 0:
            print(f"Processing step {step} to {step + 10000}...")
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]  # temp of last 10mn in 10 day period over batch_size samples (128x1)
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
    return np.mean(batch_maes)


# return MAE of .29 which apparently translate as a 2.57C kf average absolute error
# celsius_mae = evaluate_naive_method() * std[1]  # multiplyin by tempC std

# p236
