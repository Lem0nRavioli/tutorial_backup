import matplotlib.pyplot as plt
import numpy as np


def show_pic(pic, color=False):
    if color:
        plt.imshow(pic)
    else:
        plt.imshow(pic, cmap=plt.cm.binary)
    plt.show()


def plot_history(train, validation, epoch, metric='', show=True):
    plt.clf()
    plt.plot(range(1, epoch + 1), train, 'bo', label=f'Training {metric}')
    plt.plot(range(1, epoch + 1), validation, 'b', label=f'Validation {metric}')
    plt.title(f'Training and validation {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    if show:
        plt.show()


def plot_acc_loss(history, epochs):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_history(acc, val_acc, epochs, metric='Accuracy', show=False)
    plt.figure()
    plot_history(loss, val_loss, epochs, metric='Loss')


def to_one_hot(labels):
    num_cat = np.max(labels) + 1
    results = np.zeros((len(labels), num_cat))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


def normalize(train_data, test_data=None, sample_axis=0):
    data = np.array(train_data)
    mean = data.mean(axis=sample_axis)
    data -= mean
    std = data.std(axis=sample_axis)
    data /= std
    if test_data is not None:
        test_data = (test_data - mean) / std
        return data, test_data
    return data


def smooth_curve(points, factor=.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def train_k_fold(model, data, k):
    num_validation_samples = len(data) // k
    np.random.shuffle(data)

    validation_scores = []
    for fold in range(k):
        validation_data = data[num_validation_samples * fold:num_validation_samples * (fold + 1)]
        training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]
        model.train(training_data)
        validation_score = model.evaluate(validation_data)
        validation_scores.append(validation_score)

    return np.average(validation_scores)


def show_conv(model, x_test, first=0, second=7, third=26, convnum=1, laynum=4):
    """ Not supposed to be used as it is, saving the template """
    from tensorflow import keras
    f, axarr = plt.subplots(3, laynum)
    FIRST_IMAGE = first
    SECOND_IMAGE = second
    THIRD_IMAGE = third
    CONVOLUTION_NUMBER = convnum
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    for x in range(0, laynum):
        f1 = activation_model.predict(x_test[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[0, x].grid(False)
        f2 = activation_model.predict(x_test[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[1, x].grid(False)
        f3 = activation_model.predict(x_test[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[2, x].grid(False)

    plt.show()