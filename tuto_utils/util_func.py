import matplotlib.pyplot as plt
import numpy as np


def show_pic(pic, color=False):
    if color:
        plt.imshow(pic)
    else:
        plt.imshow(pic, cmap=plt.cm.binary)
    plt.show()


def plot_history(train, validation, epoch, metric=''):
    plt.clf()
    plt.plot(range(1, epoch + 1), train, 'bo', label=f'Training {metric}')
    plt.plot(range(1, epoch + 1), validation, 'b', label=f'Validation {metric}')
    plt.title(f'Training and validation {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()


def to_one_hot(labels):
    num_cat = np.max(labels) + 1
    results = np.zeros((len(labels), num_cat))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results