import numpy as np


def margin(x, y, th, th0):
    pred = np.dot(th.T, x) + th0
    norm_th = np.linalg.norm(th)  # np.sum(np.square(th)) ** .5
    return y * pred / norm_th


def signed_distance(x, th, th0):
    pred = np.dot(th.T, x) + th0
    norm_th = np.linalg.norm(th)  # np.sum(np.square(th)) ** .5
    return pred / norm_th


def sum_margin(x, y, th, th0):
    return np.sum(margin(x, y, th, th0))


def min_margin(x, y, th, th0):
    return np.min(margin(x, y, th, th0))


def max_margin(x, y, th, th0):
    return np.max(margin(x, y, th, th0))


data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
red_th = np.array([[1, 0]]).T
red_th0 = -2.5

print(sum_margin(data, labels, blue_th, blue_th0))
print(min_margin(data, labels, blue_th, blue_th0))
print(max_margin(data, labels, blue_th, blue_th0))