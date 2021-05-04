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


def hinge_loss(margin_ref, x, y, th, th0):
    mar = margin(x, y, th, th0)
    hinger = lambda x: 1 - x/margin_ref if x < margin_ref else 0
    v_hinger = np.vectorize(hinger)
    return v_hinger(mar)


data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4
margin_ref = 2**.5 / 2

print(hinge_loss(margin_ref, data, labels, th, th0))