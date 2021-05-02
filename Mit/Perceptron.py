""" data : d*n
    labels : 1 * n
    T : iterations
    hook : graph function (non-imp)
    return tuple of (th, th0) th.shape = (d,1), th0.shape = (1,1)"""

import numpy as np


def averaged_perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    d, n = data.shape
    th = np.zeros((d)).reshape((d,1)).astype('float64')
    th0 = np.zeros((1,1)).astype('float64')
    ths = th.copy(); th0s = th0.copy()
    for i in range(T):
        for j in range(n):
            point = data[:, j:j+1]
            lab = labels[:, j:j+1]
            pred = np.sign(np.dot(th.T, point) + th0)
            if pred != lab:
                th0 += lab
                th += lab * point
            ths += th
            th0s += th0
    nt = n * T
    return (ths/nt, th0s/nt)


def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0 = learner(data_train, labels_train)
    total_correct = score(data_test, labels_test, th, th0)
    total_test = float(labels_test.shape[-1])
    acc = total_correct / total_test
    return acc


def xval_learning_alg(learner, data, labels, k):
    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)

    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])
        score_sum += eval_classifier(learner, data_train, labels_train,
                                              data_test, labels_test)
    return score_sum/k