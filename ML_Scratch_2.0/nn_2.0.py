import numpy as np

m = 10
n = 5
labels = 1
it = 1000
a = 0.01
l = 0.01
check_on = it // 20


th = np.random.rand(n, 1)
X = np.random.rand(m, n)
y = np.round(np.random.rand(m, labels))
layers_shape = [X.shape[1], 5, 8, labels]
layers = []


def add_bias(arr):
    """ take an m x n matrix and return an m x n + 1 matrix 1 as first parameter"""
    return np.hstack((np.ones((arr.shape[0], 1), dtype=arr.dtype), arr))


def sigmoid(x):
    return 1/(1 + np.e**-x)


def cost(x, y_true, y_pred, layers, l=0.):
    regul = np.sum([np.sum(layer.weights[:,1:] ** 2) for layer in layers])
    return -1/len(x) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) + l/(2 * len(x)) * regul


class NnLayer:
    def __init__(self, input, output):
        """
        Weight in ouput x input + 1 (activation previous layer / inital input + bias)
        Adding bias post activation calculation, activation will be m x n + 1 form
        """
        self.weights = np.random.rand(output, input + 1)
        self.activation = None
        self.delta = None


for ind, layer in enumerate(layers_shape[:-1]):
    layers.append(NnLayer(layer, layers_shape[ind + 1]))


squish = sigmoid
X_bias = add_bias(X)
layers[0].activation = add_bias(squish(np.dot(layers[0].weights, X_bias.T)).T)
for ind, layer in enumerate(layers[1:]):
    layer.activation = add_bias(squish(np.dot(layer.weights, layers[ind].activation.T)).T)

layers[-1].activation = layers[-1].activation[:, 1:]  # remove bias from output layer

predict = layers[-1].activation


b_layers = layers.copy()
b_layers.reverse()
b_layers[0].delta = b_layers[0].activation - y
print(b_layers[0].delta)