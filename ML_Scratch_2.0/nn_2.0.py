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


def add_bias(arr):
    """ take an m x n matrix and return an m x n + 1 matrix 1 as first parameter"""
    return np.hstack((np.ones((arr.shape[0], 1), dtype=arr.dtype), arr))


def sigmoid(x): return 1/(1 + np.e**-x)


def sigmoid_derivative(x): return x * (1 - x)


def relu(x): return max(0, x)


def relu_derivative(x): return 1 if x > 0 else 0


def cost(x, y_true, y_pred, layers, l=0.):
    regul = np.sum([np.sum(layer.weights[:,1:] ** 2) for layer in layers])
    return -1/len(x) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) + l/(2 * len(x)) * regul


class NnLayer:
    def __init__(self, input, output, squish=relu, deriv=relu_derivative):
        """
        Weight in ouput x input + 1 (activation previous layer / inital input + bias)
        Adding bias post activation calculation, activation will be m x n + 1 form
        """
        self.weights = np.random.rand(output, input + 1)
        self.squish = np.vectorize(squish)
        self.deriv = np.vectorize(deriv)
        self.activation = None
        self.delta = None


def build_network(shape):
    layers = []
    for ind, layer in enumerate(shape[:-1]):
        layers.append(NnLayer(layer, shape[ind + 1]))
    layers[-1].squish = np.vectorize(sigmoid)
    layers[-1].deriv = np.vectorize(sigmoid_derivative)
    return layers


layers = build_network(layers_shape)
X_bias = add_bias(X)

# process input layer then hidden & output with for loop
layers[0].activation = add_bias(layers[0].squish(np.dot(layers[0].weights, X_bias.T)).T)
for ind, layer in enumerate(layers[1:]):
    layer.activation = add_bias(layer.squish(np.dot(layer.weights, layers[ind].activation.T)).T)

layers[-1].activation = layers[-1].activation[:, 1:]  # remove bias from output layer
predict = layers[-1].activation
error = np.mean(predict - y, axis=0, keepdims=True)
# print(error)
# print(layers[-1].weights.T)

layers[-1].delta = np.dot(layers[-1].weights.T, error).T * \
                   layers[-1].deriv(np.mean(layers[-1].activation, axis=0, keepdims=True))
# print(layers[-1].delta)
# print(np.dot(layers[-1].weights.T, error).T)
print(layers[-1].delta.shape)
for ind, layer in enumerate(reversed(layers)):
    if ind == 0: continue
    print(layer.weights.T.shape)
    print(layers[-ind].delta.T[1:].shape)
    print(np.mean(layer.activation, axis=0, keepdims=True)[:, 1:].shape)
    layer.delta = (np.dot(layer.weights.T, layers[-ind].delta.T[1:]) * \
                  layer.deriv(np.mean(layer.activation, axis=0, keepdims=True)[:,1:])).T
    print('passed layer ' + str(ind) + " delta shape: " + str(layer.delta.shape))


# b_layers = layers.copy()
# b_layers.reverse()
# delta_init = b_layers[0].activation - y
# b_layers[0].delta = np.mean(np.dot(b_layers[0].weights.T, delta_init.T).T * (b_layers[0].activation * (1 - b_layers[0].activation)), axis=0, keepdims=True)
# print(b_layers[0].delta.shape)
# print(b_layers[0].weights.shape)
# for ind, layer in enumerate(b_layers[1:]):
#     print(layer.weights.T.shape)
#     print(b_layers[ind].delta.T[1:].shape)
#     layer.delta = np.dot(layer.weights.T, b_layers[ind].delta.T[1:]) * np.mean(derivasquish(layer.activation), axis=0, keepdims=True)
#     print(layer.delta.shape)

# print(b_layers[0].delta)
