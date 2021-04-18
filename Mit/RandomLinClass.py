import numpy as np

D = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([3, 12])


def random_lin_class(D, y, k):
    min_loss = np.inf
    theta_least = None
    for i in range(k):
        theta_zero = np.random.uniform(-100, 100, 1)
        theta = np.random.uniform(-10, 10, len(D[0]))
        loss = np.sum(np.abs(np.dot(theta, D.T) + theta_zero - y)) / len(D)
        if loss < min_loss:
            min_loss = loss
            theta_least = (theta, theta_zero)

    return theta_least, min_loss


thetas, loss = random_lin_class(D, y, 10000)
print(thetas)
print(loss)