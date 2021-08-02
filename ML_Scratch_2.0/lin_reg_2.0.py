import numpy as np

m = 5
n = 8


r_matrice = np.random.rand(m, n)
r_theta = np.random.rand(n, 1)
r_y = np.random.rand(m, 1)

# print(r_matrice)
# print(r_theta)
# print(r_y)
#
# r_dot = np.dot(r_matrice, r_theta.T)
#
# print(r_dot)


def cost(x, y, th):
    return 1/(2 * len(x)) * sum(np.dot(x, th) - y)


def get_delta(x, y, th):
    error = 1/len(x) * sum((np.dot(x, th) - y))  # add xij https://discuss.cloudxlab.com/t/derivative-of-mse-function/5030 )
    return 2 * error


print(cost(r_matrice, r_y, r_theta))