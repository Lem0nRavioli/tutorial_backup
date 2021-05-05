""" Part of :
https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week4/week4_homework/?child=first """


import numpy as np

"""f: a function whose input is an x, a column vector, and returns a scalar.
df: a function whose input is an x, a column vector, and returns a column vector representing the gradient of f at x.
x0: an initial value of xx, x0, which is a column vector.
step_size_fn: a function that is given the iteration index (an integer) and returns a step size.
max_iter: the number of iterations to perform
Our function gd returns a tuple:

x: the value at the final step
fs: the list of values of f found during all the iterations (including f(x0))
xs: the list of values of x found during all the iterations (including x0)"""


def gd(f, df, x0, step_size_fn, max_iter):
    fs = []
    xs = []
    x = x0.copy()
    fs.append(f(x0))
    xs.append(x0)
    for i in range(max_iter):
        x = x - step_size_fn(i) * df(x)
        fs.append(f(x))
        xs.append(x)
    return np.array(x), np.array(fs), np.array(xs)


""" Calculating gradient of x column vector in respect to f a function that take column vector and return scalar """


def num_grad(f, delta=0.001):
    def df(x):
        gradient_x = np.zeros(x.shape)
        for i in range(len(x)):
            delta_x = np.zeros(x.shape)
            delta_x[i] = delta
            grad = (f(x + delta_x) - f(x - delta_x)) / (2 * delta)
            gradient_x[i] = grad
        return gradient_x
    return df