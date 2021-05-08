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


def num_grad(f, delta=0.001):
    """ Calculating gradient of x column vector in respect to f a function that take column vector and return scalar """
    def df(x):
        gradient_x = np.zeros(x.shape)
        for i in range(len(x)):
            delta_x = np.zeros(x.shape)
            delta_x[i] = delta
            grad = (f(x + delta_x) - f(x - delta_x)) / (2 * delta)
            gradient_x[i] = grad
        return gradient_x
    return df


def minimize(f, x0, step_size_fn, max_iter):
    """ Finding derivative using the def above, as - small + small to find slope """
    df = num_grad(f, delta=.001)
    return gd(f, df, x0, step_size_fn, max_iter)


######################################################################################################################
# SVM HINGE #

def hinge(v):
    return np.where(v < 1, 1 - v, 0)

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    return hinge(y * (np.dot(th.T, x) + th0))

# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    total_loss = np.sum(hinge_loss(x, y, th, th0)) / x.shape[-1]
    regul = lam * (np.linalg.norm(th) ** 2)
    return total_loss + regul

# SVM HINGE DERIVATIVE #

# Returns the gradient of hinge(v) with respect to v.
def d_hinge(v):
    return np.where(v < 1, -1, 0)

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    return d_hinge(y * (np.dot(th.T, x) + th0)) * y * x

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    return d_hinge(y * (np.dot(th.T, x) + th0)) * y

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    total_loss = np.sum(d_hinge_loss_th(x, y, th, th0)) / x.shape[-1]
    regul = lam * th * 2
    return total_loss + regul

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    return np.sum(d_hinge_loss_th0(x, y, th, th0)) / x.shape[-1]

# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(X, y, th, th0, lam):
    g_th = d_svm_obj_th(X, y, th, th0, lam)
    g_th0 = d_svm_obj_th0(X, y, th, th0, lam)
    return np.vstack((g_th, g_th0))


# not mine but smart one
def batch_svm_min(data, labels, lam):
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    init = np.zeros((data.shape[0] + 1, 1))

    def f(th):
      return svm_obj(data, labels, th[:-1, :], th[-1:,:], lam)

    def df(th):
      return svm_obj_grad(data, labels, th[:-1, :], th[-1:,:], lam)

    x, fs, xs = gd(f, df, init, svm_min_step_size_fn, 10)
    return x, fs, xs


""" MINE but less efficient // exactly same behavior """

def batch_svm_min(data, labels, lam):
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    th = np.zeros((data.shape[0] + 1, 1))
    fs = []
    ths = []
    fs.append(svm_obj(data, labels, th[:-1], th[-1:], lam))
    ths.append(th)
    for i in range(10):
        th = th - svm_min_step_size_fn(i+1) * svm_obj_grad(data, labels, th[:-1], th[-1:], lam)
        fs.append(svm_obj(data, labels, th[:-1], th[-1:], lam))
        ths.append(th)
    return np.array(th), np.array(fs), np.array(ths)