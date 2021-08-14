import numpy as np

m = 3000
n = 280
it = 100000
a = 0.001
check_on = it // 20

th = np.random.rand(n, 1)
X = np.random.rand(m, n)
y = np.random.rand(m, 1)


def cost(x, y, th, l=0.):
    regul = l / (2 * (len(x))) * sum(th[1:] ** 2)
    return 1/(2 * len(x)) * (np.dot(x, th) - y)**2 + regul


def get_delta(x, y, th, l=0.):
    regul = l / len(x) * th
    regul[0] = 0
    return 1/len(x) * np.dot(x.T, (np.dot(x, th) - y)) + regul


def fit(x, y, th, l=0.,it=1000, see_th=False, check_on=100):
    th_witness = th.copy()
    for i in range(it):
        th -= a * get_delta(x, y, th, l=l)
        if i % check_on == 0:
            print('iteration:' + str(i) + ' cost: ' + str(sum(cost(x, y, th, l=l))))

    if see_th:
        print('initial theta')
        print(th_witness)
        print('reduced cost th')
        print(th)


fit(X, y, th, l=0.1, it=it, check_on=check_on)