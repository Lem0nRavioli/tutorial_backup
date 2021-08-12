import numpy as np

m = 3000
n = 280
it = 100000
a = 0.001
check_on = it / 20

th = np.random.rand(n, 1)
X = np.random.rand(m, n)
y = np.random.rand(m, 1)


def cost(x, y, th):
    return 1/(2 * len(x)) * (np.dot(x, th) - y)**2


def get_delta(x, y, th):
    return 1/len(x) * np.dot(x.T, (np.dot(x, th) - y))


def fit(x, y,  th, it=1000, see_th=False, check_on=100):
    th_witness = th.copy()
    for i in range(it):
        th -= a * get_delta(x, y, th)
        if i % check_on == 0:
            print('iteration:' + str(i) + ' cost: ' + str(sum(cost(x, y, th))))

    if see_th:
        print('initial theta')
        print(th_witness)
        print('reduced cost th')
        print(th)


fit(X, y, th, it=it, check_on=check_on)