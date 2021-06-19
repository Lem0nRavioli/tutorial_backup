import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 100 * np.e ** (-x / 2)


def g(x):
    return 100 * (1 - np.e ** (-x / 2))

def plot_fg():
    result_f = []
    result_g = []
    for i in range(12):
        result_f.append(f(i))
        result_g.append(g(i))


    print(result_f)
    print(result_g)
    plt.plot(range(12), result_f)
    plt.plot(range(12), result_g)
    plt.legend(["F", "G"])
    plt.grid()
    plt.show()


def v(t):
    C = 4.0 * 10 ** -6  # in farads
    R = 10000
    return 20 * np.e ** (-t / (R * C))


print(v(.01))
print(v(.1))