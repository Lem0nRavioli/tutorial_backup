import numpy as np

m = 100
n = 50
it = 1000
a = 0.01
l = 0.01
check_on = it // 20

th = np.random.rand(n, 1)
X = np.random.rand(m, n)
y = np.round(np.random.rand(m, 1))

