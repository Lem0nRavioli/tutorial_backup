import numpy as np

# lazy solving for 2x2 matrice
a = (0, 0)
b = (0, 1)
c = (1, 0)
d = (1, 1)


# Lazy function with mathematical presolving
def get_2x2_inv(A):
    U = np.zeros((2,2))
    Aa = A[a]
    Ab = A[b]
    Ac = A[c]
    Ad = A[d]

    U[a] = ((Ab * Ac) / (Aa * (Ad - Ab * Ac))) + 1
    U[b] = - Ab / (Ad - (Ab * Ac))
    U[c] = - Ac / (Ad - (Ab * Ac))
    U[d] = Aa / (Ad - (Ab * Ac))

    return U


M_test = np.array([[1, 3],[2, 7]])
print(M_test)
print("inverse of M_test")
print(get_2x2_inv(M_test))