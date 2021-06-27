import numpy as np

# lazy solving for 2x2 matrice
a = (0, 0)
b = (0, 1)
c = (1, 0)
d = (1, 1)
M_test = np.array([[1, 3],[2, 7]])
A_test = np.array([[1, 3, 4], [2, 4, 1], [1, 2, 3]])
A_reduced = np.array([[4, 1], [2, 3]])

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


def test_1():
    print(M_test)
    print("inverse of M_test")
    print(get_2x2_inv(M_test))


# get Mij for a 3x3 matrice
def get_reduced_matrix(A, i, j):
    M = np.delete(A, i, 0)
    M = np.delete(M, j, 1)
    return M


# M = R2
def get_reduced_det(M, i, j):
    sign = (-1) ** (i + j)
    return sign * (M[0, 0] * M[1, 1] - (M[0, 1] * M[1, 0]))


# get cofactor matrix of a 3x3 A
def get_cofactor_matrix(A):
    C = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            M = get_reduced_matrix(A, i, j)
            C[i, j] = get_reduced_det(M, i, j)
    return C


# get adjugate matrix of 3x3 A
def get_adjugate_matrix(A):
    return get_cofactor_matrix(A).T


# det A for 3x3 based on det(A) = sum(a1j*sign(a1j)M1j = sum(a1jc1j)
def get_det_matrix(A):
    det = 0
    C = get_cofactor_matrix(A)
    for j in range(3):
        det += A[0, j] * C[0, j]
    return det


# get inverse of 3x3 matrix
def get_inverse_matrix(A):
    A_inv = (1/get_det_matrix(A)) * get_adjugate_matrix(A)
    return A_inv


# print(A_test)
# print(get_cofactor_matrix(A_test))
# print(get_adjugate_matrix(A_test))
# print(A_reduced)
# print(get_reduced_det(A_reduced, 0, 0))
A_inv = get_inverse_matrix(A_test)
print(A_test)
print(A_inv)
print(np.dot(A_test, A_inv))