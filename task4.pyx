import numpy as np
cimport numpy as np

def invert_matrix_cython(np.ndarray a):
    cdef np.ndarray out = np.zeros(a.shape)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            M = np.delete(a, i, 0)
            M = np.delete(M, j, 1)
            out[j, i] = np.linalg.det(M)
    return out / np.linalg.det(a)