import numpy as np
cimport numpy as np

def inv_cython(np.ndarray a):
    cdef float det = np.linalg.det(a)
    cdef np.ndarray out = np.zeros((a.shape[0], a.shape[1]))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            M = np.delete(np.delete(a, i, 0), j, 1)
            out[j, i] = np.linalg.det(M) / det
    return out
