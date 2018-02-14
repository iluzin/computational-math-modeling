#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Cython.Build import cythonize
from distutils.core import setup
import numpy as np
import multiprocessing
from multiprocessing import Pool
import threading
from time import time

def inv(a):
    det = np.linalg.det(a)
    out = np.empty(a.shape)
    for i in xrange(a.shape[0]):
        for j in xrange(a.shape[1]):
            M = np.delete(a, i, 0)
            M = np.delete(M, j, 1)
            out[j, i] = np.linalg.det(M) / det
    return out

def inv_single(args):
    a, i = args
    det = np.linalg.det(a)
    out = np.empty(a.shape[1])
    M = np.delete(a, i, 0)
    for j in xrange(a.shape[1]):
        out[j] = np.linalg.det(np.delete(M, j, 1)) / det
    return out

def inv_multiprocessing(a, processes=multiprocessing.cpu_count()):
    pool = multiprocessing.Pool(processes)
    out = np.asarray(pool.map(inv_single, [(a, i) for i in xrange(a.shape[0])]))
    return out.T

def inv_threading(a, n_threads=multiprocessing.cpu_count()):
    det = np.linalg.det(a)
    out = np.empty(a.shape)
    sem = threading.Semaphore(n_threads)
    for i in xrange(a.shape[0]):
        def lam_f(*args):
            M = np.delete(a, i, 0)
            for j in xrange(a.shape[1]):
                out[j, i] = np.linalg.det(np.delete(M, j, 1)) / det
            sem.release()
        sem.acquire()
        threading.Thread(target=lam_f).start()
    for i in xrange(n_threads):
        sem.acquire()
    return out

if __name__ == '__main__':
    A = np.random.rand(3, 3)
    
    print np.linalg.inv(A)
    
    t = time()
    print inv(A)
    t = time() - t
    print t
    
    t = time()
    print inv_threading(A)
    t = time() - t
    print t
    
    t = time()
    print inv_multiprocessing(A)
    t = time() - t
    print t
    
    setup(ext_modules=cythonize("task4.pyx"))
    import task4
    t = time()
    #print inv_cython(A)
    t = time() - t
    print t