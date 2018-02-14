#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing
from multiprocessing import Pool
import threading
from time import time

def dot(a, b):
    out = np.zeros((a.shape[0], b.shape[1]))
    for i in xrange(a.shape[0]):
        for j in xrange(b.shape[1]):
            for item in zip(a[i], b[:, j]):
                out[i, j] += item[0] * item[1]
    return out

def dot_i(args):
    a, b, i = args
    return [sum(item[0] * item[1] for item in zip(a[i], b[:, j])) for j in xrange(b.shape[1])]

def dot_ij(args):
    a, b, i, j = args
    return sum(item[0] * item[1] for item in zip(a[i], b[:, j]))
    
def dot_multiprocessing(a, b, processes=multiprocessing.cpu_count()):
    if a.shape[0] < b.shape[1]:
        return np.transpose(dot_multiprocessing(b.T, a.T, processes))
    pool = multiprocessing.Pool(processes)
    if a.shape[0] >= processes:
        return np.asarray(pool.map(dot_i, [(a, b, i) for i in xrange(a.shape[0])]))
    out = np.empty((a.shape[0], b.shape[1]))
    for i in xrange(a.shape[0]):
        out[i, :] = np.asarray(pool.map(dot_ij, [(a, b, i, j) for j in xrange(b.shape[1])]))
    return out

def dot_threading(a, b, n_threads=multiprocessing.cpu_count()):
    if a.shape[0] < b.shape[1]:
        return np.transpose(dot_threading(b.T, a.T, n_threads))
    out = np.empty((a.shape[0], b.shape[1]))
    sem = threading.Semaphore(n_threads)
    if a.shape[0] >= n_threads:
        for i in xrange(a.shape[0]):
            def lam_f(*args):
                for j in xrange(b.shape[1]):
                    out[i, j] = sum(item[0] * item[1] for item in zip(a[i], b[:, j]))
                sem.release()
            sem.acquire()
            threading.Thread(target=lam_f).start()
    else:
        for i in xrange(a.shape[0]):
            for j in xrange(b.shape[1]):
                def lam_f(*args):
                    out[i, j] = sum(item[0] * item[1] for item in zip(a[i], b[:, j]))
                    sem.release()
                sem.acquire()
                threading.Thread(target=lam_f).start()
    for i in xrange(n_threads):
        sem.acquire()
    return out

if __name__ == '__main__':
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)
    
    t = time()
    dot(A, B)
    t = time() - t
    print t
    
    t = time()
    dot_multiprocessing(A, B)
    t = time() - t
    print t
    
    t = time()
    dot_threading(A, B)
    t = time() - t
    print t
