#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

EPS = 1e-6

def area(a, b, c):
    return np.linalg.norm(np.cross(b - a, c - a)) / 2

def intersect(a, b, c, d):
    res = all(max(min(i, j), min(k, l)) - min(max(i, j), max(k, l)) < EPS for i, j, k, l in zip(a, b, c, d))
    res &= np.dot(np.cross(b - a, c - a), np.cross(b - a, d - a)) < EPS
    res &= np.dot(np.cross(d - c, a - c), np.cross(d - c, b - c)) < EPS
    return res

def process(a, b, c, p, q):
    coef = np.cross(b - a, c - a)
    intercept = -coef.dot(a)
    res = False
    if np.abs(coef.dot(p - q)) < EPS:
        if np.abs(coef.dot(p) + intercept) < EPS:
            res |= area(a, b, p) + area(b, c, p) + area(c, a, p) - area(a, b, c) < EPS
            res |= intersect(a, b, p, q) or intersect(b, c, p, q) or intersect(c, a, p, q)
    else:
        mu = float(coef.dot(p) + intercept) / coef.dot(p - q)
        if -EPS < mu and mu < 1 + EPS:
            m = p + mu * (q - p)
            res |= area(a, b, m) + area(b, c, m) + area(c, a, m) - area(a, b, c) < EPS
    return res

if __name__ == '__main__':
    trianglePairs = []
    
    # 1
    p1 = [0, 0, 0]
    p2 = [0, 5, 0]
    p3 = [6, 5, 0]
    tr1 = [p1, p2, p3]
    q1 = [1, 4, 0]
    q2 = [2, 4, 0]
    q3 = [2, 3, 0]
    tr2 = [q1, q2, q3]
    trianglePairs.append((tr1, tr2))
    
    # 2
    p1 = [-1, 0, 0]
    p2 = [0, -1, 0]
    p3 = [0, 0, 0]
    tr1 = [p1, p2, p3]
    q1 = [0, 0, 0]
    q2 = [0, 3, 0]
    q3 = [5, 0, 0]
    tr2 = [q1, q2, q3]
    trianglePairs.append((tr1, tr2))
    
    # 3
    p1 = [-1, 0, 0]
    p2 = [0, 2, 0]
    p3 = [0, 0, 0]
    tr1 = [p1, p2, p3]
    q1 = [0, 0, 0]
    q2 = [5, 0, 0]
    q3 = [0, 4, 0]
    tr2 = [q1, q2, q3]
    trianglePairs.append((tr1, tr2))
    
    # 4
    p1 = [0, 0, 0]
    p2 = [0, 2, 0]
    p3 = [1, 0, 0]
    tr1 = [p1, p2, p3]
    q1 = [0, -1, 0]
    q2 = [0, 3, 0]
    q3 = [7, -1, 0]
    tr2 = [q1, q2, q3]
    trianglePairs.append((tr1, tr2))
    
    # 5
    p1 = [0, 0, 0]
    p2 = [0, 4, 0]
    p3 = [4, 0, 0]
    tr1 = [p1, p2, p3]
    q1 = [1, 2, 0]
    q2 = [1, 1, -3]
    q3 = [0.5, 2, -2]
    tr2 = [q1, q2, q3]
    trianglePairs.append((tr1, tr2))
    
    # 6
    p1 = [0, 0, 0]
    p2 = [0, 4, 0]
    p3 = [4, 0, 0]
    tr1 = [p1, p2, p3]
    q1 = [4, 0, 0]
    q2 = [1, 1, -3]
    q3 = [0.5, 2, -2]
    tr2 = [q1, q2, q3]
    trianglePairs.append((tr1, tr2))
    
    # 7
    p1 = [0, 0, 0]
    p2 = [0, 4, 0]
    p3 = [4, 0, 0]
    tr1 = [p1, p2, p3]
    q1 = [1, 2, 0]
    q2 = [2, 1, 0]
    q3 = [0.5, 2, -2]
    tr2 = [q1, q2, q3]
    trianglePairs.append((tr1, tr2))
    
    # 8
    p1 = [0, 0, 0]
    p2 = [0, 4, 0]
    p3 = [4, 0, 0]
    tr1 = [p1, p2, p3]
    q1 = [-1, 2, 2]
    q2 = [0, 2, 2]
    q3 = [0, 0, -2]
    tr2 = [q1, q2, q3]
    trianglePairs.append((tr1, tr2))
    
    # 9
    p1 = [0, 0, 0]
    p2 = [0, 4, 0]
    p3 = [5, 0, 0]
    tr1 = [p1, p2, p3]
    q1 = [1, 1, 2]
    q2 = [5, 6, -2]
    q3 = [3, -4, -1]
    tr2 = [q1, q2, q3]
    trianglePairs.append((tr1, tr2))
    
    for item in trianglePairs:
        tr1, tr2 = np.asarray(item)
        p1, p2, p3 = tr1
        q1, q2, q3 = tr2
        print process(p1, p2, p3, q1, q2) or process(p1, p2, p3, q2, q3) or process(p1, p2, p3, q3, q1) or process(q1, q2, q3, p1, p2) or process(q1, q2, q3, p2, p3) or process(q1, q2, q3, p3, p1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.add_collection3d(Poly3DCollection(item, facecolors=['yellow', 'blue']))
        ax.set_xlim(min(p1[0], p2[0], p3[0], q1[0], q2[0], q3[0]), max(p1[0], p2[0], p3[0], q1[0], q2[0], q3[0]))
        ax.set_ylim(min(p1[1], p2[1], p3[1], q1[1], q2[1], q3[1]), max(p1[1], p2[1], p3[1], q1[1], q2[1], q3[1]))
        ax.set_zlim(min(p1[2], p2[2], p3[2], q1[2], q2[2], q3[2]), max(p1[2], p2[2], p3[2], q1[2], q2[2], q3[2]))
        plt.show()
