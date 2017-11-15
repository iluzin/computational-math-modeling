#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from sympy import Matrix, lambdify, solve, symbols

def func(xy, t0, *args):
    k1, k1m, k2, k3, k3m = args
    z = 1 - xy[0] - xy[1]
    return k1 * z - k1m * xy[0] - k3 * xy[0] + k3m * xy[1] - k2 * z ** 2 * xy[0], k3 * xy[0] - k3m * xy[1]

def plot_single_param(*args, **kwargs):
    for key, value in kwargs.iteritems():
        exec '{} = value'.format(key)
    exprs, J, k, x, y = args
    exprs += J.det(), J.trace()
    exprs = list(exprs)
    x_range=np.linspace(0, 1, 1e3)
    for i in xrange(-2, 0):
        exprs[i] = exprs[i].subs(y, exprs[0]).subs(k[1], exprs[1])
    for i, expr in enumerate(exprs):
        exprs[i] = expr.subs(k[2], k2).subs(k[-1], k1m).subs(k[3], k3).subs(k[-3], k3m)
    funcs = [lambdify(x, expr) for expr in exprs]
    det_array = [] 
    det_list = list(funcs[-2](x_range))
    det_points = []
    trace_array = []
    trace_list = list(funcs[-1](x_range))
    trace_points = []
    for i in xrange(1, len(x_range)):
        if det_list[i] * det_list[i - 1] <= 0:
            det_array.append(x_range[i])
            det_points.append(x_range[i - 1])
            det_points[-1] -= det_list[i - 1] * (x_range[i] - x_range[i - 1]) / (det_list[i] - det_list[i - 1])
        if trace_list[i] * trace_list[i - 1] <= 0:
            trace_array.append(x_range[i])
            trace_points.append(x_range[i - 1])
            trace_points[-1] -= trace_list[i - 1] * (x_range[i] - x_range[i - 1]) / (trace_list[i] - trace_list[i - 1])
    det_array = np.asarray(det_array)
    det_points = np.asarray(det_points)
    trace_array = np.asarray(trace_array)
    trace_points = np.asarray(trace_points)
    plt.grid(True)
    plt.plot(funcs[1](x_range), x_range, color='g', label='$x_{k_1}$') 
    plt.plot(funcs[1](x_range), funcs[0](x_range), color='b', label='$y_{k_1}$')
    plt.plot(funcs[1](trace_points), trace_points, color='r', linestyle='', marker='x')
    plt.plot(funcs[1](trace_points), funcs[0](trace_points), color='r', linestyle='', marker='o')
    plt.plot(funcs[1](det_points), det_points, color='k', linestyle='', marker='x')
    plt.plot(funcs[1](det_points), funcs[0](det_points), color='k', linestyle='', marker='o')
    plt.legend()
    plt.title(u'Однопараметрический анализ')
    plt.xlabel('$k_1$')
    plt.xlim(0, max(funcs[1](det_array)))
    plt.ylabel('x, y')
    plt.ylim(det_array.min(), max(det_array.max(), max(funcs[0](det_array))))
    plt.show()

def plot_double_param(*args, **kwargs):
    for key, value in kwargs.iteritems():
        exec '{} = value'.format(key)
    eqs, exprs, J, k, x, y = args
    exprs += J.det(), J.trace()
    x_range=np.linspace(0, 1, 1e3)
    plt.grid(True)
    sol = solve(exprs[-2].subs(y, exprs[0]), k[1])
    k2_sol = solve(sol[0] - exprs[1], k[2])[0]
    k2_f = lambdify((x, k[-1], k[3], k[-3]), k2_sol, 'numpy')
    k1_sol = exprs[1].subs(k[2], k2_sol)
    k1_f = lambdify((x, k[-1], k[3], k[-3]), k1_sol, 'numpy')
    plt.plot(k1_f(x_range, k1m, k3, k3m), k2_f(x_range, k1m, k3, k3m), color='c', linestyle='--', label=u'Линия кратности')
    sol = solve(exprs[-1].subs(y, exprs[0]), k[1])
    k2_sol = solve(sol[0] - exprs[1], k[2])[0]
    k2_f = lambdify((x, k[-1], k[3], k[-3]), k2_sol, 'numpy')
    k1_sol = exprs[1].subs(k[2], k2_sol)
    k1_f = lambdify((x, k[-1], k[3], k[-3]), k1_sol, 'numpy')
    plt.plot(k1_f(x_range, k1m, k3, k3m), k2_f(x_range, k1m, k3, k3m), color='m', label=u'Линия нейтральности')
    plt.legend()
    plt.xlabel('$k_1$')
    plt.xlim(0, 0.25)
    plt.ylabel('$k_2$')
    plt.ylim(0, 4)
    plt.show()
    for y_sol in solve(eqs[0], y):
        lam_f = lambdify((x, k[1], k[-1], k[2], k[3], k[-3]), y_sol, 'numpy')
        plt.plot(x_range, lam_f(x_range, k1, k1m, k2, k3, k3m), color='tab:orange', linestyle='--')
    plt.plot([], [], color='tab:orange', linestyle='--', label='x\'(t) = 0')
    for y_sol in solve(eqs[1], y):
        lam_f = lambdify((x, k[1], k[-1], k[2], k[3], k[-3]), y_sol, 'numpy')
        plt.plot(x_range, lam_f(x_range, k1, k1m, k2, k3, k3m), color='tab:purple', linestyle='--')
    plt.plot([], [], color='tab:purple', linestyle='--', label='y\'(t) = 0')
    plt.legend()
    funcs = [lambdify((x, y, k[1], k[-1], k[2], k[3], k[-3]), eq) for eq in eqs]
    Y, X = np.mgrid[0:1:1000j, 0:1:1000j]
    U = funcs[0](X, Y, k1, k1m, k2, k3, k3m)
    V = funcs[1](X, Y, k1, k1m, k2, k3, k3)
    plt.streamplot(X, Y, U, V, density=(2.5, 0.8), color=np.hypot(U, V))
    plt.title(u'Фазовый портрет системы')
    plt.xlabel('x')
    plt.xlim(0, 1)
    plt.ylabel('y')
    plt.ylim(0, 1)
    plt.show()
    t_range = np.linspace(0, 1e3, len(x_range))
    sol = integrate.odeint(func, (0.38, 0.61), t_range, args=(k1, k1m, k2, k3, k3m))
    plt.grid(True)
    plt.plot(t_range, sol[:, 0], color='y', label='x(t)')
    plt.plot(t_range, sol[:, 1], color='k', label='y(t)')
    plt.legend()
    plt.xlabel('t')
    plt.xlim(t_range.min(), t_range.max())
    plt.ylabel('x, y')
    plt.ylim(0, 1)
    plt.show()

if __name__ == '__main__':
    k = [None] * 7
    k[1], k[-1], k[2], k[3], k[-3], x, y = symbols('k1 k1m k2 k3 k3m x y')
    dxdt, dydt = func((x, y), 0, k[1], k[-1], k[2], k[3], k[-3])
    sol = solve((dxdt, dydt), y, k[1])
    A = Matrix([dxdt, dydt])
    J = A.jacobian((x, y))
    for coef in (0.001, 0.005, 0.01, 0.015, 0.02):
        plot_single_param(sol[0], J, k, x, y, k1=1, k1m=coef, k2=2, k3=0.0032, k3m=0.002)
    for coef in (0.0005, 0.001, 0.002, 0.003, 0.004):
        plot_single_param(sol[0], J, k, x, y, k1=1, k1m=0.01, k2=2, k3=0.0032, k3m=coef)
    plot_double_param(tuple(A), sol[0], J, k, x, y, k1=0.035, k1m=0.01, k2=0.4, k3=0.0032, k3m=0.002)
