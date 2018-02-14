#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from scipy.integrate import odeint
from sympy import Matrix, lambdify, solve, symbols
import task1

def plot_A_dependece(k1val, k1mval, k2mval, k3_0val, alphaval):
    k2_func = lambdify((y, k_1, k1, k_2, k30, alpha), eqk2, 'numpy')
    k2_values = k2_func(y_range, k_1_value, k1val, k_2_value, k3_0val, alphaval)
    Xf = lambdify((y, k_1, k1, k_2, k30, alpha), x_fromSystem)
    Af=lambdify((x, y, k_1, k1, k2, k_2, k30, alpha), jacobianA)
    X = Xf(y_range, k1mval, k1val, k2mval, k3_0val, alphaval)
    A11 = Af(X, y_range, k1mval, k1val,k2_values, k2mval, k3_0val, alphaval)[0, 0]
    A22 = Af(X, y_range, k1mval, k1val,k2_values, k2mval, k3_0val, alphaval)[1, 1]
    tracaAf = lambdify((x,y,k_1, k1,k2, k_2, k30, alpha),traceA)
    traceA_value = tracaAf(X, y_range, k1mval, k1val,k2_values, k2mval, k3_0val, alphaval)
    min_a11 = A11[0]
    max_a22 = A22[0]
    for i in range(0,len(k2_values)):
        if(A11[i] > min_a11 and A11[i]<0):
            min_a11 = A11[i]
        if (A22[i] < max_a22 and A22[i] > 0):
            max_a22 = A22[i]
    print(abs(min_a11)/max_a22)
    line1, = plt.plot(k2_values,A11, 'r', linewidth=1, label='$a_{11}$')
    line2, = plt.plot(k2_values,A22, 'g', linewidth=1, label='$a_{22}$')
    line3, = plt.plot(k2_values,traceA_value, 'b', linewidth=1, label='$trace(A)$')
    plt.legend(loc=2)
    plt.xlabel('$k_2$')
    plt.ylabel('$A_values$')
    plt.xlim(0,0.08)
    plt.ylim(-1,1)
    plt.grid(True)
    plt.show()


def plot_turing_sustainability(D1, D2, k1val, km1val, km2val, k3_0val, alphaval):
    Di = (D1 * a22 + D2 * a11) ** 2 - 4 * detA_new * D1 * D2
    k_volna_1 = ((D1*a22 + D2*a11) + Di**(1/2))/(2*D1*D2)
    k_volna_2 = ((D1*a22 + D2*a11) - Di**(1/2)) / (2 * D1 * D2)
    k_volna_1_lamd = lambdify((y,k2, k_1, k1, k_2, k30, alpha), k_volna_1,'numpy')
    k_volna_2_lamd = lambdify((y,k2, k_1, k1, k_2, k30, alpha), k_volna_2, 'numpy')
    k2_func = lambdify((y, k_1, k1, k_2, k30, alpha), eqk2,'numpy')
    k2_values = k2_func(y_range, km1val,k1val, km2val, k3_0val, alphaval)
    Values_k_volna_1 = k_volna_1_lamd(y_range,k2_values, km1val,k1val, km2val, k3_0val, alphaval)
    Values_k_volna_2 = k_volna_2_lamd(y_range,k2_values, km1val, k1val, km2val, k3_0val, alphaval)
    line1, = plt.plot(Values_k_volna_1,k2_values, color='r',linestyle='-',linewidth=1,label ='$k_2$($k^2$)')
    line2, = plt.plot(Values_k_volna_2,k2_values, color='g',linestyle='-',linewidth =1)
    plt.legend(loc=2)
    plt.xlim((0,17))
    plt.ylim((0.034,0.07))
    plt.xlabel('$k^2$')
    plt.ylabel('$k_2$')
    plt.show()


def plot_B_usus(D1,D2,k1val, k1mval, k2val, k2mval, k3_0val,alphaval):
    a11_f = lambdify((y, k_1, k1, k2, k_2, k30, alpha), a11, 'numpy')
    a22_f = lambdify((y, k_1, k1, k2, k_2, k30, alpha), a22, 'numpy')
    a11_mass = a11_f(y_range, k1mval,k1val, k2val, k2mval, k3_0val,alphaval)
    a22_mass = a22_f(y_range,k1mval,k1val, k2val, k2mval, k3_0val,alphaval)
    detA_f =  lambdify((y, k_1, k1,k2, k_2, k30, alpha), detA_new, 'numpy')
    detA_mass = detA_f(y_range,k1mval,k1val, k2val, k2mval, k3_0val,alphaval)
    KPM = np.linspace(0, 6, 10000)
    DetB = detA_mass - (D1 * a22_mass + D2 * a11_mass) * KPM ** 2 + D1 * D2 * KPM ** 4
    for i in range(0,len(KPM)):
        if(DetB[i] == 0):
            print(KPM[i])
    line1, = plt.plot(KPM, DetB, color='g',linestyle='-', linewidth=1,label = '$\Delta B(k)$')
    plt.legend(loc=2)
    plt.xlabel('$k$')
    plt.ylabel('$\Delta B$')
    plt.xlim(0,3)
    plt.ylim(-2.5,2.5)
    plt.grid(True)
    plt.show()


def plot_B_eigenvalues_k(D1,D2,k1val, k1mval, k2val, k2mval, k3_0val,alphaval):
    k2_func = lambdify((y, k_1, k1, k_2, k30, alpha), eqk2, 'numpy')
    k2_values = k2_func(y_range, k_1_value, k1val, k_2_value, k3_0val, alphaval)
    a11_f = lambdify((y, k_1, k1, k2, k_2, k30, alpha), a11, 'numpy')
    a22_f = lambdify((y, k_1, k1, k2, k_2, k30, alpha), a22, 'numpy')
    a11_mass = a11_f(y_range, k1val, k1mval, k2_values, k2mval, k3_0val, alphaval)
    a22_mass = a22_f(y_range, k1val, k1mval, k2_values, k2mval, k3_0val, alphaval)
    detA_f = lambdify((y, k_1, k1, k2, k_2, k30, alpha), detA_new, 'numpy')
    detA_mass = detA_f(y_range, k1val, k1mval, k2_values, k2mval, k3_0val, alphaval)
    indoft = (np.abs(k2_values - k2val)).argmin()
    KPM = np.linspace(0, 6, 1000)
    traceA_new = a11_mass[indoft] + a22_mass[indoft]
    DetB = detA_mass[indoft] - (D1 * a22_mass[indoft] + D2 * a11_mass[indoft]) * KPM ** 2 + D1 * D2 * KPM ** 4
    traceB = traceA_new - (D1+D2)*KPM**2
    eigenvalues_1 = 0.5*(traceB + (traceB**2 - 4*DetB)**(1/2))
    eigenvalues_2 = 0.5*(traceB - (traceB ** 2 - 4 * DetB) ** (1 / 2))
    line1, = plt.plot(KPM, eigenvalues_1, 'y',color='b', linewidth=1, label='$\gamma(k^2)$')
    #line2, = plt.plot(KPM, eigenvalues_2, 'b', linewidth=1, label='$\gamma(k^2)$')
    plt.legend(loc=2)
    plt.xlabel('$k$')
    plt.xlim(0, 0.5)
    plt.ylim(-1, 1)
    plt.ylabel('$\gamma(k^2)$')
    plt.grid(True)
    plt.show()
    Di = (D1 * a22_mass[indoft] + D2 * a11_mass[indoft]) ** 2 - 4 * detA_mass[indoft] * D1 * D2
    kc = ((D1 * a22_mass[indoft] + D2 * a11_mass[indoft]) + Di ** (1 / 2)) / (2 * D1 * D2)
    print(kc)
    Lmin = np.pi / kc ** 0.5
    print "Lmin=", Lmin
    Lmin = 13
    L = np.ceil(10 * Lmin)
    print(L)
    NM = KPM * L / np.pi
    line1, = plt.plot(NM, eigenvalues_1,color='m', linewidth=1, label='$\gamma(n)$')
    plt.xlabel('$n$')
    #plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.ylabel('$\gamma$')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    k = [None] * 7
    k[1], k[-1], k[2], k[3], k[-3], x, y = symbols('k1 k1m k2 k3 k3m x y')
    dxdt, dydt = task1.func((x, y), 0, k[1], k[-1], k[2], k[3], k[-3])
    sol = solve((dxdt, dydt), y, k[1])
    A = Matrix([dxdt, dydt])
    J = A.jacobian((x, y))
    y_expr, k1_expr = sol[0]
    x_range = np.linspace(0, 1, 1e4)
    k1 = k[1]
    k_1 = k[-1]
    k2 = k[2]
    k_2 = k[-3]
    k3 = k[3]
    k30, alpha = symbols("k30 alpha")
    eq1 = k1 * (1 - x - y) - k_1 * x - x * y * k30 * (1 - y) ** alpha
    eq2 = k2 * (1 - x - y) ** 2 - k_2 * y ** 2 - x * y * k30 * (1 - y) ** alpha
    res = solve([eq1, eq2], x, k2)
    print res
    eqk2 = res[0][1]
    print eq2
    x_fromSystem = res[0][0]
    print x_fromSystem
    A = Matrix([eq1, eq2])
    #print A
    var_vector = Matrix([x, y])
    jacobianA = A.jacobian(var_vector)
    #print jacobianA
    detA = jacobianA.det()
    traceA = jacobianA.trace()
    print detA
    y_range = np.linspace(0, 1, 10000)
    #x_range = np.linspace(0, 1, 10000)
    #x_y_range = np.linspace(0, 1, 100)
    a11 = jacobianA[0, 0].subs(x, x_fromSystem)
    a22 = jacobianA[1, 1].subs(x, x_fromSystem)
    detA_new = detA.subs(x, x_fromSystem)
    alpha_value = 16.0
    k1_value = 0.03
    k_1_value = 0.01
    k2_value = 0.05
    k_2_value = 0.01
    k3_value = 10.0
    alpha_value_array = np.array([10.0, 15.0, 18.0, 20.0, 25.0])
    k3_value_array = np.array([1.0, 5.0, 10.0, 50.0, 100.0])
    #print alpha_value_array[0]
    #alpha_value = alpha_value_array[4]
    #k3_value = k3_value_array[4]
    print "alpha_value=", alpha_value
    print "k3_value=", k3_value
    #one_p_analysis(alpha_value_array[0], k3_value[1])
    #k2_points = plot_single_param(alpha_value, k3_value)
    #print "k2_points=", k2_points
    k2_value = 0.038
    D1 = 10
    D2 = 0.01
    #plot_A_dependece(k1_value, k_1_value, k_2_value, k3_value, alpha_value)
    #plot_turing_sustainability(D1, D2, k1_value, k_1_value, k_2_value, k3_value, alpha_value)
    #plot_B_usus(D1, D2, k1_value, k_1_value, k2_value, k_2_value, k3_value, alpha_value)
    plot_B_eigenvalues_k(D1, D2, k1_value, k_1_value, k2_value, k_2_value, k3_value, alpha_value)
