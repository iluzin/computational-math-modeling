#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fenics import *
from math import pi, cosh, sinh
from matplotlib import tri
import matplotlib.pyplot as plt
from mshr import *
import numpy as np

if __name__ == '__main__':
    mesh = generate_mesh(Rectangle(Point(0, 0), Point(5, 3)), 256)
    space = FunctionSpace(mesh, 'P', 1)
    tol = 1e-14
    u_bottom = Expression('sin(pi * x[0] / 5)', degree=1)
    u_right = Expression('sin(2 * pi * x[1] / 3)', degree=1)
    u_top = Expression('sin(3 * pi * x[0] / 5)', degree=1)
    u_left = Expression('sin(3 * pi * x[1] / 3)', degree=1)
    bc_left = DirichletBC(space, u_left, lambda x, b: b and near(x[0], 0, tol))
    bc_right = DirichletBC(space, u_right, lambda x, b: b and near(x[0], 5, tol))
    bc_bottom = DirichletBC(space, u_bottom, lambda x, b: b and near(x[1], 0, tol))
    bc_top = DirichletBC(space, u_top, lambda x, b: b and near(x[1], 3, tol))
    f = Expression("-sin(pi * x[0]) * sin(2 * pi * x[1])", degree=1)
    u = TrialFunction(space)
    v = TestFunction(space)
    a = dot(grad(u), grad(v)) * dx
    numerical_solution = Function(space)
    solve(a == f * v * dx, numerical_solution, [bc_left, bc_right, bc_bottom, bc_top])
    analytical_solution = Expression('-sin(pi * x[0]) * sin(2 * pi * x[1]) / (5 * pi * pi) + sinh(2 * pi * x[0] / 3) * sin(2 * pi * x[1] / 3) / sinh(10 * pi / 3) + sin(pi * x[1]) * (cosh(pi * x[0]) - cosh(pi * 5) * sinh(pi * x[0]) / sinh(pi * 5)) + sinh(3 * pi * x[1] / 5) * sin(3 * pi * x[0] / 5) / sinh(9 * pi / 5) + sin(pi * x[0] / 5) * (cosh(pi * x[1] / 5) - cosh(3 * pi / 5) * sinh(pi * x[1] / 5) / sinh(3 * pi / 5))', degree=2)
    l2_error = errornorm(analytical_solution, numerical_solution, 'L2')
    vertex_values_numerical = numerical_solution.compute_vertex_values(mesh)
    vertex_values_analytical = analytical_solution.compute_vertex_values(mesh)
    max_error = np.max(np.abs(vertex_values_numerical - vertex_values_analytical))
    print l2_error, max_error
    dims = mesh.geometry().dim()
    n_vertices = mesh.num_vertices()
    mesh_coordinates = mesh.coordinates().reshape((n_vertices, dims))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
    plt.figure(1)
    plt.title(u'Аналитическое решение')
    zfaces = np.asarray([analytical_solution(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='none')
    plt.colorbar()
    plt.savefig('figure1.pdf')
    plt.figure(2)
    plt.title(u'Численное решение')
    zfaces = np.asarray([numerical_solution(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='none')
    plt.colorbar()
    plt.savefig('figure2.pdf')
