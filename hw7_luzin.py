#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from hw1_luzin import EPS, process
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from random import random
import scipy.integrate as si
import scipy.optimize as so

class Solver(object):
    def __init__(self, lam, QR, c, eps, S, tau):
        self.lam = lam
        self.QR = QR
        self.c = c
        self.M = len(self.c)
        self.eps = eps
        self.S = S
        self.counter = 0
        self.T0 = so.fsolve(func_solve, np.zeros(self.M), args=(0, lam, QR, c, eps, S))
        self.T = copy.copy(self.T0)
        self.tau = tau
    
    def __call__(self):
        Tm=np.linspace((self.counter-1)*self.tau,self.counter*self.tau,2)
        self.counter+=1
        self.T=si.odeint(func_solve,self.T0,Tm,args=(self.lam,self.QR,self.c,self.eps,self.S,))
        self.T0=copy.copy(self.T[1])
        return self.T[1]

def func_solve(T, t, lam, QR, c, eps, S):
    M = len(c)
    rvalue = np.zeros(M)
    C0 = 5.67
    for i in range(M):
        for j in range(M):
            if i!=j:
                rvalue[i] -= lam[i, j] * S[i, j] * (T[i] - T[j])
        rvalue[i] -= eps[i] * S[i, i] * C0 * (T[i] / 100) ** 4
        rvalue[i] += QR[i](t)
        rvalue[i] /= c[i]
    return rvalue

def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    return shader

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    height = glutGet(GLUT_SCREEN_HEIGHT)
    width = glutGet(GLUT_SCREEN_WIDTH)
    gluPerspective(np.degrees(np.arctan2(1, 1)), width * 1e0 / height, 1e-3, 1e3)
    global eye
    global lat
    global lon
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    center = eye + (np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon))
    up = -np.sin(lat) * np.sin(lon), np.cos(lat), -np.sin(lat) * np.cos(lon)
    gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2])
    glUseProgram(program)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, triangles)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0, 3 * irange[len(list_of_vel)])
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    glutSwapBuffers()

def keyboard(key, x, y):
    global eye
    global lat
    global lon
    if key.upper() == 'W':
        eye += np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
    elif key.upper() == 'S':
        eye -= np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
    elif key.upper() == 'A':
        eye += np.cos(lon), 0, -np.sin(lon)
    elif key.upper() == 'D':
        eye += -np.cos(lon), 0, np.sin(lon)
    elif key.upper() == 'X':
        for i in range(len(list_of_vel)):
            luminance = normalize_heat(sol()[i])
            for j in xrange(irange[i], irange[i + 1]):
                pointcolor[j] = [luminance, luminance, luminance]
        glutPostRedisplay()
    elif key == '\x1b':
        glutLeaveMainLoop()

def motion(x, y):
    height = glutGet(GLUT_SCREEN_HEIGHT)
    width = glutGet(GLUT_SCREEN_WIDTH)
    center = width / 2, height / 2
    if (x, y) == center:
        return
    glutWarpPointer(center[0], center[1])
    global lat
    lat = min(max(lat - np.arcsin(y * 2e0 / height - 1), -np.arctan2(1, 0)), np.arctan2(1, 0))
    global lon
    lon -= np.arcsin(x * 2e0 / width - 1)

def triangulate(list_of_vel, list_of_tri):
    triangles = []
    irange = np.zeros(len(list_of_vel) + 1, dtype=np.int)
    for i in range(len(list_of_vel)):
        irange[i + 1] = irange[i] + len(list_of_tri[i])
        for el in list_of_tri[i]:
            triangle = np.array([list_of_vel[i][int(el[0])], list_of_vel[i][int(el[1])], list_of_vel[i][int(el[2])]])
            triangles.append(triangle)
    return np.array(triangles), irange

def normalize_heat(Te):
    B, D = 500, 500
    if Te < 50:
        return [np.exp(-Te ** 2 / B), np.exp(-Te ** 2 / B), np.exp(-Te ** 2 / B)]
    return [np.exp(-(Te - 100)**2 / D), np.exp(-(Te - 100) ** 2 / D), np.exp(-(Te - 100) ** 2 / D)]

def parse_wavefront(path):
    count = 0
    start = 0
    list_of_count_of_vel = []
    list_of_vel = []
    list_of_triang = []
    index = 0
    lst_vel = []
    lst_f = []
    total = 1
    count_of_v = 0
    for line in open(path, 'r'):
        values = line.split()
        if len(values) < 2:
            continue
        if(values[0] == '#' and values[1] == 'object' and count != 0):
            list_of_count_of_vel.append(count)
            list_of_vel.append(lst_vel)
            list_of_triang.append(lst_f)
            index = index + 1
            total = total + count_of_v
            count_of_v = 0
            count = 0
            lst_vel = []
            lst_f = []
        if (values[0] == '#' and values[1] == 'object' and count == 0):
            start = 1
        if(values[0] == 'f' and count == 0):
            start = 1
        if(start == 1 and values[0] == 'f'):
            count = count + 1
            lst_f.append([int(values[1]) - total, int(values[2]) - total, int(values[3]) - total])
        if (start == 1 and values[0] == 'v'):
            lst_vel.append([float(values[1]), float(values[2]), float(values[3])])
            count_of_v = count_of_v + 1
    list_of_vel.append(lst_vel)
    list_of_triang.append(lst_f)
    list_of_count_of_vel.append(count)
    return  list_of_count_of_vel, list_of_triang, list_of_vel

def projection(a, b, c, d):
    return ((d[0] - a[0]) * ((b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2])) -
            (d[1] - a[1]) * ((b[0] - a[0]) * (c[2] - a[2]) - (c[0] - a[0]) * (b[2] - a[2])) +
            (d[2] - a[2]) * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])))

def internal(a, b, c, d):
    eps = 10 ** -6
    val1=(b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1])
    val2=(b[0]-a[0])*(c[2]-a[2])-(c[0]-a[0])*(b[2]-a[2])
    val3=(b[1]-a[1])*(c[2]-a[2])-(c[1]-a[1])*(b[2]-a[2])
    if (abs(val1)>eps):
        u=((d[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(d[1]-a[1]))/val1
        v=-((d[0]-a[0])*(b[1]-a[1])-(b[0]-a[0])*(d[1]-a[1]))/val1
    elif (abs(val2)>eps):
        u=((d[0]-a[0])*(c[2]-a[2])-(c[0]-a[0])*(d[2]-a[2]))/val2
        v=-((d[0]-a[0])*(b[2]-a[2])-(b[0]-a[0])*(d[2]-a[2]))/val2
    elif (abs(val3)>eps):
        u=((d[1]-a[1])*(c[2]-a[2])-(c[1]-a[1])*(d[2]-a[2]))/val3
        v=-((d[1]-a[1])*(b[2]-a[2])-(b[1]-a[1])*(d[2]-a[2]))/val3
    else:
        return -1
    if (abs(u-0.5)<0.5+eps) and (abs(v-0.5)<0.5+eps) and (u+v<1+eps):
        return 1
    else:
        return 0

def intersect(p1, p2, p3, q1, q2, q3):
    if process(p1, p2, p3, q1, q2) or process(p1, p2, p3, q2, q3) or process(p1, p2, p3, q3, q1) or process(q1, q2, q3, p1, p2) or process(q1, q2, q3, p2, p3) or process(q1, q2, q3, p3, p1):
        if abs(projection(p1, p2, p3, q1)) > EPS:
            return False
        if abs(projection(p1, p2, p3, q2)) > EPS:
            return False
        if abs(projection(p1, p2, p3, q3)) > EPS:
            return False
        return True
    return False

def area_small(hx, hy):
    l1 = np.linalg.norm(hx)
    l2 = np.linalg.norm(hy)
    l3 = np.linalg.norm(hx - hy)
    p = (l1 + l2 + l3) / 2
    return 2 * (p * (p - l1) * (p - l2) * (p - l3)) ** 0.5

def area_triangle_intersection(p1, p2, p3, q1, q2, q3):
    if not intersect(p1, p2, p3, q1, q2, q3):
        return .0
    N = 100
    out = .0
    x = p3 - p1
    y = p2 - p1
    hx = x / N
    hy = y / N
    Sh = area_small(hx, hy)
    for i in range(N):
        for j in range(N - i):
            temp = 0
            pp = [p1 + i * hx + j * hy, p1 + (i + 1) * hx + j * hy, p1 + i * hx + (j + 1) * hy, p1 + (i + 1) * hx + (j + 1) * hy]
            for P in pp:
                if internal(q1, q2, q3, P) == 1:
                    temp += 0.25
            if j < N - i - 1:
                out += temp * Sh
            else:
                out += temp * Sh / 2
    return out

def area_triangle(lst, idx):
    l1 = np.linalg.norm(np.asarray(lst[idx[0]]) - lst[idx[1]])
    l2 = np.linalg.norm(np.asarray(lst[idx[0]]) - lst[idx[2]])
    l3 = np.linalg.norm(np.asarray(lst[idx[2]]) - lst[idx[1]])
    p = (l1 + l2 + l3) / 2
    return (p * (p - l1) * (p - l2) * (p - l3)) ** 0.5

def area_intersection(glob, locind, i, j):
    out = .0
    for ind1 in locind[i]:
        for ind2 in locind[j]:
            p1 = np.asarray(glob[i][ind1[0]])
            p2 = np.asarray(glob[i][ind1[1]])
            p3 = np.asarray(glob[i][ind1[2]])
            q1 = np.asarray(glob[j][ind2[0]])
            q2 = np.asarray(glob[j][ind2[1]])
            q3 = np.asarray(glob[j][ind2[2]])
            out += area_triangle_intersection(p1, p2, p3, q1, q2, q3)
    return out

def area_polygon(glob, locind, i):
    out = .0
    for el in locind[i]:
        out += area_triangle(glob[i], el)
    return out

if __name__ == '__main__':
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_MULTISAMPLE | GLUT_RGBA)
    glutEnterGameMode()
    glutDisplayFunc(display)
    glutIdleFunc(display)
    glutKeyboardFunc(keyboard)
    glutMotionFunc(motion)
    glutPassiveMotionFunc(motion)
    glutSetCursor(GLUT_CURSOR_NONE)
    height = glutGet(GLUT_SCREEN_HEIGHT)
    width = glutGet(GLUT_SCREEN_WIDTH)
    glutWarpPointer(width / 2, height / 2)
    glClearColor(.8, .8, 1, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    vertex = create_shader(GL_VERTEX_SHADER, """
        varying vec4 vertex_color;
        
        void main() {
            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            vertex_color = gl_Color;
        }
        """)
    fragment = create_shader(GL_FRAGMENT_SHADER, """
        varying vec4 vertex_color;
        
        void main() {
            gl_FragColor = vertex_color;
        }
        """)
    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    glUseProgram(program)
    list, list_of_tri, list_of_vel = parse_wavefront('model2.obj')
    M = len(list)
    triangles, irange = triangulate(list_of_vel, list_of_tri)
    triangles /= (2 * triangles.max())
    pointcolor = np.zeros((irange[len(list_of_vel)], 3, 3))
    for i in range(0, len(list)):
        m = random()
        k = random()
        for j in range(irange[i], irange[i+1]):
            pointcolor[j] = [k, m, .0]
    square = np.zeros((len(list_of_vel), len(list_of_vel)))
    square = np.array([[36.1672016977, 12.4639895929, 0.0, 0.0, 0.0], [12.4714188099, 99.4076906933, 0.0, 12.0674722235, 0.0], [0.0, 0.0, 12.3660143311, 3.06968959526, 3.06973444641], [0.0, 12.2865, 3.1506, 268.0, 0.0], [0.0, 0.0, 3.1320625, 0.0, 219.834109727]])
    #for i in xrange(len(list_of_vel)):
    #    for j in xrange(len(list_of_vel)):
    #        if i == j:
    #            square[i, j] = area_polygon(np.asarray(list_of_vel), list_of_tri, i)
    #        else:
    #            square[i, j] = area_intersection(np.asarray(list_of_vel), list_of_tri, i, j)
    eps = [0.1, 0.1, 0.05, 0.02, 0.05]
    c = [900, 900, 520, 1930, 520]
    lam = np.zeros((M, M))
    lam[0, 1] = lam[1, 0] = 240
    lam[1, 2] = lam[2, 1] = 130
    lam[2, 3] = lam[3, 2] = 118
    lam[3, 4] = lam[4, 3] = 10.5
    QR=[]
    for i in xrange(M):
        QR.append(lambda t: [0])
    A = 2
    QR[4] = lambda t: [A * (20 + 3 * np.sin(t / 4))]
    tau = 1e2
    sol = Solver(lam, QR, c, eps, square, tau)
    for i in range(len(list_of_vel)):
        luminance = normalize_heat(sol.T0[i])
        for j in xrange(irange[i], irange[i + 1]):
            pointcolor[j] = [luminance, luminance, luminance]
    global eye
    eye = np.zeros(3)
    global lat
    lat = 0
    global lon
    lon = np.arctan2(0, -1)
    glutMainLoop()
