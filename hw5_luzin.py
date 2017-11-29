#!/usr/bin/env python
# -*- coding: utf-8 -*-

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import sys

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
    glScale(8, 8, 8)
    glTranslate(0, -0.25, -4)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_DOUBLE, 0, pointdata)
    glNormalPointer(GL_DOUBLE, 0, normals)
    glColorPointer(3, GL_DOUBLE, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0, len(pointdata))
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glRotate(np.degrees(np.arctan2(-1, 0)), 1, 0, 0) 
    glTranslate(0, 0, 0.5)
    glColor(1, 0, 0)
    glutSolidCylinder(0.5, 1e-3, 60, 1)
    glTranslate(0, 0, 1e-3)
    glColor(1, 1, 0)
    glutSolidCone(0.5, 0.5, 60, 60)
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
    glClearColor(0.2, 0.2, 0.2, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    vertex = create_shader(GL_VERTEX_SHADER, """
        varying vec3 N;
        varying vec3 v;
        varying vec4 vertex_color;
        
        void main() {
            v = vec3(gl_ModelViewMatrix * gl_Vertex);
            N = normalize(gl_NormalMatrix * gl_Normal);
            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            vertex_color = gl_Color;
        }
        """)
    fragment = create_shader(GL_FRAGMENT_SHADER, """
        varying vec3 N;
        varying vec3 v;
        varying vec4 vertex_color;
        
        void main() {
            vec3 L = normalize(-v);
            vec3 R = normalize(-reflect(L, N));
            vec4 Idiff = vertex_color * max(dot(N, L), 0.0);
            vec4 Ispec = vec4(0.7, 0.7, 0.0, 1.0) * pow(max(dot(R, L), 0.0), 30.0);
            //Ispec = specColor * pow ( max ( dot ( n2, h2 ), 0.0 ), specPower );
            gl_FragColor = Idiff + Ispec;
            //gl_FragColor = vertex_color;
        }
        """)
    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    glUseProgram(program)
    pointdata = [[-0.5, 0, 0.5], [0.5, 0, 0.5], [0.5, 0, -0.5],
                 [0.5, 0, -0.5], [-0.5, 0, -0.5], [-0.5, 0, 0.5],
                 [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5],
                 [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
                 [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0, 0.5],
                 [0.5, 0, 0.5], [-0.5, 0, 0.5], [-0.5, -0.5, 0.5],
                 [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0, -0.5],
                 [0.5, 0, -0.5], [-0.5, 0, -0.5], [-0.5, -0.5, -0.5],
                 [0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0, -0.5],
                 [0.5, 0, -0.5], [0.5, 0, 0.5], [0.5, -0.5, 0.5],
                 [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5], [-0.5, 0, -0.5],
                 [-0.5, 0, -0.5], [-0.5, 0, 0.5], [-0.5, -0.5, 0.5]]
    pointcolor = [[1, 0, 0], [1, 0, 0], [1, 0, 0],
                  [1, 0, 0], [1, 0, 0], [1, 0, 0],
                  [0, 1, 0], [0, 1, 0], [0, 1, 0],
                  [0, 1, 0], [0, 1, 0], [0, 1, 0],
                  [1, 1, 0], [1, 1, 0], [1, 1, 0],
                  [1, 1, 0], [1, 1, 0], [1, 1, 0],
                  [1, 1, 0], [1, 1, 0], [1, 1, 0],
                  [1, 1, 0], [1, 1, 0], [1, 1, 0],
                  [1, 1, 0], [1, 1, 0], [1, 1, 0],
                  [1, 1, 0], [1, 1, 0], [1, 1, 0],
                  [1, 1, 0], [1, 1, 0], [1, 1, 0],
                  [1, 1, 0], [1, 1, 0], [1, 1, 0]]
    normals = [[0, 1, 0], [0, 1, 0], [0, 1, 0],
               [0, 1, 0], [0, 1, 0], [0, 1, 0],
               [0, -1, 0], [0, -1, 0], [0, -1, 0],
               [0, -1, 0], [0, -1, 0], [0, -1, 0],
               [0, 0, 1], [0, 0, 1], [0, 0, 1],
               [0, 0, 1], [0, 0, 1], [0, 0, 1],
               [0, 0, -1], [0, 0, -1], [0, 0, -1],
               [0, 0, -1], [0, 0, -1], [0, 0, -1],
               [1, 0, 0], [1, 0, 0], [1, 0, 0],
               [1, 0, 0], [1, 0, 0], [1, 0, 0],
               [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
               [-1, 0, 0], [-1, 0, 0], [-1, 0, 0]]
    global eye
    eye = np.zeros(3)
    global lat
    lat = 0
    global lon
    lon = np.arctan2(0, -1)
    glutMainLoop()
