#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.integrate import odeint
from solver import *
import sys
import time
import xml.etree.ElementTree as ET

class Application(QtWidgets.QApplication):
    def exec_(self):
        window = Window()
        window.setWindowTitle(__file__)
        window.show()
        return QtWidgets.QApplication.exec_()

class Circle(dict):
    def __init__(self, x=.0, y=.0, dxdt=.0, dydt=.0, mass=.1, radius=1, color=None):
        dict.__init__(self, x=x, y=y, dxdt=dxdt, dydt=dydt, mass=mass, radius=radius, color=color)
        for key, value in self.iteritems():
            exec 'self.{} = value'.format(key)
        self.update(zip(self.keys(), map(Circle._str, self.values())))
    
    def pos(self, x=None, y=None):
        if not x is None:
            self.x = x
            self['x'] = Circle._str(x)
        if not y is None:
            self.y = y
            self['y'] = Circle._str(y)
        return self.x, self.y
    
    _str = staticmethod(lambda x: x.hex() if isinstance(x, float) else str(x))

class Window(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        hbox = QtWidgets.QHBoxLayout()
        self.setLayout(hbox)
        vbox_left = QtWidgets.QVBoxLayout()
        vbox_right = QtWidgets.QVBoxLayout()
        hbox.addLayout(vbox_left)
        hbox.addLayout(vbox_right)
        figure = plt.figure()
        self.canvas = FigureCanvas(figure)
        vbox_left.addWidget(self.canvas)
        hbox_zoom = QtWidgets.QHBoxLayout()
        vbox_left.addLayout(hbox_zoom)
        button_plus = QtWidgets.QPushButton('+')
        button_plus.clicked.connect(self.plusButtonClickedEvent)
        button_minus = QtWidgets.QPushButton('-')
        button_minus.clicked.connect(self.minusButtonClickedEvent)
        hbox_zoom.addWidget(button_plus)
        hbox_zoom.addWidget(button_minus)
        hbox_open = QtWidgets.QHBoxLayout()
        hbox_save = QtWidgets.QHBoxLayout()
        vbox_left.addLayout(hbox_save)
        vbox_left.addLayout(hbox_open)
        openlabel = QtWidgets.QLabel("Open:")
        self.openfilepath = QtWidgets.QLineEdit()
        self.openfilepath.setDisabled(True)
        button_open = QtWidgets.QPushButton('Browse...')
        button_open.clicked.connect(self.openFileEvent)
        hbox_open.addWidget(openlabel)
        hbox_open.addWidget(self.openfilepath)
        hbox_open.addWidget(button_open)
        savelabel = QtWidgets.QLabel("Save:")
        self.savefilepath = QtWidgets.QLineEdit()
        self.savefilepath.setDisabled(True)
        button_save = QtWidgets.QPushButton('Browse...')
        button_save.clicked.connect(self.saveFileEvent)
        hbox_save.addWidget(savelabel)
        hbox_save.addWidget(self.savefilepath)
        hbox_save.addWidget(button_save)
        tabs = QtWidgets.QTabWidget()
        hbox.addWidget(tabs)
        tab_edit = QtWidgets.QWidget()
        tabs.addTab(tab_edit, 'Edit')
        vbox_edit = QtWidgets.QVBoxLayout()
        tab_edit.setLayout(vbox_edit)
        tab_model = QtWidgets.QWidget()
        tabs.addTab(tab_model, 'Model')
        hbox_pos = QtWidgets.QHBoxLayout()
        vbox_edit.addLayout(hbox_pos)
        xlabel = QtWidgets.QLabel("X:")
        ylabel = QtWidgets.QLabel("Y:")
        self.xline = QtWidgets.QLineEdit()
        self.xline.setDisabled(True)
        self.yline = QtWidgets.QLineEdit()
        self.yline.setDisabled(True)
        self.canvas.mpl_connect('motion_notify_event', self.mouseMoveEvent)
        self.canvas.mpl_connect('button_press_event', self.mouseClickEvent)
        hbox_pos.addWidget(xlabel)
        hbox_pos.addWidget(self.xline)
        hbox_pos.addWidget(ylabel)
        hbox_pos.addWidget(self.yline)
        hbox_color = QtWidgets.QHBoxLayout()
        vbox_edit.addLayout(hbox_color)
        self.colorbox = QtWidgets.QComboBox()
        self.colorbox.addItem('Red')
        self.colorbox.addItem('Green')
        self.colorbox.addItem('Blue')
        colorlabel = QtWidgets.QLabel('Color:')
        hbox_color.addWidget(colorlabel)
        hbox_color.addWidget(self.colorbox)
        hbox_size = QtWidgets.QHBoxLayout()
        vbox_edit.addLayout(hbox_size)
        sizelabel = QtWidgets.QLabel('Size:')
        self.sizeline = QtWidgets.QLineEdit()
        hbox_size.addWidget(sizelabel)
        hbox_size.addWidget(self.sizeline)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        vbox_edit.addWidget(self.slider)
        self.slider.setMaximum(100)
        self.slider.setMinimum(1)
        self.slider.sliderMoved.connect(self.sliderChangeEvent)
        self.slider.sliderPressed.connect(self.sliderChangeEvent)
        self.sizeline.setText(str(self.slider.value()))
        self.sizeline.textEdited.connect(self.sizeChangedEvent)
        vbox_model = QtWidgets.QVBoxLayout()
        tab_model.setLayout(vbox_model)
        button_scipy = QtWidgets.QRadioButton('scipy')
        button_scipy.clicked.connect(self.scipyButtonClickedEvent)
        vbox_model.addWidget(button_scipy)
        button_verlet = QtWidgets.QRadioButton('verlet')
        button_verlet.clicked.connect(self.verletButtonClickedEvent)
        vbox_model.addWidget(button_verlet)
        button_verlet_threading = QtWidgets.QRadioButton('verlet-threading')
        button_verlet_threading.clicked.connect(self.verletThreadingButtonClickedEvent)
        vbox_model.addWidget(button_verlet_threading)
        button_verlet_multiprocessing = QtWidgets.QRadioButton('verlet-multiprocessing')
        button_verlet_multiprocessing.clicked.connect(self.verletMultiprocessingButtonClickedEvent)
        vbox_model.addWidget(button_verlet_multiprocessing)
        button_verlet_cython = QtWidgets.QRadioButton('verlet-cython')
        button_verlet_cython.clicked.connect(self.verletCythonButtonClickedEvent)
        vbox_model.addWidget(button_verlet_cython)
        button_verlet_opencl = QtWidgets.QRadioButton('verlet-opencl')
        button_verlet_opencl.clicked.connect(self.verletOpenComputingLanguageButtonClickedEvent)
        vbox_model.addWidget(button_verlet_opencl)
        plt.axis('on')
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        self.circles = []
    
    def minusButtonClickedEvent(self):
        plt.xlim(plt.xlim()[0] * Window._zoom, plt.xlim()[1] * Window._zoom)
        plt.ylim(plt.ylim()[0] * Window._zoom, plt.ylim()[1] * Window._zoom)
        self.canvas.draw()
    
    def mouseClickEvent(self, event):
        if not event.xdata is None and not event.ydata is None:
            circle = Circle(event.xdata, event.ydata, self.slider.value(), self.colorbox.currentText())
            plt.subplot().add_patch(plt.Circle(circle.pos(), radius=circle.radius, color=circle.color))
            self.circles.append(circle)
            self.canvas.draw()
    
    def mouseMoveEvent(self, event):
        if event.xdata is None:
            self.xline.setText(str())
        else:
            self.xline.setText(str(round(event.xdata, 1)))
        if event.ydata is None:
            self.yline.setText(str())
        else:
            self.yline.setText(str(round(event.ydata, 1)))
    
    def openFileEvent(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, filter='*.xml')
        if not path[0]:
            return
        tree = ET.parse(path[0])
        root = tree.getroot()
        frame = root.find('frame')
        self.move(int(frame.get('x')), int(frame.get('y')))
        self.resize(int(frame.get('width')), int(frame.get('height')))
        plt.xlim(map(.0.fromhex, eval(frame.get('xlim'))))
        plt.ylim(map(.0.fromhex, eval(frame.get('ylim'))))
        self.sizeline.setText(frame.get('radius'))
        self.sizeChangedEvent()
        self.colorbox.setCurrentText(frame.get('color'))
        del self.circles[:]
        plt.subplot().clear()
        for element in frame.findall('object'):
            circle = Circle(radius=int(element.get('radius')), color=element.get('color'))
            circle.pos(.0.fromhex(element.get('x')), .0.fromhex(element.get('y')))
            circle['dxdt'] = element.get('dxdt')
            circle.dxdt = .0.fromhex(circle['dxdt'])
            circle['dydt'] = element.get('dydt')
            circle.dydt = .0.fromhex(circle['dydt'])
            circle['mass'] = element.get('mass')
            circle.mass = .0.fromhex(circle['mass'])
            plt.subplot().add_patch(plt.Circle(circle.pos(), radius=circle.radius, color=circle.color))
            self.circles.append(circle)
        self.canvas.draw()
        self.openfilepath.setText(path[0])
    
    def plusButtonClickedEvent(self):
        plt.xlim(plt.xlim()[0] / Window._zoom, plt.xlim()[1] / Window._zoom)
        plt.ylim(plt.ylim()[0] / Window._zoom, plt.ylim()[1] / Window._zoom)
        self.canvas.draw()
    
    def saveFileEvent(self):
        path = QtWidgets.QFileDialog.getSaveFileName(self, filter='*.xml')
        if not path[0]:
            return
        root = ET.Element('data')
        tree = ET.ElementTree(root)
        frame = ET.SubElement(root, 'frame')
        frame.set('x', str(self.x()))
        frame.set('y', str(self.y()))
        frame.set('width', str(self.width()))
        frame.set('height', str(self.height()))
        frame.set('xlim', str(map(Circle._str, plt.xlim())))
        frame.set('ylim', str(map(Circle._str, plt.ylim())))
        frame.set('radius', self.sizeline.text())
        frame.set('color', self.colorbox.currentText())
        for circle in self.circles:
            ET.SubElement(frame, 'object', circle)
        tree.write(path[0])
        self.savefilepath.setText(path[0])
    
    def scipyButtonClickedEvent(self):
        def accelerations(p):
            G = 6.67408e-11
            dims = len(p) / len(self.circles)
            out = np.zeros(len(p))
            for i in xrange(len(self.circles)):
                for j, circle in enumerate(self.circles):
                    if i != j:
                        shift = p[j * dims:(j + 1) * dims] - p[i * dims:(i + 1) * dims]
                        out[i * dims:(i + 1) * dims] += G * circle.mass * shift / np.linalg.norm(shift) ** 3
            return out
        
        def func(*args):
            idx = len(args[0]) / 2
            return np.concatenate((args[0][idx:], accelerations(args[0][:idx])))
        
        tau = 24 * 60 * 60
        tspan = np.arange(365) * tau
        x0 = [(circle.x, circle.y) for circle in self.circles]
        y0 = np.concatenate(x0 + [(circle.dxdt, circle.dydt) for circle in self.circles])
        sol = odeint(func, y0, tspan)
        for i in xrange(tspan.shape[0]):
            for j, circle in enumerate(self.circles):
                x = sol[i, 2 * j]
                y = sol[i, 2 * j + 1]
                plt.subplot().add_patch(plt.Circle((x, y), radius=circle.radius, color=circle.color))
        self.canvas.draw()
    
    def sizeChangedEvent(self):
        try:
            size = max(1, min(int(self.sizeline.text()), 100))
        except:
            size = self.slider.value()
        self.slider.setValue(size)
    
    def sliderChangeEvent(self):
        self.sizeline.setText(str(self.slider.value()))
    
    def verletButtonClickedEvent(self):
        tau = 24 * 60 * 60
        tspan = np.arange(365) * tau
        method = Verlet(tspan=tspan)
        sol = method(self.circles)
        for bodies in sol:
            for j, circle in enumerate(bodies):
                plt.subplot().add_patch(plt.Circle(circle.pos(), radius=circle.radius, color=circle.color))
        self.canvas.draw()
    
    def verletThreadingButtonClickedEvent(self):
        tau = 24 * 60 * 60
        tspan = np.arange(365) * tau
        method = VerletThreading(tspan=tspan)
        sol = method(self.circles)
        for bodies in sol:
            for j, circle in enumerate(bodies):
                plt.subplot().add_patch(plt.Circle(circle.pos(), radius=circle.radius, color=circle.color))
        self.canvas.draw()
    
    def verletMultiprocessingButtonClickedEvent(self):
        tau = 24 * 60 * 60
        tspan = np.arange(365) * tau
        method = VerletMultiprocessing(tspan=tspan)
        sol = method(self.circles)
        for bodies in sol:
            for j, circle in enumerate(bodies):
                plt.subplot().add_patch(plt.Circle(circle.pos(), radius=circle.radius, color=circle.color))
        self.canvas.draw()
    
    def verletCythonButtonClickedEvent(self):
        tau = 24 * 60 * 60
        tspan = np.arange(365) * tau
        method = VerletCython(tspan=tspan)
        sol = method(self.circles)
        for bodies in sol:
            for j, circle in enumerate(bodies):
                plt.subplot().add_patch(plt.Circle(circle.pos(), radius=circle.radius, color=circle.color))
        self.canvas.draw()
    
    def verletOpenComputingLanguageButtonClickedEvent(self):
        tau = 24 * 60 * 60
        tspan = np.arange(365) * tau
        method = VerletOpenComputingLanguage(tspan=tspan)
        sol = method(self.circles)
        for bodies in sol:
            for j, circle in enumerate(bodies):
                plt.subplot().add_patch(plt.Circle(circle.pos(), radius=circle.radius, color=circle.color))
        self.canvas.draw()
    
    _zoom = 1.5

if __name__ == '__main__':
    app = Application(sys.argv)
    sys.exit(app.exec_())
