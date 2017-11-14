from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

class Application(QtWidgets.QApplication):
    def exec_(self):
        window = Window()
        window.setWindowTitle(__file__)
        window.show()
        return QtWidgets.QApplication.exec_()

class Circle(object):
    def __init__(self, x=0, y=0, radius=1, color=None):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
    
    def pos(self):
        return self.x, self.y
    
    def x(self, x=None):
        if not x is None:
            self.x = x
        return self.x
    
    def y(self, y=None):
        if not y is None:
            self.y = y
        return self.y

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
        vbox_model.addWidget(button_scipy)
        button_verlet = QtWidgets.QRadioButton('verlet')
        vbox_model.addWidget(button_verlet)
        button_verlet_threading = QtWidgets.QRadioButton('verlet-threading')
        vbox_model.addWidget(button_verlet_threading)
        button_verlet_multiprocessing = QtWidgets.QRadioButton('verlet-multiprocessing')
        vbox_model.addWidget(button_verlet_multiprocessing)
        button_verlet_cython = QtWidgets.QRadioButton('verlet-cython')
        vbox_model.addWidget(button_verlet_cython)
        button_verlet_opencl = QtWidgets.QRadioButton('verlet-opencl')
        vbox_model.addWidget(button_verlet_opencl)
        plt.axis('on')
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        self.circles = []
    
    def minusButtonClickedEvent(self):
        plt.xlim(plt.xlim()[0] * 1.5, plt.xlim()[1] * 1.5)
        plt.ylim(plt.ylim()[0] * 1.5, plt.ylim()[1] * 1.5)
        self.canvas.draw()
    
    def mouseClickEvent(self, event):
        if not event.xdata is None and not event.ydata is None:
            circle = Circle(event.xdata, event.ydata, self.slider.value(), self.colorbox.currentText())
            plt.subplot().add_patch(plt.Circle(circle.pos(), radius=circle.radius, color=circle.color))
            self.canvas.draw()
            self.circles.append(circle)
    
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
        self.openfilepath.setText(path[0])
    
    def plusButtonClickedEvent(self):
        plt.xlim(plt.xlim()[0] / 1.5, plt.xlim()[1] / 1.5)
        plt.ylim(plt.ylim()[0] / 1.5, plt.ylim()[1] / 1.5)
        self.canvas.draw()
    
    def saveFileEvent(self):
        path = QtWidgets.QFileDialog.getSaveFileName(self, filter='*.xml')
        self.savefilepath.setText(path[0])
    
    def sizeChangedEvent(self):
        size = 1
        try:
            size = max(1, min(int(self.sizeline.text()), 100))
        except:
            pass
        self.slider.setValue(size)
    
    def sliderChangeEvent(self):
        self.sizeline.setText(str(self.slider.value()))

if __name__ == '__main__':
    app = Application(sys.argv)
    sys.exit(app.exec_())