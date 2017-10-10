from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5 import QtGui, QtWidgets
import sys

class Object:
    def __init__(self, x=0, y=0, radius=1, color=None):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

def onmove(event):
    if event.xdata is None:
        xline.setText('')
    else:
        xline.setText(str(round(event.xdata, 1)))
    if event.ydata is None:
        yline.setText('')
    else:
        yline.setText(str(round(event.ydata, 1)))

def onpress(event):
    if not event.xdata is None and not event.ydata is None:
        obj = Object(event.xdata, event.ydata, slider.value(), colorbox.currentText())
        plt.subplot().add_patch(plt.Circle((obj.x, obj.y), radius=obj.radius, color=obj.color))
        canvas.draw()
        objects.append(obj)

def minus():
    plt.xlim(plt.xlim()[0] * 1.5, plt.xlim()[1] * 1.5)
    plt.ylim(plt.ylim()[0] * 1.5, plt.ylim()[1] * 1.5)
    canvas.draw()

def plus():
    plt.xlim(plt.xlim()[0] / 1.5, plt.xlim()[1] / 1.5)
    plt.ylim(plt.ylim()[0] / 1.5, plt.ylim()[1] / 1.5)
    canvas.draw()

def openfiledialog():
    path = QtWidgets.QFileDialog.getOpenFileName(w, filter='*.xml')
    openfilepath.setText(path[0])

def savefiledialog():
    path = QtWidgets.QFileDialog.getSaveFileName(w, filter='*.xml')
    savefilepath.setText(path[0])

def changed():
    sizeline.setText(str(slider.value()))

def edited():
    size = 1
    try:
        size = max(1, min(int(sizeline.text()), 100))
    except:
        pass
    slider.setValue(size)

app = QtWidgets.QApplication(sys.argv)
w = QtWidgets.QWidget()
w.resize(800, 600)
w.setWindowTitle('Task 3')
hbox = QtWidgets.QHBoxLayout()
w.setLayout(hbox)
vbox_left = QtWidgets.QVBoxLayout()
vbox_right = QtWidgets.QVBoxLayout()
hbox.addLayout(vbox_left)
hbox.addLayout(vbox_right)
figure = plt.figure()
canvas = FigureCanvas(figure)
vbox_left.addWidget(canvas)
hbox_zoom = QtWidgets.QHBoxLayout()
vbox_left.addLayout(hbox_zoom)
button_plus = QtWidgets.QPushButton('+')
button_plus.clicked.connect(plus)
button_minus = QtWidgets.QPushButton('-')
button_minus.clicked.connect(minus)
hbox_zoom.addWidget(button_plus)
hbox_zoom.addWidget(button_minus)
hbox_open = QtWidgets.QHBoxLayout()
hbox_save = QtWidgets.QHBoxLayout()
vbox_left.addLayout(hbox_save)
vbox_left.addLayout(hbox_open)
openlabel = QtWidgets.QLabel("Open:")
openfilepath = QtWidgets.QLineEdit()
openfilepath.setDisabled(True)
button_open = QtWidgets.QPushButton('Browse...')
button_open.clicked.connect(openfiledialog)
hbox_open.addWidget(openlabel)
hbox_open.addWidget(openfilepath)
hbox_open.addWidget(button_open)
savelabel = QtWidgets.QLabel("Save:")
savefilepath = QtWidgets.QLineEdit()
savefilepath.setDisabled(True)
button_save = QtWidgets.QPushButton('Browse...')
button_save.clicked.connect(savefiledialog)
hbox_save.addWidget(savelabel)
hbox_save.addWidget(savefilepath)
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
xline = QtWidgets.QLineEdit()
xline.setDisabled(True)
yline = QtWidgets.QLineEdit()
yline.setDisabled(True)
canvas.mpl_connect('motion_notify_event', onmove)
canvas.mpl_connect('button_press_event', onpress)
hbox_pos.addWidget(xlabel)
hbox_pos.addWidget(xline)
hbox_pos.addWidget(ylabel)
hbox_pos.addWidget(yline)
hbox_color = QtWidgets.QHBoxLayout()
vbox_edit.addLayout(hbox_color)
colorbox = QtWidgets.QComboBox()
colorbox.addItem('Red')
colorbox.addItem('Green')
colorbox.addItem('Blue')
colorlabel = QtWidgets.QLabel('Color:')
hbox_color.addWidget(colorlabel)
hbox_color.addWidget(colorbox)
hbox_size = QtWidgets.QHBoxLayout()
vbox_edit.addLayout(hbox_size)
sizelabel = QtWidgets.QLabel('Size:')
sizeline = QtWidgets.QLineEdit()
hbox_size.addWidget(sizelabel)
hbox_size.addWidget(sizeline)
slider = QtWidgets.QSlider(Qt.Horizontal)
vbox_edit.addWidget(slider)
slider.setMaximum(100)
slider.setMinimum(1)
slider.sliderMoved.connect(changed)
slider.sliderPressed.connect(changed)
sizeline.setText(str(slider.value()))
sizeline.textEdited.connect(edited)
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
objects = []
w.show()
sys.exit(app.exec_())