from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
import sys


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        xpos, ypos, width, height = 200, 200, 300, 300
        self.setGeometry(xpos, ypos, width, height)
        self.setWindowTitle("PyQt5 Tutorial")
        self.initUi()

    def initUi(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText("My first label")
        self.label.move(50, 50)

        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("My first Button")
        self.b1.clicked.connect(self.clicked)

    def clicked(self):
        self.label.setText("You pressed the button")
        self.update()

    def update(self):
        self.label.adjustSize()


def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())


window()


# https://www.youtube.com/watch?v=FVpho_UiDAY&list=PLzMcBGfZo4-lB8MZfHPLTEHO9zJDDLpYj&index=3&ab_channel=TechWithTim

############
"""convert .ui file into python code"""
# open cmd in folder directory
# type pyuic5 -x <ui file name>  ## will print file in command line
## or you can save it directly
# pyuic5 -x <ui file name> -o <saving file name>.py