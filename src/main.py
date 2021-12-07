import multiprocessing
import sys
import os

from PyQt5 import QtCore
from PyQt5.QtWidgets import QStyleFactory, QApplication


from src.user_interface.main_window import MainWindow
from src.utils.config import GUI_CSS_PATH

if __name__ == '__main__':

    sys.path.append(os.path.abspath(os.path.dirname(__name__)))

    app = QApplication(sys.argv)

    stream = QtCore.QFile(GUI_CSS_PATH)
    stream.open(QtCore.QIODevice.ReadOnly)
    app.instance().setStyleSheet(QtCore.QTextStream(stream).readAll())

    if "windowsvista" in QStyleFactory.keys():
        app.setStyle(QStyleFactory.create("windowsvista"))
    elif "Fusion" in QStyleFactory.keys():
        app.setStyle(QStyleFactory.create("Fusion"))

    multiprocessing.freeze_support()
    gui = MainWindow()
    sys.exit(app.exec_())