from PyQt5.QtWidgets import QWidget, QDesktopWidget, QLineEdit, QSizePolicy, QGridLayout, QSpacerItem


class ControledError(Exception):

    def __init__(self, *args):
        super(ControledError, self).__init__(*args)


def center_widget_into_screen(gui: QWidget):
    """
    Function to center the application on the screen.

    :param gui: QWidget object for PyQt5 library
    """
    centro_escritorio = QDesktopWidget().screenGeometry(0).center()
    qr = gui.frameGeometry()
    qr.moveCenter(centro_escritorio)
    gui.move(qr.topLeft())


def fix_application_width(gui: QWidget, scale_factor=550):
    """
    Function to fix an application width depending on screen resolution. Aplication height is not fixed.

    :param gui: QWidget or QMainWindow object for PyQt5 library
    :param scale_factor: scale factor to apply
    """
    scaling = QDesktopWidget().logicalDpiY() / 96
    gui.setFixedSize(-1, -1)
    gui.setMinimumWidth(scale_factor * scaling)


def fix_application_height(gui: QWidget, scale_factor=550):
    """
    Function to fix an application width depending on screen resolution. Aplication height is not fixed.

    :param gui: QWidget or QMainWindow object for PyQt5 library
    :param scale_factor: scale factor to apply
    """
    scaling = QDesktopWidget().logicalDpiY() / 96
    gui.setFixedSize(-1, -1)
    gui.setMinimumHeight(scale_factor * scaling)

def fix_application_size(gui: QWidget, scale_factor: tuple):
    scaling = QDesktopWidget().logicalDpiY() / 96
    gui.setFixedSize(-1, -1)
    gui.setMinimumSize(scale_factor[1] * scaling, scale_factor[0] * scaling)

def populate_grid_layout(layout: QGridLayout, *row_items: list, **kwargs):
    """

    :param layout:
    :param row_items:
    :param kwargs:
    :return:
    """

    hspacing = QSpacerItem(QSizePolicy.Minimum, QSizePolicy.Minimum)
    vspacing = QSpacerItem(QSizePolicy.Minimum, QSizePolicy.Minimum)

    row_span = []
    col_span = []

    for row, item in enumerate(row_items):
        try:
            row_span.append(kwargs.get('row_stretch')[row])
        except TypeError:
            row_span.append(1)

        if item is not None:
            for col, widget in enumerate(item):
                try:
                    col_span.append(kwargs.get('col_stretch')[col])
                except TypeError:
                    col_span.append(1)

                if widget is not None:
                    if widget.isWidgetType():
                        layout.addWidget(widget, sum(row_span[:row]), sum(col_span[:col]), row_span[row], col_span[col])
                    else:
                        layout.addLayout(widget, sum(row_span[:row]), sum(col_span[:col]), row_span[row], col_span[col])
                else:
                    layout.addItem(hspacing, sum(row_span[:row]), sum(col_span[:col]), row_span[row], col_span[col])
        else:
            layout.addItem(vspacing, row, 0)


class LineStyled(QLineEdit):

    def __init__(self, placeholder: str = ''):
        super(LineStyled, self).__init__()
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setReadOnly(True)
        self.setPlaceholderText(placeholder)
        self.setObjectName("Entradas")
