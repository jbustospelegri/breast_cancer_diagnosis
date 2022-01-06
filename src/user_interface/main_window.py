import os
import traceback

from threading import Thread
from typing import io
from datetime import datetime

from PyQt5.QtCore import Qt, QDir, QUrl
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QProgressBar, QLabel, QSizePolicy, QGridLayout, QGroupBox, QVBoxLayout, QFileDialog,
    QMessageBox, QMainWindow, QToolButton, QStackedWidget, QTextBrowser
)

from user_interface.signals_interface import SignalError, SignalProgressBar, SignalCompleted, SignalLogging
from user_interface.pipeline import generate_predictions_pipeline
from user_interface.utils import (
    center_widget_into_screen, LineStyled, populate_grid_layout, fix_application_width, fix_application_size
)
from utils.functions import get_path, log_error
from utils.config import APPLICATION_NAME, GUI_HTML_PATH


class ApplicationWindow(QMainWindow):
    """
    Ventana que contendrá la aplicación y el mensaje de ayuda mediante un toolbar
    """

    def __init__(self):
        super(ApplicationWindow, self).__init__()

        self.init_central_widget()
        self.init_toolbar()
        self.set_style()
        self.app_widget.show()
        self.show()

    def init_central_widget(self):
        """
        Se añade el mainwindow como widget central de la aplicación (se mostrará por defecto al abrirse)
        """
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.app_widget = MainWindow()
        self.central_widget.addWidget(self.app_widget)
        self.central_widget.setCurrentWidget(self.app_widget)

    def set_style(self):
        """
        Se setea el tamaño de la aplicación y su posición
        """
        center_widget_into_screen(gui=self)
        fix_application_width(gui=self, scale_factor=500)
        self.setWindowTitle(APPLICATION_NAME)

    def init_toolbar(self):
        """
        Se crea la barra de ayuda con la ventana de ayuda de uso de la aplicación
        """
        self.toolbar = self.addToolBar('Help')
        self.toolbar.setMovable(False)

        self.button_help = QToolButton(self)
        self.button_help.setText('Help')
        self.button_help.clicked.connect(self.show_help_window)

        self.toolbar.addWidget(self.button_help)

    def show_help_window(self):
        """
        Callback para mostrar la ventana de ayuda
        """
        HelpMessageBox()


class MainWindow(QWidget):

    """
        Ventana principal de la aplicación para introducir el excel de observaciones a clasificar y la carpeta de
        salida
    """

    WIDGETS: list = []
    EXCEL_PATH: io = None
    OUT_PATH: io = None

    def __init__(self):
        super(MainWindow, self).__init__()

        # Inicializacion de los widgets de la interficie
        self.init_widget_open_file()
        self.init_widget_open_dir()
        self.init_widget_start_process()
        self.init_widget_progress_bar()
        self.init_signals()

        # Se distribuyen los elementos a traves en la interfaz
        self.set_grid()
        self.callback_show_process_information(flag=False)

    def set_grid(self):
        """
        Grid de la aplicación en el cual se colocan los distintos widgets
        """

        # El primer layout contiene widgets para introducir el excel
        layout_open_file = QGridLayout()
        populate_grid_layout(layout_open_file, [self.line_open_file, self.button_open_file], col_stretch=[6, 2])
        group_box_open_file = QGroupBox("Input Excel Filepath")
        group_box_open_file.setLayout(layout_open_file)

        # El segundo layout contiene widgets para seleccionar la carpeta de salida.
        layout_open_dir = QGridLayout()
        populate_grid_layout(layout_open_dir, [self.line_open_dir, self.button_open_dir], col_stretch=[6, 2])
        group_box_open_dir = QGroupBox("Output Directory")
        group_box_open_dir.setLayout(layout_open_dir)

        # Layout de la barra de progreso y el label de información
        layout_progress_bar = QGridLayout()
        populate_grid_layout(
            layout_progress_bar, [self.label_info_process, self.progress_bar, self.button_start_process],
            col_stretch=[4, 3, 2]
        )

        # Layout final de la aplciación
        general_grid = QVBoxLayout()
        general_grid.addWidget(group_box_open_file)
        general_grid.addWidget(group_box_open_dir)
        general_grid.addLayout(layout_progress_bar)

        self.setLayout(general_grid)

    def init_widget_open_file(self):
        """
        Función para crear el widget de introducción del excel
        """
        self.label_open_file = QLabel('Select File')
        self.line_open_file = LineStyled('')
        self.button_open_file = QPushButton("Open")
        self.button_open_file.clicked.connect(self.callback_open_file)

        self.WIDGETS += [self.label_open_file, self.line_open_file, self.button_open_file]

    def init_widget_open_dir(self):
        """
        Función para crear el widget de introducción de la carpeta de salida
        """
        self.label_open_dir = QLabel('Select Folder')
        self.line_open_dir = LineStyled('')
        self.button_open_dir = QPushButton("Open")
        self.button_open_dir.clicked.connect(self.callback_open_dir)

        self.WIDGETS += [self.label_open_dir, self.line_open_dir, self.button_open_dir]

    def init_widget_start_process(self):
        """
        Función para crear el botono de inicio de la aplicación
        """
        self.button_start_process = QPushButton("Start")
        self.button_start_process.clicked.connect(self.callback_main_process)

        self.WIDGETS += [self.button_start_process]

    def init_widget_progress_bar(self):
        """
        Función para crear el widget de la barra de progreso y el label de información
        """
        self.progress_bar = QProgressBar()
        self.label_info_process = QLabel('')
        self.label_info_process.setWordWrap(True)
        self.label_info_process.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

    def init_signals(self):
        """
        Inicialización de las señales y los slots de la aplicación
        """

        self.signal_error = SignalError()
        self.signal_error.signal.connect(self.callback_show_error_message)

        self.signal_complete = SignalCompleted()
        self.signal_complete.signal.connect(self.callback_show_finish_message)

        self.signal_information = SignalProgressBar(widget_bar=self.progress_bar, widget_label=self.label_info_process)

        self.signal_log = SignalLogging()
        self.signal_log.signal.connect(self.callback_generate_log)

    def callback_open_file(self):
        """
        Función para abrir un codigo de dialogo que permita seleccionar un archivo xlsx
        """
        dir = QFileDialog.getOpenFileName(self, caption='Select Breast Mammogram File', filter="*.xlsx")[0]
        if dir:
            self.EXCEL_PATH = QDir().toNativeSeparators(dir)
            self.line_open_file.setText(f'../{"/".join(self.EXCEL_PATH.split(os.sep)[-2:])}')

    def callback_open_dir(self):
        """
        Función para abrir un codigo de dialogo que permita seleccionar la carpeta de almacenado
        """
        dir = QFileDialog.getExistingDirectory(self, caption='Select output directory')
        if dir:
            self.OUT_PATH = QDir().toNativeSeparators(dir)
            self.line_open_dir.setText(f'../{"/".join(self.OUT_PATH.split(os.sep)[-2:])}')

    def callback_main_process(self):
        """
        Función principal para llamar al pipeline de generación de las predicciones
        """
        pipeline_kwargs = {
            'excel_filepath': self.EXCEL_PATH,
            'out_dirpath': self.OUT_PATH,
            'signal_information': self.signal_information,
            'signal_error': self.signal_error,
            'signal_complete': self.signal_complete,
            'signal_log': self.signal_log
        }

        try:
            # Se valida que la información introducida por el usuario es correcta
            assert self.EXCEL_PATH is not None, "Excel filepath with mamographic information has not been selected."
            assert os.path.isfile(self.EXCEL_PATH), f"{self.EXCEL_PATH} does not exists"

            assert self.OUT_PATH is not None, "Output dirpath has not been selected."
            assert os.path.isdir(self.OUT_PATH), f"{self.OUT_PATH} does not exists"

            # Se desactivan los botones de la aplicación para que el usuario vea que esta en modo de 'procesad'
            self.callback_activate_buttons(flag=False, show_process_info=True)

            # Se lanza un multihilo para realizar el procesado. Este paso es necesario dado que la aplicación
            # se bloquearia al recibir las señales emitidas por el usuario al interactura con esta
            Thread(target=generate_predictions_pipeline, kwargs=pipeline_kwargs, daemon=True).start()

        except AssertionError as err:
            QMessageBox.warning(self, 'Error', err.args[0], QMessageBox.Ok)

    def callback_show_error_message(self, module_name: str, desc: str, trace: traceback, stop_exec: bool = True,
                                    activate_window: bool = True):
        """
        Función para emitir el mensaje de error y escribir el log de errores producido durante la clasificación de las
        observaciones
        :param module_name: modulo en elque se ha producido el error
        :param desc: descripción del error
        :param trace: traceback con el error
        :param stop_exec: boleano para interrumpir la conexion
        :param activate_window: boleano para volver a activar los widgets de la aplicación
        """

        # Se obtiene el filepath en el cual se volcarán los errores
        filename = get_path(self.OUT_PATH, f"error log {datetime.now():%Y%m%d}.csv")

        # Se crea una ventana con el error
        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Warning)
        self.msg.setText("Unexpected error")
        self.msg.setInformativeText(desc)

        self.msg.setWindowTitle(APPLICATION_NAME)

        self.msg.setDetailedText(f'Escrito log de errores en {filename}\n\n{"-" * 5}ERROR{"-" * 5}\n\n{trace}')
        self.msg.setStandardButtons(QMessageBox.Close)
        self.msg.exec_()

        # se escribe el log de errores y se activan los widgets en el caso de que corresponda
        if stop_exec:
            log_error(module=module_name, description=desc, error=trace, file_path=filename)
        if activate_window:
            self.callback_activate_buttons(flag=True, show_process_info=False)

    def callback_show_finish_message(self):
        """
        Función para finalizar la ejecución, reactivar los widgets e informar al usuario de que elproceso se ha
        completado exitosamente
        """
        self.callback_activate_buttons(flag=True, show_process_info=False)
        self.signal_complete.show_popup(
            widget=self, title="Diagnosis completed", message=f'Diagnosis excel created in:\n {self.OUT_PATH}'
        )

    def callback_activate_buttons(self, flag: bool, show_process_info: bool):
        """
        Función para reactivar (hacer los widgets visibles y activos) la aplicación. Adicionalmente se mostrará u
        ocultará la barra de progreso y el label de información.
        :param flag: booleano para reactivar los widgets
        :param show_process_info: booleano para mostrar u ocultar la barra de progreso y el label de información
        """

        # Se reactivan los widgets
        for widget in self.WIDGETS:
            widget.setEnabled(flag)

        # Se reactiva/descativa la barra de progreso y el label de información
        self.callback_show_process_information(show_process_info)

        # Se modifica el estado del cursor
        if not flag:
            self.setCursor(QCursor(Qt.WaitCursor))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))

    def callback_show_process_information(self, flag: bool):
        """
        Función para mostrar u ocultar la barra de progreso y el label de información.
        :param flag:  booleano para mostrar u ocultar la barra de progreso y el label de información
        """

        # Se resetea la barra de progreso y el texto informativo
        self.label_info_process.setText('')
        self.progress_bar.setValue(0)

        if flag:
            self.label_info_process.show()
            self.progress_bar.show()
        else:
            self.label_info_process.hide()
            self.progress_bar.hide()

    def callback_generate_log(self, msg: str):
        """
        Función para crear el log de ejecución del aplicativo
        :param msg: mensaje a logear
        """
        with open(get_path(self.OUT_PATH, f'Execution-logging {datetime.now():%Y%m%d}.txt'), 'a') as f:
            f.write(f'\n{"-" * 50}\n{msg}\n{"-" * 50}')


class HelpMessageBox(QWidget):
    """
        Widget para mostrar un html con mensajes de ayuda para la aplicación
    """

    def __init__(self):
        super(HelpMessageBox, self).__init__()

        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.set_style()
        self.set_grid()
        self.show()

    def set_style(self):
        """
        Función para setear el tamaño y la posición de la aplicación
        """
        center_widget_into_screen(gui=self)
        fix_application_size(gui=self, scale_factor=(450, 500))
        self.setWindowTitle('Help Window')

    def set_grid(self):
        """
        Se crea el grid con los widgets de la ventana compuesta por:
        - titulo de la ventana
        - mensaje html
        - boton de cerrar
        """

        # Se crea el grid
        layout = QVBoxLayout(self)

        # Titulo de la ventana
        self.title = QLabel('How to use breast cancer diagnosis tool')
        layout.addWidget(self.title)

        # Cuadro de dialogo html para mostrar la ayuda de la herramienta.
        self.edit = QTextBrowser()
        self.edit.setReadOnly(True)
        self.edit.setSource(QUrl.fromLocalFile(GUI_HTML_PATH))
        layout.addWidget(self.edit)

        # Boton de cerrado
        self.button = QPushButton('Close')
        self.button.clicked.connect(lambda: self.close())
        layout.addWidget(self.button)

        self.setLayout(layout)
