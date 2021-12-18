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
from utils.config import LOGGING_DATA_PATH, APPLICATION_NAME, GUI_HTML_PATH


class ApplicationWindow(QMainWindow):

    def __init__(self):
        super(ApplicationWindow, self).__init__()

        self.init_central_widget()
        self.init_toolbar()
        self.set_style()
        self.app_widget.show()
        self.show()

    def init_central_widget(self):
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.app_widget = MainWindow()
        self.central_widget.addWidget(self.app_widget)
        self.central_widget.setCurrentWidget(self.app_widget)

    def set_style(self):
        center_widget_into_screen(gui=self)
        fix_application_width(gui=self, scale_factor=500)
        self.setWindowTitle(APPLICATION_NAME)

    def init_toolbar(self):
        self.toolbar = self.addToolBar('Help')
        self.toolbar.setMovable(False)

        self.button_help = QToolButton(self)
        self.button_help.setText('Help')
        self.button_help.clicked.connect(self.show_help_window)

        self.toolbar.addWidget(self.button_help)

    def show_help_window(self):
        HelpMessageBox()


class MainWindow(QWidget):

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
        Grid de selección del tipo de extracción (perímetro definido, todo el perímetro)
        """
        layout_open_file = QGridLayout()
        populate_grid_layout(layout_open_file, [self.line_open_file, self.button_open_file], col_stretch=[6, 2])
        group_box_open_file = QGroupBox("Input Excel Filepath")
        group_box_open_file.setLayout(layout_open_file)

        layout_open_dir = QGridLayout()
        populate_grid_layout(layout_open_dir, [self.line_open_dir, self.button_open_dir], col_stretch=[6, 2])
        group_box_open_dir = QGroupBox("Output Directory")
        group_box_open_dir.setLayout(layout_open_dir)

        """
        GRID CON LA INFORMACIÓN DEL PROCESO (BARRA + INFO)
        """
        layout_progress_bar = QGridLayout()
        populate_grid_layout(
            layout_progress_bar, [self.label_info_process, self.progress_bar, self.button_start_process],
            col_stretch=[4, 3, 2]
        )

        general_grid = QVBoxLayout()
        general_grid.addWidget(group_box_open_file)
        general_grid.addWidget(group_box_open_dir)
        general_grid.addLayout(layout_progress_bar)

        self.setLayout(general_grid)

    def init_widget_open_file(self):
        self.label_open_file = QLabel('Select File')
        self.line_open_file = LineStyled('')
        self.button_open_file = QPushButton("Open")
        self.button_open_file.clicked.connect(self.callback_open_file)

        self.WIDGETS += [self.label_open_file, self.line_open_file, self.button_open_file]

    def init_widget_open_dir(self):
        self.label_open_dir = QLabel('Select Folder')
        self.line_open_dir = LineStyled('')
        self.button_open_dir = QPushButton("Open")
        self.button_open_dir.clicked.connect(self.callback_open_dir)

        self.WIDGETS += [self.label_open_dir, self.line_open_dir, self.button_open_dir]

    def init_widget_start_process(self):
        self.button_start_process = QPushButton("Start")
        self.button_start_process.clicked.connect(self.callback_main_process)

        self.WIDGETS += [self.button_start_process]

    def init_widget_progress_bar(self):
        self.progress_bar = QProgressBar()
        self.label_info_process = QLabel('')
        self.label_info_process.setWordWrap(True)
        self.label_info_process.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

    def init_signals(self):

        self.signal_error = SignalError()
        self.signal_error.signal.connect(self.callback_show_error_message)

        self.signal_complete = SignalCompleted()
        self.signal_complete.signal.connect(self.callback_show_finish_message)

        self.signal_information = SignalProgressBar(widget_bar=self.progress_bar, widget_label=self.label_info_process)

        self.signal_log = SignalLogging()
        self.signal_log.signal.connect(self.callback_generate_log)

    def callback_open_file(self):
        dir = QFileDialog.getOpenFileName(self, caption='Select Breast Mammogram File', filter="*.xlsx")[0]
        if dir:
            self.EXCEL_PATH = QDir().toNativeSeparators(dir)
            self.line_open_file.setText(f'../{"/".join(self.EXCEL_PATH.split(os.sep)[-2:])}')

    def callback_open_dir(self):
        dir = QFileDialog.getExistingDirectory(self, caption='Select output directory')
        if dir:
            self.OUT_PATH = QDir().toNativeSeparators(dir)
            self.line_open_dir.setText(f'../{"/".join(self.OUT_PATH.split(os.sep)[-2:])}')

    def callback_main_process(self):

        pipeline_kwargs = {
            'excel_filepath': self.EXCEL_PATH,
            'out_dirpath': self.OUT_PATH,
            'signal_information': self.signal_information,
            'signal_error': self.signal_error,
            'signal_complete': self.signal_complete,
            'signal_log': self.signal_log
        }

        try:
            assert self.EXCEL_PATH is not None, "Excel filepath with mamographic information has not been selected."
            assert os.path.isfile(self.EXCEL_PATH), f"{self.EXCEL_PATH} does not exists"

            assert self.OUT_PATH is not None, "Output dirpath has not been selected."
            assert os.path.isdir(self.OUT_PATH), f"{self.OUT_PATH} does not exists"

            self.callback_activate_buttons(flag=False, show_process_info=True)

            Thread(target=generate_predictions_pipeline, kwargs=pipeline_kwargs, daemon=True).start()

        except AssertionError as err:
            QMessageBox.warning(self, 'Error', err.args[0], QMessageBox.Ok)

    def callback_show_error_message(self, module_name:str, desc: str, trace: traceback, stop_exec: bool = True):
        filename = get_path(LOGGING_DATA_PATH, f"error log {datetime.now():%Y%m%d}.csv")

        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Warning)
        self.msg.setText("Unexpected error")
        self.msg.setInformativeText(desc)

        self.msg.setWindowTitle(APPLICATION_NAME)

        self.msg.setDetailedText(f'Escrito log de errores en {filename}\n\n{"-" * 5}ERROR{"-" * 5}\n\n{trace}')
        self.msg.setStandardButtons(QMessageBox.Close)
        self.msg.exec_()

        if stop_exec:
            log_error(module=module_name, description=desc, error=trace, file_path=filename)
            self.callback_activate_buttons(flag=True, show_process_info=False)

    def callback_show_finish_message(self):
        self.callback_activate_buttons(flag=True, show_process_info=False)
        self.signal_complete.show_popup(
            widget=self, title="Diagnosis completed", message=f'Diagnosis excel created in:\n {self.OUT_PATH}'
        )

    def callback_activate_buttons(self, flag: bool, show_process_info: bool):
        for widget in self.WIDGETS:
            widget.setEnabled(flag)

        self.callback_show_process_information(show_process_info)

        if not flag:
            self.setCursor(QCursor(Qt.WaitCursor))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))

    def callback_show_process_information(self, flag: bool):
        self.label_info_process.setText('')
        self.progress_bar.setValue(0)

        if flag:
            self.label_info_process.show()
            self.progress_bar.show()
        else:
            self.label_info_process.hide()
            self.progress_bar.hide()

    def callback_generate_log(self, msg: str):
        with open(get_path(self.OUT_PATH, f'Execution-logging {datetime.now():%Y%m%d}.txt'), 'a') as f:
            f.write(f'\n{"-" * 50}\n{msg}\n{"-" * 50}')


class HelpMessageBox(QWidget):

    def __init__(self):
        super(HelpMessageBox, self).__init__()

        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.set_style()
        self.set_grid()
        self.show()

    def set_style(self):
        center_widget_into_screen(gui=self)
        fix_application_size(gui=self, scale_factor=(450, 500))
        self.setWindowTitle('Help Window')

    def set_grid(self):

        layout = QVBoxLayout(self)

        self.title = QLabel('How to use breast cancer diagnosis tool')
        layout.addWidget(self.title)

        self.edit = QTextBrowser()
        self.edit.setReadOnly(True)
        self.edit.setSource(QUrl.fromLocalFile(GUI_HTML_PATH))
        layout.addWidget(self.edit)

        self.button = QPushButton('Close')
        self.button.clicked.connect(lambda: self.close())
        layout.addWidget(self.button)

        self.setLayout(layout)
