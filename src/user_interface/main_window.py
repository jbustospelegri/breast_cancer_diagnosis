import os
import traceback

from threading import Thread
from typing import io
from datetime import datetime

from PyQt5.QtCore import Qt, QDir
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QProgressBar, QLabel, QSizePolicy, QGridLayout, QGroupBox, QVBoxLayout, QFileDialog,
    QMessageBox
)

from src.user_interface.signals_interface import SignalError, SignalProgressBar, SignalCompleted
from src.user_interface.utils import center_widget_into_screen, LineStyled, populate_grid_layout, fix_application_width
from src.utils.functions import get_path, log_error
from src.utils.config import LOGGING_DATA_PATH


class MainWindow(QWidget):

    APPLICATION_NAME: str = 'Breast Cancer Diagnosis'
    WIDGETS: list = []
    EXCEL_PATH: io = None

    def __init__(self):
        super(MainWindow, self).__init__()

        # Estilo de la aplicación
        self.set_style()

        # Inicializacion de los widgets de la interficie
        self.init_widget_open_file()
        self.init_widget_start_process()
        self.init_widget_progress_bar()
        self.init_signals()

        # Se distribuyen los elementos a traves en la interfaz
        self.set_grid()
        self.callback_show_process_information(flag=False)
        self.show()

    def set_style(self):
        center_widget_into_screen(gui=self)
        fix_application_width(gui=self, scale_factor=500)
        self.setWindowTitle(self.APPLICATION_NAME)

    def set_grid(self):
        """
        Grid de selección del tipo de extracción (perímetro definido, todo el perímetro)
        """
        layout_open_file = QGridLayout()
        populate_grid_layout(
            layout_open_file, [self.label_open_file, self.line_open_file, self.button_open_file], col_stretch=[1, 5, 2]
        )
        group_box_open_file = QGroupBox("Input File")
        group_box_open_file.setLayout(layout_open_file)

        """
        GRID CON LA INFORMACIÓN DEL PROCESO (BARRA + INFO)
        """
        layout_progress_bar = QGridLayout()
        populate_grid_layout(
            layout_progress_bar, [self.label_info_process, self.progress_bar, self.button_start_process],
            col_stretch=[2, 5, 2]
        )

        general_grid = QVBoxLayout()
        general_grid.addWidget(group_box_open_file)
        general_grid.addLayout(layout_progress_bar)

        self.setLayout(general_grid)

    def init_widget_open_file(self):
        self.label_open_file = QLabel('Select File')
        self.line_open_file = LineStyled('')
        self.button_open_file = QPushButton("Open")
        self.button_open_file.clicked.connect(self.callback_open_file)

        self.WIDGETS += [self.label_open_file, self.line_open_file, self.button_open_file]

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

    def callback_open_file(self):
        dir = QFileDialog.getOpenFileName(self, caption='Select Breast Mammogram File', filter="*.xlsx")[0]
        if dir:
            self.EXCEL_PATH = QDir().toNativeSeparators(dir)
            self.line_open_file.setText(f'../{"/".join(self.EXCEL_PATH.split(os.sep)[-2:])}')

    def callback_main_process(self):

        pipeline_kwargs = {
            'excel_filepath': self.EXCEL_PATH,
            'signal_information': self.signal_information,
            'signal_error': self.signal_error,
            'signal_complete': self.signal_complete,
        }

        try:
            assert self.EXCEL_PATH is not None, "Excel filepath with mamographic information has not been selected."
            assert os.path.isfile(self.EXCEL_PATH), f"{self.EXCEL_PATH} does not exists"

            self.callback_activate_buttons(flag=False, show_process_info=True)

            self.cte_worker_thread = Thread(target=self.pipeline, kwargs=pipeline_kwargs, daemon=True)
            self.cte_worker_thread.start()

        except AssertionError as err:
            QMessageBox.warning(self, 'Error', err.args[0], QMessageBox.Ok)

    def callback_show_error_message(self, module_name:str, desc: str, trace: traceback):
        filename = get_path(LOGGING_DATA_PATH, f"error log {datetime.now():%Y%m%d}.csv")

        log_error(module=module_name, description=desc, error=trace, file_path=filename)

        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Warning)
        self.msg.setText("Unexpected error")
        self.msg.setInformativeText(desc)

        self.msg.setWindowTitle(self.APPLICATION_NAME)

        self.msg.setDetailedText(f'Escrito log de errores en {filename}\n\n{"-" * 5}ERROR{"-" * 5}\n\n{trace}')
        self.msg.setStandardButtons(QMessageBox.Close)
        self.msg.exec_()

        self.callback_activate_buttons(flag=True, show_process_info=False)

    def callback_show_finish_message(self):
        self.callback_activate_buttons(flag=True, show_process_info=False)
        self.signal_complete.show_popup(
            widget=self, title="Diagnosis completed", message=f'Diagnosis excel created in:\n COMPLETAR'
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

    @staticmethod
    def pipeline(excel_filepath: io, signal_information: SignalProgressBar, signal_error: SignalError,
                 signal_complete: SignalCompleted):
        info = ''
        try:
            info = f'Reading excel {excel_filepath}'
            signal_information.emit_update_label_and_progress_bar(0, info)
            from time import sleep
            sleep(3)
            signal_information.emit_update_label_and_progress_bar(25, 'jeje')
            sleep(3)
            signal_information.emit_update_label_and_progress_bar(50, 'shaito')
            sleep(3)
            signal_information.emit_update_label_and_progress_bar(75, 'bye')
        except Exception as err:
            signal_error.emit_error(__name__, info, err)
        else:
            signal_complete.finish_process()
