import sys
from contextlib import redirect_stdout

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMessageBox


class SignalError(QObject):
    """
        Clase con la señal de error producida por la interfaz gráfica. La señal de error devolverá al slot
        un conjunto de atributos como el modulo donde se ha producido el error, si es necesario parar la aplicación
        o si es necesario reactivar la ventana
    """

    signal = pyqtSignal(str, str, object, bool, bool)

    def __init__(self):
        super(QObject, self).__init__()

    def emit_error(self, error_module, error_hint, traceback, stop_exec, activate_window):
        self.signal.emit(error_module, error_hint, traceback, stop_exec, activate_window)

    @pyqtSlot(str, str, object)
    def write_error(self, error_module, error_hint, traceback):
        print(error_module, error_hint, traceback)


class SignalLogging(QObject):
    """
        Clase con la señal para generar el log de escritura
    """
    signal = pyqtSignal(str)

    def __init__(self):
        super(SignalLogging, self).__init__()

    def log(self, log_msg):
        self.signal.emit(log_msg)


class SignalCompleted(QObject):
    """
        Clase con la señal de finalización del procesado
    """

    signal = pyqtSignal()

    def __init__(self):
        super(QObject, self).__init__()

    def finish_process(self):
        self.signal.emit()

    def emit(self):
        self.finish_process()

    @pyqtSlot(object, str, str)
    def show_popup(self, widget, title, message):
        print(title, message)
        QMessageBox.information(widget, title, message, buttons=QMessageBox.Ok)


class SignalProgressBar(QObject):
    """
        Clase que emite una señal para actualizar los valores de una barra de prograso y su correspondiente label.
    """
    signal_update_progress_bar = pyqtSignal(int)
    signal_update_label_info = pyqtSignal(str)

    def __init__(self, widget_bar, widget_label):
        super(QObject, self).__init__()
        self.widget_bar = widget_bar
        self.widget_label = widget_label
        self.signal_update_progress_bar.connect(self.update_progress_bar)
        self.signal_update_label_info.connect(self.update_label)

    def flush(self):
        self.flush()

    def start_signal_stdout(self):
        redirect_stdout(self)

    @staticmethod
    def reset_signal_stdout():
        sys.stdout = sys.__stdout__

    def write(self, txt):
        self.signal_update_label_info.emit(txt)

    def writelines(self, bar_value, label_info):
        self.emit_update_label_and_progress_bar(bar_value, label_info)

    def emit_update_label_and_progress_bar(self, bar_value, label_info):
        self.signal_update_label_info.emit(label_info)
        self.signal_update_progress_bar.emit(bar_value)

    def emit_update_progress_bar(self, bar_value):
        self.signal_update_progress_bar.emit(bar_value)

    def emit_update_label(self, label_info):
        self.signal_update_label_info.emit(label_info)

    @pyqtSlot(int)
    def update_progress_bar(self, bar_value):
        self.widget_bar.setValue(bar_value)

    @pyqtSlot(str)
    def update_label(self, label_info):
        self.widget_label.setText(label_info)
