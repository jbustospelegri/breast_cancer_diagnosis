from glob import glob
from pathlib import Path
from typing import io, Union, Any

import os
import numpy as np
import cv2
import math
import logging
import traceback
import pygogo
import pandas as pd


def get_filename(x: io) -> str:
    """
    Función para obtener el nombre de un archivo
    :param x: filepath del archivo
    :return nombre del archivo
    """
    return os.path.basename(os.path.splitext(x)[0])


def get_dirname(x: io) -> str:
    """
    Función para obtener el nombre del directorio que contiene un archivo
    :param x: filepath del archivo
    :return: nombre del directorio
    """
    return os.path.dirname(os.path.splitext(x)[0])


def get_path(*args: Union[io, str], create: bool = True) -> io:
    """
    Función para crear un path siendo robusto al OS
    :param args: nombre de los directorios que formarán el path
    :param create: booleano para crear el directorio en caso que no exista.
    :return filepath final en función del OS.
    """
    path = os.path.join(*args)
    if create:
        create_dir(get_dirname(path))
    return path


def create_dir(path: io):
    """
    Función para crear un directorio y todos sus padres en caso de no existir
    :param path: directorio a crear
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def search_files(file: io, ext: str, in_subdirs: bool = True) -> iter:
    """
    Función para buscar todos los archivos contenidos en un directorio que presenten la extension mostrada
    :param file: directorio sobre el cual realiar la busqueda
    :param ext: extensión del archivo
    :param in_subdirs: boleano para buscar en el directorio hijo o en los subdirectorios
    :return: lista con los archivos del directorio file que tienen la extension ext
    """
    if in_subdirs:
        return glob(os.path.join(file, '**', f'*.{ext}'), recursive=True)
    else:
        return glob(os.path.join(file, f'*.{ext}'), recursive=True)


def save_img(img: np.ndarray, save_example_dirpath: io, name: str):
    """
    Función para almacenar una imagen
    :param img: imagen a almacenar
    :param save_example_dirpath:  directorio en el que se almacenará la imagen
    :param name: nombre con el que se almacenará la imagen
    """
    if save_example_dirpath is not None:
        cv2.imwrite(get_path(save_example_dirpath, f'{name}.png'), img=img)


def detect_func_err(func):
    """
    Decorador para detectar en qué función se procude un error
    """
    def _exec(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as err:
            err.args += (func.__name__, )
            raise
    return _exec


def closest_power2(x):
    """
    Return the closest power of 2 by checking whether
    the second binary number is a 1.
    """
    op = math.floor if bin(x)[3] != "1" else math.ceil
    return 2**(op(math.log(x, 2)))


def get_number_of_neurons(previous_shape: list) -> int:
    """
    Función para obtener el número de neuronas óptimo para una capa neuronal a partir del valor más próximo a la raiz
    cuadrada del número de neuronas de la capa previa
    :param previous_shape: número de neuronas en la capa prévia
    :return: número de neuronas óptimo
    """
    num_params = np.prod(list(filter(lambda x: x is not None, previous_shape)))
    return closest_power2(int(np.sqrt(num_params)) - 1)


def bulk_data(file: io, mode: str = 'w', **kwargs) -> None:
    """
    Función para escribir en un archivo los kwargs especificados.
    :param file: nombre del archivo a escribir
    :param mode: modo de esctiura del archivo (W: escritura o A: append)
    :param kwargs: los keys serán las columnas del fichero y los values los valores de las columnas a escribir
    """
    pd.DataFrame.from_dict(kwargs, orient='index').T.\
        to_csv(file, sep=';', decimal=',', header=not os.path.isfile(file) or mode == 'w', mode=mode, encoding='utf-8',
               index=False)


def get_value_from_args_if_exists(args: list, pos: int, default: Any, *exceptions) -> Any:
    """
    Función para obtener un parámetro de un conjunto de argumentos devolviendo un valor por defecto si se producen
    las excepciones especificadas
    :param args: lista de argumentos a buscar
    :param pos: posición del argumento
    :param default: valor por defecto a devolver
    :param exceptions: excepciones a tratar
    :return: valor de la posición correspondiente en la lista de parámetros
    """
    try:
        return args[pos]
    except tuple(exceptions):
        return default


def create_file_formatter(uri_log):
    """
        Creates a csv file for log with the headers
    """
    # To get path exclude file name
    get_path(uri_log, create=True)

    # If log don't exist create headers
    if not os.path.isfile(uri_log):
        with open(uri_log, mode="w") as file:
            file.write("; ".join(['Fecha', 'Modulo', 'File', 'Función', 'Linea', 'Descripción', 'Tipo Error', 'Error']))
            file.write("\n")

    return logging.FileHandler(uri_log)


def transform_error(**kwargs_in):
    """
    Función que sirve para procesar los parametros de una excepcion.
    :param kwargs_in: Keys del diccionario.
            'error': mensaje de excepcion
            'description': narrativa escrita por el usuario para tener mejor comprensión del error del programa

    :return: diccionario con los siguientes parametros:
            'func_name': nombre de la función que produce la excepción.
            'error_name': nombre del tipo de error.
            'error': descriptivo del error.
            'error_line': línea de código dónde se produce el error
            'description': narrativa escrita por el usuario para tener mejor comprensión del error del programa.
    """

    kwargs_out = {
        "file_name": "",
        "func_name": "",
        "error_line": "",
        "error_name": "",
        "error": "",
        "description": ""
    }

    if "error" in kwargs_in.keys():
        stack_info = traceback.extract_tb(kwargs_in['error'].__traceback__)[-1]
        kwargs_out['file_name'] = f'{get_filename(stack_info[0])}'
        kwargs_out['func_name'] = stack_info[2]
        kwargs_out['error_line'] = stack_info[1]
        kwargs_out["error_name"] = kwargs_in["error"].__class__.__name__
        kwargs_out["error"] = str(kwargs_in["error"]).replace("\n", " | ").replace(";", ":")

    if "description" in kwargs_in.keys():
        kwargs_out["description"] = kwargs_in['description']

    return kwargs_out


def log_error(module, file_path, **kwargs_msg):
    """
    Función que genera un log de errores.

    :param module: nombre del script y modulo que genera el error
    :param file_path: ruta dónde se generará el archivo de errores
    :param kwargs_msg:
            'error': excepción generada
            'description': narrativa escrita por el usuario para tener mejor comprensión del error del programa.
    """
    logging_fmt = '%(asctime)s;%(name)s;%(message)s'
    fmttr = logging.Formatter(logging_fmt, datefmt=pygogo.formatters.DATEFMT)
    fhdlr = create_file_formatter(file_path)

    logger = pygogo.Gogo(name=module, high_hdlr=fhdlr, high_formatter=fmttr, monolog=True).get_logger("py")

    kwargs_error = transform_error(**kwargs_msg)

    msg = "{file_name};{func_name};{error_line};{description};{error_name};{error}".format(**kwargs_error)

    if len(logger.handlers) > 2:
        logger.handlers.pop(0)
    logger.error(msg)

    for hdlr in logger.handlers:
        hdlr.close()
        logger.removeHandler(hdlr)


def load_point(point_string: str) -> tuple:
    """
    Dado un string con las tuplas (x, y) se recuperan los valores de x y de y
    :param point_string: srting con las tuplas
    :return: tupla con las coordenadas y e x convertidas a float
    """
    x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
    return y, x


def get_contours(img: np.ndarray) -> list:
    """
    Se recuperan los contornos de una máscara dada
    :param img: imagen
    :return: contornos de la imagen
    """
    return cv2.findContours(image=img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[0]


def excel_column_name(n: int) -> str:
    """
    Función para convertir un número en su respectiva letra de columna para excel
    :param n: número de la columna
    :return: letra de la columna
    """
    name = ''
    while n > 0:
        n, r = divmod(n-1, 26)
        name = chr(r + ord('A')) + name
        return name