from pathlib import Path
from typing import io
from contextlib import redirect_stdout

import functools
import logging
import os


def get_filename(x: io) -> str:
    return os.path.basename(os.path.splitext(x)[0])


def get_dirname(x: io) -> str:
    return os.path.dirname(os.path.splitext(x)[0])


def create_dir(func):
    """
    decorador utilizado para crear un directorio en caso de no existir. Para ello, la función decorada debe contener
    el argumento path o dirname
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # se recupera el directorio en caso de contener los argumentos path o dirname
            p = Path(kwargs.get('path') or kwargs.get('dirname'))
            # se crea el directorio y sus subdirectorios en caso de no existir
            p.mkdir(parents=True, exist_ok=True)
        except TypeError:
            pass
        finally:
            return func(*args, **kwargs)

    return wrapper


def log_cmd_to_file(func):
    """
    Función para redirigir la salida sys.stdout a un archivo de texto. Para ello, la función decorada debe contener el
    argumento file_log
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'file_log' in kwargs:
            try:
                assert os.path.splitext(kwargs.get('file_log'))[-1] == '.txt'
                # Se redirige la salida de sys.sdout a la hora de reproducir la función
                with open(kwargs.get('file_log'), 'a') as f:
                    with redirect_stdout(f):
                        return func(*args, **kwargs)
            except AssertionError:
                logging.warning('El fichero declarado en file_log no es del tipo txt')
                return func(*args, **kwargs)
            finally:
                # En caso de que no se realice ninguna escritura en el documento de salida, se suprime
                if os.path.getsize(kwargs.get('file_log')) == 0:
                    os.remove(kwargs.get('file_log'))
        else:
            return func(*args, **kwargs)

    return wrapper
