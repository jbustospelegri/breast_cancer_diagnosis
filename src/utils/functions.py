from glob import glob
from pathlib import Path
from typing import io, Union

import os
import numpy as np
import cv2
import math
import pandas as pd


def get_filename(x: io) -> str:
    return os.path.basename(os.path.splitext(x)[0])


def get_dirname(x: io) -> str:
    return os.path.dirname(os.path.splitext(x)[0])


def get_path(*args: Union[io, str], create: bool = True) -> io:
    path = os.path.join(*args)
    if create:
        create_dir(get_dirname(path))
    return path


def create_dir(path: io):
    Path(path).mkdir(parents=True, exist_ok=True)


def search_files(file: io, ext: str, in_subdirs: bool = True) -> iter:
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
    num_params = np.prod(list(filter(lambda x: x is not None, previous_shape)))
    return closest_power2(int(np.sqrt(num_params)) - 1)


def bulk_data(file: io, mode: str = 'w', **kwargs) -> None:
    pd.DataFrame.from_dict(kwargs, orient='index').T.\
        to_csv(file, sep=';', decimal=',', header=not os.path.isfile(file) or mode == 'w', mode=mode, encoding='utf-8',
               index=False)