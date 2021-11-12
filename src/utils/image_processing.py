from pathlib import Path

import os
import pydicom
import numpy as np
import cv2

from typing import io, Union
from PIL import Image

from utils.config import LOGGING_DATA_PATH
from utils.functions import get_dirname, get_filename


def convert_img(args) -> None:
    """
    Función encargada de convertir las imagenes del formato recibido al formato explícito.

    :param args: Los argumentos deberán ser:
        - Posición 0: (Obligatorio) Ruta la imagen a transformar.
        - Posición 1: (Obligatorio) Ruta de la imagen transformada.
    """
    try:
        # Se recuperan los valores de arg. Deben de existir los 3 argumentos obligatorios.
        assert len(args) >= 2, 'Not enough arguments for convert_dcm_img function. Minimum required arguments: 3'
        img_path: io = args[0]
        dest_path: io = args[1]

        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        assert os.path.isfile(img_path), f"{img_path} doesn't exists."
        assert os.path.splitext(img_path)[1] in ['.pgm', '.dcm'], f'Conversion only available for: png, jpg'

        if os.path.splitext(img_path)[1] == '.dcm':
            convert_dcm_imgs(ori_path=img_path, dest_path=dest_path)
        else:
            convert_pgm_imgs(ori_path=img_path, dest_path=dest_path)
    except AssertionError as err:
        err = f'Error en la función convert_dcm_imgs.\n{err}'
        with open(os.path.join(LOGGING_DATA_PATH, f'General Errors.txt'), 'a') as f:
            f.write(f'{err}')

    except Exception as err:
        with open(os.path.join(LOGGING_DATA_PATH, f'conversion_{get_filename(img_path)}.txt'), 'w') as f:
            f.write(err)


def convert_dcm_imgs(ori_path: io, dest_path: io) -> None:
    """
    Función encargada de leer imagenes en formato dcm y convertirlas al formato especificado por el usuario.
    """
    try:
        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        assert os.path.splitext(dest_path)[1] in ['.png', '.jpg'], f'Conversion only available for: png, jpg'

        # se crea el directorio y sus subdirectorios en caso de no existir
        Path(get_dirname(dest_path)).mkdir(parents=True, exist_ok=True)

        # Se lee la información de las imagenes en formato dcm
        img = pydicom.dcmread(ori_path)

        # Se convierte las imagenes a formato de array
        img_array = img.pixel_array

        # Se almacena la imagen
        cv2.imwrite(dest_path, img_array)

    except AssertionError as err:
        err = f'Error en la función convert_dcm_imgs.\n{err}'
        with open(os.path.join(LOGGING_DATA_PATH, f'General Errors.txt'), 'a') as f:
            f.write(f'{err}')

    except Exception as err:
        with open(os.path.join(LOGGING_DATA_PATH, f'conversion_{get_filename(ori_path)}.txt'), 'w') as f:
            f.write(err)


def convert_pgm_imgs(ori_path: io, dest_path: io) -> None:
    """
    Función encargada de leer imagenes en formato pgm y convertirlas al formato especificado por el usuario.
    """
    try:
        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        assert os.path.splitext(dest_path)[1] in ['.png', '.jpg'], f'Conversion only available for: png, jpg'

        # se crea el directorio y sus subdirectorios en caso de no existir
        Path(get_dirname(dest_path)).mkdir(parents=True, exist_ok=True)

        # Se lee la información de las imagenes en formato pgm y se almacena en el formato deseado
        Image.open(ori_path).save(dest_path, os.path.splitext(dest_path)[1].replace('.', ''))

    except AssertionError as err:
        err = f'Error en la función convert_dcm_imgs.\n{err}'
        with open(os.path.join(LOGGING_DATA_PATH, f'General Errors.txt'), 'a') as f:
            f.write(f'{err}')

    except Exception as err:
        with open(os.path.join(LOGGING_DATA_PATH, f'conversion_{get_filename(ori_path)}.txt'), 'w') as f:
            f.write(err)


def crop_borders(img: np.ndarray, left: float = 0.01, right: float = 0.01, top: float = 0.01, bottom: float = 0.01) \
        -> np.ndarray:

    n_rows, n_cols, _ = img.shape

    left_crop, right_crop = int(n_cols * left), int(n_cols * (1 - right))
    top_crop, bottom_crop = int(n_rows * top), int(n_rows * (1 - bottom))

    return img[top_crop:bottom_crop, left_crop:right_crop, :]


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """

    :param img:
    :return:
    """
    # Se normaliza las imagenes para poder realizar la ecualización del histograma. Para ello, se aplica
    # como valor mínimo el 0 y como máximo el valor 255.
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def binarize_img(img: np.ndarray, thresh: Union[float, int] = 0.5, maxval: Union[float, int] = 1) -> np.ndarray:
    """

    Función utilizada para asignar el valor maxval a todos aquellos píxeles cuyo valor sea superior al threshold
    establecido.

    :param img: imagen a binarizar
    :param thresh: Valor de threshold. Todos aquellos valores inferiores al threshold se asignarán a 0. (negro)
    :param maxval: Valor asignado a todos aquellos valores superiores al threshold.
    :return: imagen binarizada.
    """
    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval

    return binarised_img


def edit_mask(mask: np.ndarray, kernel_size: tuple = (23, 23)) -> np.ndarray:
    """

    :param mask:
    :param kernel_size:
    :return:
    """

    # Se genera el kernel para realizar la transformación morgológica de la imagen.
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=kernel_size)

    # Se realiza una erosión seguida de una dilatación para eliminar el ruido situado en el fondo de la imagen
    # a la vez que posible ruido alrededor del pecho.
    edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Se realiza una dilatación para evitar emmascarar parte del seno.
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)

    return edited_mask


def get_breast_zone(mask: np.ndarray) -> Union[np.ndarray, tuple]:

    """
    Función encargada de encontrar los contornos de la imagen y retornar los top_x más grandes.

    :param mask: mascara sobre la cual se realizará la búsqueda de contornos y de las zonas más largas.

    :return: Máscara que contiene el contorno con mayor area.

    """

    # Se obtienen los contornos de las zonas de la imagen de color blanco.
    contours, _ = cv2.findContours(image=cv2.cvtColor(mask.copy(), cv2.COLOR_BGR2GRAY), mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_NONE)

    # Se obtiene el contorno más grande a partir del area que contiene
    largest_countour = sorted(contours, key=cv2.contourArea, reverse=True)[0:1]

    # Se crea la máscara con el area y el contorno obtenidos.
    breast_zone = cv2.drawContours(
        image=np.zeros(mask.shape, np.uint8), contours=largest_countour, contourIdx=-1, color=(1, 1, 1), thickness=-1
    )

    # Se obtiene el rectangulo que contiene el pecho
    x, y, w, h = cv2.boundingRect(largest_countour[0])

    return breast_zone, (x, y, w, h)


def remove_artifacts(img: np.ndarray, **kwargs) -> np.ndarray:
    """

    :param img:
    :param kwargs:
    :return:
    """

    # Se obtiene una mascara que permitirá elimianr los artefactos de la imágene obteniendo exclusivamente la parte
    # del seno. Para ello, primero se realiza una binarización de los datos para:
    #    1- Poner a negro el fondo de la imágen. Existen partes de la imágen que no son del todo negras y que podrían
    #       producir errores en el entrenamiento.
    #    2- Detectar las zonas pertenecientes al seno y a los artefactos asignandoles el valor de 1 mediante una
    #       binarización.s.
    bin_mask = binarize_img(img=img, **kwargs.get('bin_kwargs', {}))

    # La binarización de datos puede producir pérdida de información de los contornos del seno, por lo que se
    # incrementará el tamaño de la mascara. Adicionalmente, se pretende capturar de forma completa los artefactos
    # de la imagene para su eliminación suprimiendo posibles zonas de ruido.
    modified_mask = edit_mask(mask=bin_mask, **kwargs.get('mask_kwargs', {}))

    # Una vez identificados con un 1 tanto artefactos como el seno, se debe de identificar cual es la región
    # perteneciente al seno. Esta será la que tenga un área mayor.
    mask, (x, y, w, h) = get_breast_zone(mask=modified_mask)

    # Se aplica y se devuelve la mascara.
    img[mask == 0] = 0

    return img[y:y+h, x:x+w]


def flip_breast(img: np.ndarray, orient: str = 'left') -> np.ndarray:
    """
    Función utilizada para realizar el giro de los senos en caso de ser necesario. Esta funcionalidad pretende
    estandarizar las imagenes de forma que los flips se realizen posteriormente en el data augmentation-

    :param img: imagen para realizar el giro
    :param orient: 'left' o 'right'. Orientación que debe presentar el seno a la salida de la función
    :return: imagen girada.
    """

    # Se obtiene el número de columnas (número de píxeles) que contiene la imagen para obtener el centro.
    _, n_col, _ = img.shape

    # Se obtiene el centro dividiendo entre 2
    x_center = n_col // 2

    # Se suman los valores de cada columna y de cada hilera. En función del mayor valor obtenido, se obtendrá
    # la orientación del seno ya que el pecho está identificado por las zonas blancas (valor 1).
    left_side = img[:, :x_center].sum()
    right_side = img[:, x_center:].sum()

    # Si se desea que el seno esté orientado hacia la derecha y está orientado a la izquierda se girará.
    # Si se desea que el seno esté orientado hacia la izquierda y está orientado a la derecha se girará.
    cond = {'right': left_side > right_side, 'left': left_side < right_side}
    if cond[orient]:
        return cv2.flip(img.copy(), 1)
    # En caso contrario el seno estará bien orientado.
    else:
        return img


def apply_clahe_transform(img: np.ndarray, clip: int = 1) -> np.ndarray:
    """
    función que aplica una ecualización sobre la intensidad de píxeles de la imagen para mejorar el contraste
    de estas.
    :param img: imagen original sobre la cual realizar la ecualización adaptativa
    :return: imagen ecualizada
    """

    # Se transforma la imagen a escala de grises.
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    img_uint8 = gray.astype("uint8")

    # Se aplica la ecualización adaptativa del histograma de píxeles.
    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    clahe_img = clahe_create.apply(img_uint8)

    return clahe_img


def pad_image_into_square(img: np.ndarray) -> np.ndarray:
    """

    :param img:
    :return:
    """

    nrows, ncols, dim = img.shape

    padded_img = np.zeros(shape=(max(nrows, ncols), max(nrows, ncols), dim), dtype=np.uint8)
    padded_img[:nrows, :ncols, :] = img

    return padded_img


def resize_img(img: np.ndarray, size: tuple = (300, 300)) -> np.ndarray:
    """

    :param img:
    :param size:
    :return:
    """

    return cv2.resize(src=img.copy(), dsize=size, interpolation=cv2.INTER_LANCZOS4)