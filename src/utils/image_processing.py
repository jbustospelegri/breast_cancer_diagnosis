from pathlib import Path

import os
import pydicom
import pandas as pd
import numpy as np
import cv2

from typing import io, Union

from utils.config import LOGGING_DATA_PATH


def convert_dcm_imgs(args) -> None:
    """

    :param args:

    """
    img_path: io = args[0]
    dest_path: io = args[1]
    name: str = args[2]
    extension: str = args[3]
    aux: str = args[4]

    try:
        # se crea el directorio y sus subdirectorios en caso de no existir
        for img_type in ['CROP', 'FULL', 'MASK']:
            Path(os.path.join(dest_path, img_type)).mkdir(parents=True, exist_ok=True)

        # Se lee la información de las imagenes en formato dcm
        img = pydicom.dcmread(img_path)

        # Se convierte las imagenes a formato de array
        img_array = img.pixel_array

        # Se recuperan los distintos valores de píxeles que contiene la imagen. Este array servirá para diferenciar
        # entre imagenes recortadas o mascaras
        unique_pixel_values = np.unique(img_array).tolist()

        # En función del atributo SeriesDescription, el valor de píexeles único o bien, si la imagen termina en un
        # dígito o no (variable aux), se determinará si se trata de una imagen recortada, completa o una mascara
        if ('full' in getattr(img, 'SeriesDescription', [])) or (aux == 'FULL'):
            cv2.imwrite(os.path.join(dest_path, 'FULL', f'{name}.{extension}'), img_array)
        elif ('mask' in getattr(img, 'SeriesDescription', [])) or (len(unique_pixel_values) == 2):
            cv2.imwrite(os.path.join(dest_path, 'MASK', f'{name}.{extension}'), img_array)
        elif ('crop' in getattr(img, 'SeriesDescription', [])) or (len(unique_pixel_values) != 2):
            cv2.imwrite(os.path.join(dest_path, 'CROP', f'{name}.{extension}'), img_array)
        else:
            raise ValueError(f'Imagen {img_path} no clasificada correctamente en crop, mask o crop')

    except Exception as err:
        with open(os.path.join(LOGGING_DATA_PATH, f'{name}.txt'), 'w') as f:
            f.write(err)


def crop_borders(img: np.ndarray, left: float = 0.01, right: float = 0.01, top: float = 0.01, bottom: float = 0.01) \
        -> np.ndarray:

    n_rows, n_cols, _ = img.shape

    left_crop, right_crop = int(n_cols * left), int(n_cols * (1 - right))
    top_crop, bottom_crop = int(n_rows * top), int(n_rows * (1 - bottom))

    return img[top_crop:bottom_crop, left_crop:right_crop, :]


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    return (img - img.min()) / (img.max() - img.min())


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
    :param operation:
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


def get_breast_zone(mask: np.ndarray) -> np.ndarray:

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

    return breast_zone


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
    mask = get_breast_zone(mask=modified_mask)

    # Se aplica y se devuelve la mascara.
    img[mask == 0] = 0

    return img

