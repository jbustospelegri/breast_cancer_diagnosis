import io
import os
import cv2
import plistlib

import numpy as np
import pydicom

from skimage.draw import polygon
from PIL import Image

from utils.config import INBREAST_DB_XML_ROI_PATH, LOGGING_DATA_PATH, CBIS_DDSM_DB_PATH
from utils.functions import load_point, get_path, get_filename, search_files, get_value_from_args_if_exists


def get_inbreast_roi_mask(args) -> None:
    """
    Función para obtener las máscaras para el set de datos INBreast
    :param args: Lista de argumentos que contendra según las siguientes posiciones:
            1 - Path de la imagen original
            2 - Path del fichero XML con las anataciones de las mascaras
            3 - Path para guardar la máscara generada
    """
    try:
        if not len(args) == 3:
            raise ValueError('Incorrect number of arguments for function get_inbrest_roi_mask')

        ori_img = args[0]
        xlsm_file = args[1]
        out_file = args[2]

        if not os.path.isfile(ori_img):
            raise FileNotFoundError(f'{ori_img} image does not exists.')

        if not os.path.isfile(get_path(INBREAST_DB_XML_ROI_PATH, f'{xlsm_file}.xml')):
            raise FileNotFoundError(f'{xlsm_file} xlsm does not exists.')

        assert not os.path.isfile(out_file), f'Mask {out_file} already created.'

        # Se recuperan las dimensiones de la imagen original para crear una imágen negra con las mismas dimensiones
        shape = cv2.imread(ori_img).shape

        mask = np.zeros(shape)

        # Se leen los ficheros XML recuperando la lista de anotaciones a nivel de píxeles contenidas en la key 'ROIS'
        # Unicamente se recuperarán las lesiones que sean masa.
        with open(get_path(INBREAST_DB_XML_ROI_PATH, f'{xlsm_file}.xml'), 'rb') as f:
            plist_dict = plistlib.load(f, fmt=plistlib.FMT_XML)['Images'][0]
            for roi in plist_dict['ROIs']:
                if roi['Name'] in ['Mass']:
                    x, y = zip(*[load_point(point) for point in roi['Point_px']])
                    # Se obtienen las tuplas x e y de cada anotación para pintarlas de blanco
                    poly_x, poly_y = polygon(np.array(x), np.array(y), shape=shape)
                    mask[poly_x, poly_y] = 255

        Image.fromarray(np.uint8(mask)).save(out_file)

    except AssertionError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors (Assertions).txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except ValueError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nError calling function get_inbrest_roi_mask pipeline\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(ori_img)}\n{err}\n{"=" * 100}')


def get_mias_roi_mask(args) -> None:
    """
    Función para obtener las máscaras del set de datos MIAS
    :param args: Lista de argumentos que contendra según las siguientes posiciones:
            1 - Path de la imagen original
            2 - coordenada X de la lesion (eje de coordenadas margen inferior izquierdo de la imagen)
            3 - Coordenada Y de la lesión (eje de coordenadas margen inferior izquierdo de la imagen)
            4 - Radio del circulo que contiene la lesión.
    """

    try:
        if len(args) != 4:
            raise ValueError('Incorrect number of args for function get_mias_roi')

        assert not os.path.isfile(args[0]), f'Mask {args[0]} already created.'

        # Se crea una máscara de tamaño 1024x1024 (resolución de las imágenes de MIAS).
        mask = np.zeros(shape=(1024, 1024), dtype=np.uint8)
        # Se dibuja un circulo blanco en las coordenadas indicadas. Si una imagen contiene multiples ROI's, se dibujan
        # todas las zonas.
        for x, y, rad in zip(args[1], args[2], args[3]):
            cv2.circle(mask, center=(int(x), int(y)), radius=int(rad), thickness=-1, color=(255, 255, 255))
        cv2.imwrite(args[0], mask)

    except AssertionError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors (Assertions).txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except ValueError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nError calling function get_inbrest_roi_mask pipeline\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(get_filename(args[0]))}\n{err}\n{"=" * 100}')


def get_cbis_roi_mask(args) -> None:
    """
    Función para obtener las máscaras del set de datos CBIS - DDSM
    :param args: Lista de argumentos que contendra según las siguientes posiciones:
            1 - Path de la imagen original
            2 - Path para guardar la máscara generada
    """
    try:
        if len(args) != 2:
            raise ValueError('Incorrect number of arguments for function get_cbis_roi_mask')

        assert not os.path.isfile(args[1]), f'Mask {args[1]} already created.'

        # Dado que una misma imagen puede contener multiples lesiones informadas mediante sufijos _N siendo N un entero
        # se recuperan todas las máscaras de ROIS para una misma imágen mamográfica.
        masks = []
        for img in search_files(file=get_path(CBIS_DDSM_DB_PATH, f'{args[0]}*_[0-9]'), ext='dcm'):

            # Se lee la información de las imagenes en formato dcm
            img = pydicom.dcmread(img)

            # Se convierte las imagenes a formato de array
            img_array = img.pixel_array.astype(float)

            # Las imagenes binarias únicamente pueden contener dos valroes de pixel distintos
            if len(np.unique(img_array)) == 2:

                # Se realiza un reescalado de la imagen para obtener los valores entre 0 y 255
                rescaled_image = (np.maximum(img_array, 0) / max(img_array)) * 255

                # Se limpian mascaras que sean de menos de 10 píxeles
                _, _, h, w = cv2.boundingRect(np.uint8(rescaled_image))

                if (h > 10) and (w > 10):
                    # Se convierte la imagen al ipode datos unsigned de 8 bytes
                    masks.append(np.uint8(rescaled_image))

        # Las distintas mascaras se sumarán para obtener una única máscara por mamografia
        final_mask = sum(masks)
        final_mask[final_mask > 1] = 255

        # Se almacena la mascara
        cv2.imwrite(args[1], final_mask)

    except AssertionError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors (Assertions).txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except ValueError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nError calling function get_cbis_roi_mask pipeline\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(args[1])}\n{err}\n{"=" * 100}')


def get_test_mask(args) -> None:
    """
    Función para dibujar las máscaras en el deployment de la aplicación
    :param args: Lista de argumentos que contendra según las siguientes posiciones:
            1 - Path de la imagen original
            2 - Path de la máscara de salida
            3 - coordenada X de la lesion (eje de coordenadas margen inferior izquierdo de la imagen)
            4 - Coordenada Y de la lesión (eje de coordenadas margen inferior izquierdo de la imagen)
            5 - Radio del circulo que contiene la lesión.
            6 - Filepath del directorio en el cual escribir los errores producidos.
    """

    error_path: io = get_value_from_args_if_exists(args, 5, LOGGING_DATA_PATH, IndexError, KeyError)

    try:
        if len(args) < 5:
            raise ValueError('Incorrect number of args for function get_mias_roi')

        img_io_in = args[0]
        mask_io_out = args[1]
        x_cord = args[2]
        y_cord = args[3]
        rad = args[4]

        # Se revisa que la imagen original exista
        if not os.path.isfile(img_io_in):
            raise FileNotFoundError(f'{img_io_in} not found')

        # Se revisa que la máscara no haya sido creada previamente
        assert not os.path.isfile(mask_io_out), f'Mask {mask_io_out} already created.'

        # Tamaño de la imagen original
        shape = cv2.imread(img_io_in).shape[:2]

        # Validación de las coordenadas x, y e radio
        if not 0 <= x_cord <= shape[1]:
            raise ValueError(f'{x_cord} is outside available pixels in image')

        if not 0 <= y_cord <= shape[0]:
            raise ValueError(f'{x_cord} is outside available pixels in image')

        if rad <= 0:
            raise ValueError(f'Incorrect value for {rad}')

        # Se modifica la coordenada Y en función del sistema de referencia de cv2
        y_cord = shape[0] - y_cord

        # Se crea una imagen en negro con la resolución de la imagen original
        mask = np.zeros(shape=shape, dtype=np.uint8)
        # Se dibuja un circulo en blanco en la zona interés de cada mamografía
        cv2.circle(mask, center=(int(x_cord), int(y_cord)), radius=int(rad), thickness=-1, color=(255, 255, 255))
        cv2.imwrite(mask_io_out, mask)

    except AssertionError as err:
        with open(get_path(error_path, f'Conversion Errors (Assertions).txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except ValueError as err:
        with open(get_path(error_path, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nError calling function get_test_mask pipeline\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(error_path, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(get_filename(args[0]))}\n{err}\n{"=" * 100}')
