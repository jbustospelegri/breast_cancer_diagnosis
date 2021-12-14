from itertools import repeat

import numpy as np
import cv2
import os

from typing import io
from PIL import Image

from preprocessing.functions import (
    apply_clahe_transform, remove_artifacts, remove_noise, crop_borders, pad_image_into_square, resize_img, binarize,
    normalize_breast, flip_breast, correct_axis
)
from src.utils.config import LOGGING_DATA_PATH, PREPROCESSING_FUNCS, PREPROCESSING_CONFIG
from src.utils.functions import get_filename, save_img, get_path, get_value_from_args_if_exists, get_dirname, \
    get_contours


def full_image_pipeline(args) -> None:
    """
    Función utilizada para realizar el preprocesado de las mamografías. Este preprocesado consiste en:
        1 - Recortar los bordes de las imagenes.
        2 - Realziar una normalización min-max para estandarizar las imagenes a 8 bits.
        3 - Quitar anotaciones realziadas sobre las iamgenes.
        4 - Relizar un flip horizontal para estandarizar la orientacion de los senos.
        5 - Mejorar el contraste de las imagenes en blanco y negro  mediante CLAHE.
        6 - Recortar las imagenes para que queden cuadradas.
        7 - Normalización min-max para estandarizar el valor de los píxeles entre 0 y 255
        8 - Resize de las imagenes a un tamaño de 300 x 300

    :param args: listado de argumentos cuya posición debe ser:
        1 - path de la imagen sin procesar.
        2 - path de destino de la imagen procesada.
        3 - extensión con la que se debe de almacenar la imagen
        4 - directorio en el que se deben de almacenar los ejemplos.
    """

    try:

        # Se recuperan los valores de arg. Deben de existir los 3 argumentos obligatorios.
        if not (len(args) >= 2):
            raise IndexError('Not enough arguments for convert_dcm_img function. Minimum required arguments: 2')

        img_filepath: io = args[0]
        dest_dirpath: io = args[1]

        save_intermediate_steps = get_value_from_args_if_exists(args, 2, False, IndexError, TypeError)
        img_mask_out_path = get_value_from_args_if_exists(args, 3, None, IndexError, TypeError)
        img_mask_filepath = get_value_from_args_if_exists(args, 4, None, IndexError, TypeError)

        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        if not os.path.isfile(img_filepath):
            raise FileNotFoundError(f'The image {img_filepath} does not exists.')
        if os.path.splitext(dest_dirpath)[1] not in ['.png', '.jpg']:
            raise ValueError(f'Conversion only available for: png, jpg')

        assert not os.path.isfile(dest_dirpath), f'Processing file exists: {dest_dirpath}'

        # Se almacena la configuración del preprocesado
        prep_dict = PREPROCESSING_FUNCS[PREPROCESSING_CONFIG]

        # Se lee la imagen original sin procesar.
        img = cv2.cvtColor(cv2.imread(img_filepath), cv2.COLOR_BGR2GRAY)

        # Se lee la mascara
        if img_mask_filepath is None:
            img_mask = np.ones(shape=img.shape, dtype=np.uint8)
        else:
            if not os.path.isfile(img_mask_filepath):
                raise FileNotFoundError(f'The mask {img_mask_filepath} does not exists.')
            img_mask = cv2.cvtColor(cv2.imread(img_mask_filepath), cv2.COLOR_BGR2GRAY)

        images = {'ORIGINAL': img}

        # Primero se realiza un crop de las imagenes en el caso de que sean imagenes completas
        images['CROPPING 1'] = crop_borders(images[list(images.keys())[-1]].copy(), **prep_dict.get('CROPPING_1', {}))

        # Se aplica el mismo procesado a la mascara
        img_mask = crop_borders(img_mask, **prep_dict.get('CROPPING_1', {}))

        # A posteriori se quita el ruido de las imagenes utilizando un filtro medio
        images['REMOVE NOISE'] = remove_noise(
            img=images[list(images.keys())[-1]].copy(), **prep_dict.get('REMOVE_NOISE', {}))

        # El siguiente paso consiste en eliminar los artefactos de la imagen. Solo aplica a imagenes completas
        images['REMOVE ARTIFACTS'], _, mask, img_mask = remove_artifacts(
            img=images[list(images.keys())[-1]].copy(), mask=img_mask, **prep_dict.get('REMOVE_ARTIFACTS', {})
        )

        # Una vez eliminados los artefactos, se realiza una normalización de la zona del pecho
        images['IMAGE NORMALIZED'] = \
            normalize_breast(images[list(images.keys())[-1]].copy(), mask, **prep_dict.get('NORMALIZE_BREAST', {}))

        # A continuación se realiza aplica un conjunto de ecualizaciones la imagen. El número máximo de ecualizaciones
        # a aplicar son 3 y serán representadas enc ada canal
        ecual_imgs = []
        img_to_ecualize = images[list(images.keys())[-1]].copy()
        assert 0 < len(prep_dict['ECUALIZATION'].keys()) < 4, 'Número de ecualizaciones incorrecto'
        for i, (ecual_func, ecual_kwargs) in enumerate(prep_dict['ECUALIZATION'].items(), 1):

            if 'CLAHE' in ecual_func.upper():
                images[ecual_func.upper()] = apply_clahe_transform(img=img_to_ecualize, mask=mask, **ecual_kwargs)
                ecual_imgs.append(images[list(images.keys())[-1]].copy())

            elif 'GCN' in ecual_func.upper():
                pass

        if len(prep_dict['ECUALIZATION'].keys()) == 2:
            images['IMAGES SYNTHESIZED'] = cv2.merge((img_to_ecualize, *ecual_imgs))
        elif len(prep_dict['ECUALIZATION'].keys()) == 3:
            images['IMAGES SYNTHESIZED'] = cv2.merge(tuple(ecual_imgs))

        # Se realiza el flip de la imagen en caso de ser necesario:
        images['IMG_FLIP'], flip = flip_breast(images[list(images.keys())[-1]].copy(), **prep_dict.get('FLIP_IMG', {}))

        if flip:
            img_mask = cv2.flip(src=img_mask, flipCode=1)

        # Se aplica el ultimo crop de la parte izquierda
        # Primero se realiza un crop de las imagenes
        images['CROPPING LEFT'] = crop_borders(images[list(images.keys())[-1]].copy(), **prep_dict.get('CROPPING_2', {}))

        img_mask = crop_borders(img=img_mask,  **prep_dict.get('CROPPING_2', {}))

        # # Se aplica el padding de las imagenes para convertirlas en imagenes cuadradas
        if prep_dict.get('RATIO_PAD', False):
            images['IMAGE RATIO PADDED'] = \
                pad_image_into_square(img=images[list(images.keys())[-1]].copy(), **prep_dict.get('RATIO_PAD', {}))
            img_mask = pad_image_into_square(img=img_mask, **prep_dict.get('RATIO_PAD', {}))

        # Se aplica el resize de la imagen:
        if prep_dict.get('RESIZING', False):
            images['IMAGE RESIZED'] = \
                resize_img(img=images[list(images.keys())[-1]].copy(), **prep_dict.get('RESIZING', {}))
            img_mask = resize_img(img=img_mask, **prep_dict.get('RESIZING', {}), interpolation=cv2.INTER_NEAREST)

        if save_intermediate_steps:
            for i, (name, imag) in enumerate(images.items()):
                save_img(imag, get_dirname(dest_dirpath), f'{i}. {name}')

        if img_mask_out_path and len(get_contours(img_mask)) > 0:
            Image.fromarray(np.uint8(img_mask)).save(img_mask_out_path)

        # Se almacena la imagen definitiva
        Image.fromarray(np.uint8(images[list(images.keys())[-1]].copy())).save(dest_dirpath)

    except AssertionError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Preprocessing Errors (Assertions).txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except IndexError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Preprocessing Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nError calling function convert_dcm_img pipeline\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(LOGGING_DATA_PATH, f'Preprocessing Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(img_filepath)}\n{err}\n{"=" * 100}')


def crop_image_pipeline(args) -> None:
    """
    Función utilizada para realizar el preprocesado de las mamografías. Este preprocesado consiste en:
        1 - Recortar los bordes de las imagenes.
        2 - Realziar una normalización min-max para estandarizar las imagenes a 8 bits.
        3 - Quitar anotaciones realziadas sobre las iamgenes.
        4 - Relizar un flip horizontal para estandarizar la orientacion de los senos.
        5 - Mejorar el contraste de las imagenes en blanco y negro  mediante CLAHE.
        6 - Recortar las imagenes para que queden cuadradas.
        7 - Normalización min-max para estandarizar el valor de los píxeles entre 0 y 255
        8 - Resize de las imagenes a un tamaño de 300 x 300

    :param args: listado de argumentos cuya posición debe ser:
        1 - path de la imagen sin procesar.
        2 - path de destino de la imagen procesada.
        3 - extensión con la que se debe de almacenar la imagen
        4 - directorio en el que se deben de almacenar los ejemplos.
    """

    try:
        # Se recuperan los valores de arg. Deben de existir los 3 argumentos obligatorios.
        if not (len(args) >= 3):
            raise ValueError('Not enough arguments for convert_dcm_img function. Minimum required arguments: 5')

        img_filepath: io = args[0]
        out_filepath: io = args[1]
        extension: str = os.path.splitext(out_filepath)[1]
        roi_mask_path: io = args[2]

        n_background_imgs: int = get_value_from_args_if_exists(args, 3, 0, IndexError, TypeError)
        n_roi_imgs: int = get_value_from_args_if_exists(args, 4, 1, IndexError, TypeError)
        overlap_roi: float = get_value_from_args_if_exists(args, 5, 1.0, IndexError, TypeError)
        margin_roi: float = get_value_from_args_if_exists(args, 6, 1.0, IndexError, TypeError)
        save_intermediate_steps: bool = get_value_from_args_if_exists(args, 7, False, IndexError, TypeError)

        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        if not os.path.isfile(img_filepath):
            raise FileNotFoundError(f'The image {img_filepath} does not exists.')
        if not os.path.isfile(roi_mask_path):
            raise FileNotFoundError(f'The image {roi_mask_path} does not exists.')
        if extension not in ['.png', '.jpg']:
            raise ValueError(f'Conversion only available for: png, jpg')

        # Se almacena la configuración del preprocesado
        prep_dict = PREPROCESSING_FUNCS[PREPROCESSING_CONFIG]

        # Se lee la imagen original sin procesar.
        img = cv2.cvtColor(cv2.imread(img_filepath), cv2.COLOR_BGR2GRAY)

        # Se lee la mascara
        mask = cv2.cvtColor(cv2.imread(roi_mask_path), cv2.COLOR_BGR2GRAY)

        # Primero se realiza un crop de las imagenes en el caso de que sean imagenes completas
        crop_img = crop_borders(img, **prep_dict.get('CROPPING_1', {}))

        # Se aplica el mismo procesado a la mascara
        img_mask = crop_borders(mask, **prep_dict.get('CROPPING_1', {}))

        # A posteriori se quita el ruido de las imagenes utilizando un filtro medio
        img_denoised = remove_noise(crop_img, **prep_dict.get('REMOVE_NOISE', {}))

        # Se obtienen las zonas de patologia de la mascara juntamente con las
        # El siguiente paso consiste en eliminar los artefactos de la imagen. Solo aplica a imagenes completas
        _, _, breast_mask, mask = remove_artifacts(img_denoised, img_mask, False, **prep_dict.get('REMOVE_ARTIFACTS', {}))

        # Se obtienen los parches de las imagenes con patologías.
        roi_zones = []
        mask_zones = []
        breast_zone = breast_mask.copy()
        for contour in get_contours(img=mask):
            x, y, w, h = cv2.boundingRect(contour)

            if (h > 15) & (w > 15):
                center = (y + h // 2, x + w // 2)
                y_min, x_min = int(center[0] - h * margin_roi // 2), int(center[1] - w * margin_roi // 2)
                y_max, x_max = int(center[0] + h * margin_roi // 2), int(center[1] + w * margin_roi // 2)
                x_max, x_min, y_max, y_min = correct_axis(img_denoised.shape, x_max, x_min, y_max, y_min)
                roi_zones.append(img_denoised[y_min:y_max, x_min:x_max])
                mask_zones.append(breast_zone[y_min:y_max, x_min:x_max])

                # Se suprimen las zonas de la patología para posteriormente obtener la zona del background
                cv2.rectangle(breast_mask, (x_min, y_min), (x_max, y_max), color=(0, 0, 0), thickness=-1)

        # TODO: Se obtienen los parches de las zonas sin patología

        # Se procesan las zonas de interes recortadas
        for idx, (roi, roi_mask, tipo) in enumerate(zip(roi_zones, mask_zones, repeat('roi', len(roi_zones)))):

            roi_norm = normalize_breast(roi, roi_mask, **prep_dict.get('NORMALIZE_BREAST', {}))

            # A continuación se realiza aplica un conjunto de ecualizaciones la imagen. El número máximo de
            # ecualizaciones a aplicar son 3 y serán representadas enc ada canal
            ecual_imgs = []
            img_to_ecualize = roi_norm.copy()
            assert 0 < len(prep_dict['ECUALIZATION'].keys()) < 4, 'Número de ecualizaciones incorrecto'
            for i, (ecual_func, ecual_kwargs) in enumerate(prep_dict['ECUALIZATION'].items(), 1):

                if 'CLAHE' in ecual_func.upper():
                    ecual_imgs.append(apply_clahe_transform(img_to_ecualize, roi_mask, **ecual_kwargs))

                elif 'GCN' in ecual_func.upper():
                    pass

            if len(prep_dict['ECUALIZATION'].keys()) == 2:
                roi_synthetized = cv2.merge((img_to_ecualize, *ecual_imgs))
            elif len(prep_dict['ECUALIZATION'].keys()) == 3:
                roi_synthetized = cv2.merge(tuple(ecual_imgs))
            else:
                roi_synthetized = ecual_imgs[-1]

            # Se almacena la imagen definitiva
            path = get_path(get_dirname(out_filepath), f'{tipo}_{get_filename(out_filepath)}_{idx}{extension}')
            Image.fromarray(np.uint8(roi_synthetized)).save(path)

    except AssertionError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Preprocessing Errors (Assertions).txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(LOGGING_DATA_PATH, f'Preprocessing Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(img_filepath)}\n{err}\n{"=" * 100}')

