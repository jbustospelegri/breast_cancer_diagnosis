import numpy as np
import cv2
import os

from typing import io, Union, Tuple, Any
from pathlib import Path

from PIL import Image

from utils.config import LOGGING_DATA_PATH, PREPROCESSING_FUNCS, PREPROCESSING_CONFIG
from utils.functions import get_filename, save_img, get_path, detect_func_err


def image_pipeline(args) -> None:
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
        assert len(args) >= 2, 'Not enough arguments for convert_dcm_img function. Minimum required arguments: 3'

        img_filepath: io = args[0]
        dest_dirpath: io = args[1]
        mode: str = args[2]

        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        assert os.path.isfile(img_filepath), f'The image {img_filepath} does not exists.'
        assert os.path.splitext(dest_dirpath)[1] in ['.png', '.jpg'], f'Conversion only available for: png, jpg'
        assert not os.path.isfile(dest_dirpath), f'Processing file exists: {dest_dirpath}'

        # Se asigna el cuarto argumento en función de su existencia. En caso contrario se asignan valores por
        # defecto
        try:
            save_example_dirname: io = args[3]
            assert os.path.isdir(save_example_dirname)
        except AssertionError:
            Path(save_example_dirname).mkdir(parents=True, exist_ok=True)
        except (IndexError, TypeError):
            save_example_dirname = None

        # Se almacena la configuración del preprocesado
        prep_dict = PREPROCESSING_FUNCS[PREPROCESSING_CONFIG]

        # Se lee la imagen original sin procesar.
        img = cv2.cvtColor(cv2.imread(img_filepath), cv2.COLOR_BGR2GRAY)

        images = {'ORIGINAL': img}

        # Primero se realiza un crop de las imagenes en el caso de que sean imagenes completas
        if mode == 'FULL':
            images['CROPPING 1'] = crop_borders(
                images[list(images.keys())[-1]].copy(), **prep_dict.get('CROPPING_1', {}))

        # A posteriori se quita el ruido de las imagenes utilizando un filtro medio
        images['REMOVE NOISE'] = remove_noise(
            img=images[list(images.keys())[-1]].copy(), **prep_dict.get('REMOVE_NOISE', {}))

        # El siguiente paso consiste en eliminar los artefactos de la imagen. Solo aplica a imagenes completas
        if mode == 'FULL':
            images['REMOVE ARTIFACTS'], bin_mask, modified_mask, mask = remove_artifacts(
                img=images[list(images.keys())[-1]].copy(), **prep_dict.get('REMOVE_ARTIFACTS', {}))
        else:
            mask = np.ones(images[list(images.keys())[-1]].shape)

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

            elif 'gcn' in ecual_func:
                pass

        if len(prep_dict['ECUALIZATION'].keys()) == 2:
            images['IMAGES SYNTHESIZED'] = cv2.merge((img_to_ecualize, *ecual_imgs))
        elif len(prep_dict['ECUALIZATION'].keys()) == 3:
            images['IMAGES SYNTHESIZED'] = cv2.merge(tuple(ecual_imgs))

        # Se realiza el flip de la imagen en caso de ser necesario:
        if mode == 'FULL':
            images['IMG_FLIP'] = flip_breast(
                img=images[list(images.keys())[-1]].copy(), **prep_dict.get('FLIP_IMG', {})
            )

        # Se aplica el padding de las imagenes para convertirlas en imagenes cuadradas
        if (mode == 'FULL') and (prep_dict.get('SQUARE_PAD', False)):
            images['IMAGE PADDED'] = pad_image_into_square(img=images[list(images.keys())[-1]].copy())

        # Se aplica el resize de la imagen:
        if prep_dict.get('RESIZING', False):
            images['IMAGE RESIZED'] = \
                resize_img(img=images[list(images.keys())[-1]].copy(), **prep_dict.get('RESIZING', {}))

        # Se aplica el ultimo crop de la parte izquierda
        # Primero se realiza un crop de las imagenes
        if mode == 'FULL':
            images['CROPPING LEFT'] = crop_borders(
                images[list(images.keys())[-1]].copy(), **prep_dict.get('CROPPING_2', {}))

        for i, (name, imag) in enumerate(images.items()):
            save_img(imag, save_example_dirname, f'{i}. {name}')

        # Se almacena la imagen definitiva
        Image.fromarray(np.uint8(images[list(images.keys())[-1]].copy())).save(dest_dirpath)

    except AssertionError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Preprocessing Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(LOGGING_DATA_PATH, f'Preprocessing Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(img_filepath)}\n{err}\n{"=" * 100}')


@detect_func_err
def remove_noise(img: np.ndarray, **kwargs) -> np.ndarray:

    img_without_noise = img.copy()

    # Se suprime el ruido aditivo
    img_without_noise = cv2.medianBlur(img_without_noise, **kwargs)

    return img_without_noise


@detect_func_err
def crop_borders(img: np.ndarray, left: float = 0.01, right: float = 0.01, top: float = 0.01, bottom: float = 0.01) \
        -> np.ndarray:

    if len(img.shape) == 2:
        n_rows, n_cols = img.shape

        left_crop, right_crop = int(n_cols * left), int(n_cols * (1 - right))
        top_crop, bottom_crop = int(n_rows * top), int(n_rows * (1 - bottom))

        return img[top_crop:bottom_crop, left_crop:right_crop]
    else:
        n_rows, n_cols, _ = img.shape

        left_crop, right_crop = int(n_cols * left), int(n_cols * (1 - right))
        top_crop, bottom_crop = int(n_rows * top), int(n_rows * (1 - bottom))

        return img[top_crop:bottom_crop, left_crop:right_crop, :]


@detect_func_err
def normalize_breast(img: np.ndarray, mask: np.ndarray = None, type_norm: str = 'min_max') -> np.ndarray:
    """

    :param img:
    :return:
    """

    # Se transforma la imagen a float para no perder informacion
    img_float = img.copy().astype(float)

    if mask is None:
        mask = np.ones(img.shape)

    if type_norm == 'min_max':
        min = img_float[mask != 0].min()
        max = img_float[mask != 0].max()

    elif type_norm == 'truncation':
        min = np.percentile(img_float[mask != 0], 10)
        max = np.percentile(img_float[mask != 0], 99)

    else:
        raise ValueError(f'{type_norm} not implemented in normalize_breast')

    img_norm = ((np.clip(img_float, min, max) - min) / (max - min)) * 255
    img_norm[mask == 0] = 0

    return np.uint8(img_norm)


@detect_func_err
def binarize(img: np.ndarray, thresh: str = 'otsu', threshval: int = 1) -> np.ndarray:
    """

    Función utilizada para asignar el valor maxval a todos aquellos píxeles cuyo valor sea superior al threshold
    establecido.

    :param img: imagen a binarizar
    :param thresh: Valor de threshold. Todos aquellos valores inferiores al threshold se asignarán a 0. (negro)
    :param maxval: Valor asignado a todos aquellos valores superiores al threshold.
    :return: imagen binarizada.
    """

    # Primero se aplica un filtro adaptativo para crear una binarización por thresholding
    if thresh == 'otsu':
        return cv2.threshold(img, threshval, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    elif thresh == 'constant':
        return cv2.threshold(img, threshval, 255, cv2.THRESH_BINARY)[1]
    else:
        raise ValueError(f'Thresholding method {thresh} not implemented')


@detect_func_err
def edit_mask(mask: np.ndarray, operations: tuple, kernel_size: tuple = (23, 23),
              kernel_shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    """

    :param mask:
    :param kernel_size:
    :return:
    """

    # Se genera el kernel para realizar la transformación morgológica de la imagen.
    kernel = cv2.getStructuringElement(shape=kernel_shape, ksize=kernel_size)

    # Se realiza una erosión seguida de una dilatación para eliminar el ruido situado en el fondo de la imagen
    # a la vez que posible ruido alrededor del pecho.
    for (transformation, iters) in operations:
        cv2.morphologyEx(mask, transformation, kernel, iterations=iters, dst=mask)

    return mask


@detect_func_err
def get_breast_zone(mask: np.ndarray, convex_contour: bool = False) -> Union[np.ndarray, tuple]:

    """
    Función encargada de encontrar los contornos de la imagen y retornar los top_x más grandes.

    :param mask: mascara sobre la cual se realizará la búsqueda de contornos y de las zonas más largas.

    :return: Máscara que contiene el contorno con mayor area.

    """

    # Se obtienen los contornos de las zonas de la imagen de color blanco.
    contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    # Se obtiene el contorno más grande a partir del area que contiene
    largest_countour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    if convex_contour:
        largest_countour = cv2.convexHull(largest_countour)

    # Se crea la máscara con el area y el contorno obtenidos.
    breast_zone = cv2.drawContours(
        image=np.zeros(mask.shape, np.uint8), contours=[largest_countour], contourIdx=-1, color=(255, 255, 255),
        thickness=-1
    )

    # Se obtiene el rectangulo que contiene el pecho
    x, y, w, h = cv2.boundingRect(largest_countour)

    return breast_zone, (x, y, w, h)


@detect_func_err
def remove_artifacts(img: np.ndarray, **kwargs) -> Tuple[Any, np.ndarray, np.ndarray, Any]:
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
    bin_mask = binarize(img=img, **kwargs.get('bin_kwargs', {}))

    # La binarización de datos puede producir pérdida de información de los contornos del seno, por lo que se
    # incrementará el tamaño de la mascara. Adicionalmente, se pretende capturar de forma completa los artefactos
    # de la imagene para su eliminación suprimiendo posibles zonas de ruido.
    modified_mask = edit_mask(mask=bin_mask, **kwargs.get('mask_kwargs', {}))

    # Una vez identificados con un 1 tanto artefactos como el seno, se debe de identificar cual es la región
    # perteneciente al seno. Esta será la que tenga un área mayor.
    mask, (x, y, w, h) = get_breast_zone(mask=modified_mask, **kwargs.get('contour_kwargs', {}))

    # Se aplica y se devuelve la mascara.
    img[mask == 0] = 0

    return img[y:y + h, x:x + w], bin_mask, modified_mask, mask[y:y + h, x:x + w]


@detect_func_err
def flip_breast(img: np.ndarray, orient: str = 'left') -> np.ndarray:
    """
    Función utilizada para realizar el giro de los senos en caso de ser necesario. Esta funcionalidad pretende
    estandarizar las imagenes de forma que los flips se realizen posteriormente en el data augmentation-

    :param img: imagen para realizar el giro
    :param orient: 'left' o 'right'. Orientación que debe presentar el seno a la salida de la función
    :return: imagen girada.
    """

    # Se obtiene el número de columnas (número de píxeles) que contiene la imagen para obtener el centro.
    if len(img.shape) == 2:
        _, n_col = img.shape
        # Se obtiene el centro dividiendo entre 2
        x_center = n_col // 2

        # Se suman los valores de cada columna y de cada hilera. En función del mayor valor obtenido, se obtendrá
        # la orientación del seno ya que el pecho está identificado por las zonas blancas (valor 1).
        left_side = img[:, :x_center].sum()
        right_side = img[:, x_center:].sum()

    else:
        _, n_col, _ = img.shape

        # Se obtiene el centro dividiendo entre 2
        x_center = n_col // 2

        # Se suman los valores de cada columna y de cada hilera. En función del mayor valor obtenido, se obtendrá
        # la orientación del seno ya que el pecho está identificado por las zonas blancas (valor 1).
        left_side = img[:, :x_center, :].sum()
        right_side = img[:, x_center:, :].sum()

    # Si se desea que el seno esté orientado hacia la derecha y está orientado a la izquierda se girará.
    # Si se desea que el seno esté orientado hacia la izquierda y está orientado a la derecha se girará.
    cond = {'right': left_side > right_side, 'left': left_side < right_side}
    if cond[orient]:
        return cv2.flip(img.copy(), 1)
    # En caso contrario el seno estará bien orientado.
    else:
        return img


@detect_func_err
def apply_clahe_transform(img: np.ndarray, mask: np.ndarray = None, clip: int = 1) -> np.ndarray:
    """
    función que aplica una ecualización sobre la intensidad de píxeles de la imagen para mejorar el contraste
    de estas.
    :param img: imagen original sobre la cual realizar la ecualización adaptativa
    :return: imagen ecualizada
    """

    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    clahe_img = clahe_create.apply(img)

    if mask is not None:
        # Se aplica la ecualización adaptativa del histograma de píxeles.
        clahe_img[mask == 0] = 0

    return clahe_img


@detect_func_err
def pad_image_into_square(img: np.ndarray) -> np.ndarray:
    """

    :param img:
    :return:
    """

    if len(img.shape) == 2:
        nrows, ncols = img.shape

        padded_img = np.zeros(shape=(max(nrows, ncols), max(nrows, ncols)), dtype=np.uint8)
        padded_img[:nrows, :ncols] = img

        return padded_img
    else:
        nrows, ncols, d = img.shape

        padded_img = np.zeros(shape=(max(nrows, ncols), max(nrows, ncols), d), dtype=np.uint8)
        padded_img[:nrows, :ncols, :] = img

        return padded_img


@detect_func_err
def resize_img(img: np.ndarray, size: tuple = (300, 300)) -> np.ndarray:
    """

    :param img:
    :param size:
    :return:
    """

    return cv2.resize(src=img.copy(), dsize=size, interpolation=cv2.INTER_LANCZOS4)
