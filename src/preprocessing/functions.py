import numpy as np
import cv2

from typing import Union, Tuple, Any, List

from utils.functions import detect_func_err, get_contours


@detect_func_err
def remove_noise(img: np.ndarray, **kwargs) -> np.ndarray:
    """
    Función para eliminar el ruido granular mediante el uso de un filtro medio
    :param img: array con la imagen
    :param kwargs: parámetros de la función medianblur de cv2
    :return: imagen sin ruido granular
    """

    img_without_noise = img.copy()

    # Se suprime el ruido aditivo
    img_without_noise = cv2.medianBlur(img_without_noise, **kwargs)

    return img_without_noise


@detect_func_err
def crop_borders(img: np.ndarray, left: float = 0.01, right: float = 0.01, top: float = 0.01, bottom: float = 0.01) \
        -> np.ndarray:
    """
    Funión para recortar los bordes de una imagen
    :param img: imagen a recortar
    :param left: proporción del margen izquierdo a recortar
    :param right: proporción del margen derecho a recortar
    :param top: proporción del margen superior a recortar
    :param bottom: proporción del margen inferior a recortar
    """

    # En función de si es una imagen en escala de negros o RGB/BGR se recortan los bordes
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
    Función para normalizar una imagen teniendo en cuenta exclusivamente los valores presentes en una mascara.
    Existen dos tipos de normalizaciones (min_max o truncation).
    :param img: Imagen a normalizar
    :param mask: Masara de las zonas de la imagen a tenen en cuenta para normalizar
    :param type_norm: Tipo de normalización. Los valores aceptados son.
                      - min_max = se realiza un escalado min_max
                      - truncation = se realiza un escalado por truncado utilizando el percentil 99 y percentil 10 de
                                     los píxeles de cada imagen
    :return: imagen normalizada
    """

    # Se transforma la imagen a float para no perder informacion
    img_float = img.copy().astype(float)

    # En caso de no existir mascara se utilizará toda la imagen
    if mask is None:
        mask = np.ones(img.shape)

    # Normalización min, max
    if type_norm == 'min_max':
        min_ = img_float[mask != 0].min()
        max_ = img_float[mask != 0].max()

    # normalización por truncado
    elif type_norm == 'truncation':
        min_ = np.percentile(img_float[mask != 0], 10)
        max_ = np.percentile(img_float[mask != 0], 99)

    else:
        raise ValueError(f'{type_norm} not implemented in normalize_breast')

    # Se asigna el valor 255 como máximo de la normalización
    img_norm = ((np.clip(img_float, min_, max_) - min_) / (max_ - min_)) * 255
    img_norm[mask == 0] = 0

    return np.uint8(img_norm)


@detect_func_err
def binarize(img: np.ndarray, thresh: str = 'otsu', threshval: int = 1) -> np.ndarray:
    """

    Función utilizada retornar una máscara a partir de una imagen de entrada

    :param img: imagen a binarizar
    :param threshval: Valor de threshold. Todos aquellos valores inferiores al threshold se asignarán a 0. (negro)
    :param thresh: Tipo de binarización a realizar. Las opciones validas son otsu o constant.
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
def edit_mask(mask: np.ndarray, operations: List[tuple] = None, kernel_size: tuple = (23, 23),
              kernel_shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    """
    Función para modificar una máscara aplicando filtros morfologicos.
    :param mask: mascara a modificar
    :param kernel_size: tamaño del kernel para aplicar los filtros morfológicos
    :param kernel_shape: forma del filtro
    :param operations: lista de tuplas cuyo primer elemento es la operación morfológica a aplicar y la segunda el número
                       de veces a aplicar la transformación
    :return: mascara modificada
    """

    # Se genera el kernel para realizar la transformación morgológica de la imagen.
    if operations is None:
        operations = [(cv2.MORPH_OPEN, 1)]

    kernel = cv2.getStructuringElement(shape=kernel_shape, ksize=kernel_size)

    # Se realiza una erosión seguida de una dilatación para eliminar el ruido situado en el fondo de la imagen
    # a la vez que posible ruido alrededor del pecho.
    for (transformation, iters) in operations:
        cv2.morphologyEx(mask, transformation, kernel, iterations=iters, dst=mask)

    return mask


@detect_func_err
def get_breast_zone(mask: np.ndarray, convex_contour: bool = False) -> Union[np.ndarray, tuple]:

    """
    Función de obtener la zona del seno de una imagen a partir del area mayor contenido en una mascara.

    :param mask: mascara sobre la cual se realizará la búsqueda de contornos y de las zonas más largas.
    :param convex_contour: boleano para aplicar contornos convexos.
    :return: Máscara que contiene el contorno con mayor area juntamente con el vértice x e y con la anchura y la altura
             del cuadrado que contienen la zona de mayor area de la mascara-

    """

    # Se obtienen los contornos de las zonas de la imagen de color blanco.
    contours = get_contours(img=mask)

    # Se obtiene el contorno más grande a partir del area que contiene
    largest_countour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Se modifican los contornos si se decide obtener contornos convexos.
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
def remove_artifacts(img: np.ndarray, mask: np.ndarray = None, crop: bool = True, **kwargs) \
        -> Tuple[Any, np.ndarray, Any, Any]:
    """
    Función con el pipeline utilizado para remover los artefactos de una imágen mamográfica y obtener exclusivamente
    la zona del seno
    :param img: imágen con los artefactos
    :param mask: másara de cada imagen con el ROI. Se aplicarán las mismas modificaciones que a la imagen original
    :param crop: booleano para recortar la zona del seno de cada imagen
    :param kwargs: parámetrizaciones para cada función del pipeline
    :return: imagen original sin artefactos, mascara del seno, mascara sin artefactos, mascara del roi modificada
    """

    # Se obtiene una mascara que permitirá eliminar los artefactos de la imágenes obteniendo exclusivamente la parte
    # del seno. Para ello, primero se realiza una binarización de los datos para:
    #    1- Poner a negro el fondo de la imágen. Existen partes de la imágen que no son completamente negras pudiendo
    #       producir errores en el entrenamiento.
    #    2- Detectar las zonas pertenecientes al seno y a los artefactos asignandoles el valor de 1 mediante una
    #       binarización.s.

    if mask is None:
        mask = np.zeros(img.shape)

    bin_mask = binarize(img=img, **kwargs.get('bin_kwargs', {}))

    # La binarización de datos puede producir pérdida de información de los contornos del seno, por lo que se
    # incrementará el tamaño de la mascara. Adicionalmente, se pretende capturar de forma completa los artefactos
    # de la imagene para su eliminación suprimiendo posibles zonas de ruido.
    modified_mask = edit_mask(mask=bin_mask, **kwargs.get('mask_kwargs', {}))

    # Una vez identificados con un 1 tanto artefactos como el seno, se debe de identificar cual es la región
    # perteneciente al seno. Esta será la que tenga un área mayor.
    breast_mask, (x, y, w, h) = get_breast_zone(mask=modified_mask, **kwargs.get('contour_kwargs', {}))

    # Se aplica y se devuelve la mascara.
    img[breast_mask == 0] = 0

    if crop:
        return img[y:y + h, x:x + w], modified_mask, breast_mask[y:y + h, x:x + w], mask[y:y + h, x:x + w]
    else:
        return img, modified_mask, breast_mask, mask


@detect_func_err
def flip_breast(img: np.ndarray, orient: str = 'left') -> Union[Tuple[Any, bool], Tuple[np.ndarray, bool]]:
    """
    Función utilizada para realizar el giro de los senos en caso de ser necesario. Esta funcionalidad pretende
    estandarizar las imagenes de forma que los flips se realizen posteriormente en el data augmentation-

    :param img: imagen para realizar el giro
    :param orient: 'left' o 'right'. Orientación que debe presentar el seno a la salida de la función
    :return: imagen girada juntamente con el booleano indicando si se ha realziado un giro o no
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
        return cv2.flip(img.copy(), 1), True
    # En caso contrario el seno estará bien orientado.
    else:
        return img, False


@detect_func_err
def apply_clahe_transform(img: np.ndarray, mask: np.ndarray = None, clip: int = 1) -> np.ndarray:
    """
    función que aplica una ecualización sobre la intensidad de píxeles de la imagen para mejorar el contraste
    de estas.
    :param img: imagen original sobre la cual realizar la ecualización adaptativa
    :param mask: máscara para indicar la zona de la imagen original a tener en cuenta para crear el clahe
    :param clip: parámero clip de la función crateCLAHE de cv2.
    :return: imagen ecualizada
    """

    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    clahe_img = clahe_create.apply(img)

    if mask is not None:
        # Se aplica la ecualización adaptativa del histograma de píxeles.
        clahe_img[mask == 0] = 0

    return clahe_img


@detect_func_err
def pad_image_into_square(img: np.ndarray, ratio: str = '1:2') -> np.ndarray:
    """
    Función para crear bordes negros a una imagen para respetar el ratio indicado y que el rescalado de las imagenes
    no afecte a las formas presentes en un seno
    :param img: imagen original
    :param ratio: string indicando el aspect ratio de salida width:height
    :return: imagen con padding
    """

    # Se obtienen los ratios de altura y anchura
    height_ratio, width_ratio = map(int, ratio.split(':'))

    # Se calculan los tamaños de anchura y altura deseados en función del ratio
    desired_width = img.shape[0] * height_ratio // width_ratio
    desired_height = img.shape[1] * width_ratio // height_ratio

    # Se crea una imagen en negro con los valores de altura y anchura deseados
    padded_img = np.zeros(shape=(max(img.shape[0], desired_height), max(desired_width, img.shape[1])), dtype=np.uint8)

    # Se asignan los píxeles de la imagen original
    if len(img.shape) == 2:
        nrows, ncols = img.shape
        padded_img[:nrows, :ncols] = img

    else:
        nrows, ncols, d = img.shape
        padded_img[:nrows, :ncols, :] = img

    return padded_img


@detect_func_err
def resize_img(img: np.ndarray, height: int, width: int, interpolation=cv2.INTER_LANCZOS4) -> np.ndarray:
    """
    Función para reescalar una imagen.
    :param img: imagen original
    :param height: altura de salida
    :param width: anchura de salida
    :param interpolation: método de interpolación para realizar el reescalado
    :return: imagen reescalada
    """

    return cv2.resize(src=img.copy(), dsize=(width, height), interpolation=interpolation)


@detect_func_err
def correct_axis(shape: tuple, x_max: int, x_min: int, y_max: int, y_min: int) -> Tuple[int, int, int, int]:
    """
    Función para corregir y validar las posiciones de píxel de el cuadrado que contiene un roi en un a imagen
    :param shape: dimensiones de la imagen
    :param x_max: valor x máximo de la zona de interes
    :param x_min: valor x mínimo de la zona de interes
    :param y_max: valor y máximo de la zona de itneres
    :param y_min: valor y minimo de la zona de interes
    :return valores x e y modificados
    """

    # Si el valor de X es mayor que la anchura de la imagen se desplazan los ejes de forma proporcional al tamaño del
    # roi hacia la izquierda
    if x_max > shape[1]:
        x_min -= x_max - shape[1]
        x_max = shape[1]

    # Si el valor de X menor que 0 se desplazan los ejes de forma proporcional al tamaño del roi hacia la derecha
    if x_min < 0:
        x_max += abs(x_min)
        x_min = 0

    # Si el valor de Y es mayor que la altura de la imagen se desplazan los ejes de forma proporcional al tamaño del
    # roi hacia abajo
    if y_max > shape[0]:
        y_min -= y_max - shape[0]
        y_max = shape[0]

    # Si el valor de Y es menor que 0 se desplazan los ejes de forma proporcional al tamaño del roi hacia arriba
    if y_min < 0:
        y_max += abs(y_min)
        y_min = 0

    return int(x_max), int(x_min), int(y_max), int(y_min)
