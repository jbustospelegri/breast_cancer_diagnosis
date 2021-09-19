from keras import backend, models
from pathlib import Path
from contextlib import redirect_stdout

from tensorflow.python.keras.preprocessing.image import DataFrameIterator
from tensorflow import one_hot, argmax

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import functools
import logging
import os


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


def f1_score(y_true, y_pred):
    """
    Función utilizada para calcular la métrica F1 a partir de las predicciones y los labels verdaderos
    :param y_true: array con labels verdaderos
    :param y_pred: array con las predicciones
    :return: métrica f1
    """

    def get_recall(true_label, pred_label):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = backend.sum(backend.round(backend.clip(true_label * pred_label, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(true_label, 0, 1)))
        return true_positives / (possible_positives + backend.epsilon())

    def get_precision(true_label, pred_label):
        """
        Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = backend.sum(backend.round(backend.clip(true_label * pred_label, 0, 1)))
        predicted_positives = backend.sum(backend.round(backend.clip(pred_label, 0, 1)))
        return true_positives / (predicted_positives + backend.epsilon())

    precision = get_precision(y_true, y_pred)

    recall = get_recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + backend.epsilon()))


@create_dir
def get_predictions(keras_model: models, data: DataFrameIterator, class_labels: dict, **kwargs) -> pd.DataFrame:
    """
    Función utilizada para generar las predicciones de un modelo. El dataframe generado contendrá el path de la imagen,
    la clase verdadera (en caso de existir) y la clase predicha.

    :param keras_model: modelo sobre el que se aplicará el método .predict para generar las predicciones
    :param data: dataset sobre el cual aplicar las predicciones
    :param class_labels: diccionario con las clases verdaderas del set de datos. Este diccionario debe contener como
                         key la posición correspondiente a cada prediccion dentro del vector de probabilidades devuelto
                         por el método predict() de keras y, como value, el nombre de la clase
    :param kwargs: columnas adicionales para añadir al dataframe devuelto. El key de kwargs será el nombre de la columna
                   y los values serán el valor asignado a cada columna.
    :return: dataframe con el path de la imagen, la clase verdadera (en caso de existir), la clase predicha y columnas
             definidas por kwargs.
    """

    # Se recuepra el número de clases
    num_clases = len(class_labels.keys())

    # Se genera un orden aleatorio del datagenerator
    data.on_epoch_end()

    # se recupera el path de los archivos del dataset generator
    fnames = [data.filenames[i] for i in data.index_array] if data.index_array is not None else data.filenames

    true_labels = []
    # En caso de que exista la clase verdadera, se recupera y se añade al dataset
    if hasattr(data, 'classes'):
        for idx in range(0, (data.samples // data.batch_size) + 1):
            true_labels += [class_labels[label] for label in data[idx][1].argmax(axis=-1).tolist()]

    # Se predicen los datos. Debido a que la salida del modelo para cada muestra es un vector de probabilidades,
    # dada por la función de activacón softmax, se obtiene la clase predicha mediante el valor máximo (clase más
    # probable).
    predictions = [class_labels[pred] for pred in one_hot(argmax(keras_model.predict(data), axis=1), num_clases).numpy()
        .argmax(axis=-1).tolist()]

    # Se crea el dataset final
    dataset = pd.DataFrame({'filename': fnames, 'predictions': predictions, 'true_labels': true_labels}) \
        if true_labels else pd.DataFrame({'filename': fnames, 'predictions': predictions})

    # Se añaden columnas adicionales al dataset
    for col, value in kwargs.get('add_columns', {}).items():
        dataset.loc[:, col] = [value] * len(dataset)

    return dataset


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14, header_color='#40466e', row_colors=None,
                     edge_color='w', bbox=None, header_columns=0, ax=None, **kwargs):
    """
    Función utilizada para renderizar un dataframe en una tabla de matplotlib. Función recuperada de
    https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure
    """
    if bbox is None:
        bbox = [0, 0, 1, 1]

    if row_colors is None:
        row_colors = ['#f1f1f2', 'w']

    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax.get_figure(), ax
