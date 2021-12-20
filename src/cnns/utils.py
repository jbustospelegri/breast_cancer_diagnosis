import os

import pandas as pd

import utils.config as conf

from multiprocessing import Queue
from typing import Union, io, Callable

from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow import one_hot
from tensorflow.keras import Model
from tensorflow.keras.backend import argmax
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from utils.functions import get_path, bulk_data
from breast_cancer_dataset.database_generator import BreastCancerDataset


def get_predictions(keras_model: models, data: Iterator, **kwargs) -> pd.DataFrame:
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

    if data.class_indices:
        class_labels = {v: i for i, v in data.class_indices.items()}
    else:
        class_labels = {0: 'BENIGN', 1: 'MALIGNANT'}

    # Se genera un orden aleatorio del datagenerator
    data.on_epoch_end()

    # se recupera el path de los archivos del dataset generator
    fnames = [data.filenames[i] for i in data.indexes]

    true_labels = []
    # En caso de que exista la clase verdadera, se recupera y se añade al dataset
    if hasattr(data, 'classes'):
        true_labels = [data.classes[i] for i in data.indexes]

    # Se predicen los datos. Debido a que la salida del modelo para cada muestra es un vector de probabilidades,
    # dada por la función de activacón softmax, se obtiene la clase predicha mediante el valor máximo (clase más
    # probable).
    data.set_batch_size(1)
    probabilities = keras_model.predict(data)
    predictions = [
        class_labels[pred] for pred in one_hot(argmax(probabilities, axis=1), len(class_labels)).numpy().\
            argmax(axis=-1).tolist()]

    # Se crea el dataset final
    if true_labels:
        dataset = pd.DataFrame({
            'PROCESSED_IMG': fnames,
            'PREDICTION': predictions,
            'PROBABILTY': probabilities[:, 1],
            'IMG_LABEL': true_labels
        })
    else:
        dataset = pd.DataFrame({'PROCESSED_IMG': fnames, 'PREDICTION': predictions, 'PROBABILTY': probabilities[:, 1]})

    # Se añaden columnas adicionales al dataset
    for col, value in kwargs.get('add_columns', {}).items():
        dataset.loc[:, col] = [value] * len(dataset)

    return dataset


def training_pipe(m: Model, db: BreastCancerDataset, q: Queue, c: conf.MODEL_FILES, task_type: str,
                  weight_init: Union[str, io] = None, frozen_layers: Union[str, int] = None) -> None:
    """
    Función utilizada para generar el pipeline de entrenamiento de cada modelo.

    :param q:
    :param c:
    :param df: objeto dataset con el cual obtener los dataframe iterators de entrenamiento y validacion.
    :param model_name: nombre del modelo para seleccionar qué modelo entrenar
    :param bs: tamaño del batch
    :param n_epochs: número de épocas
    :param name: nombre del test para almacenar el modelo y los logs generados
    :param opt: optimizador de gradiente descendiente a utilizar durante el proceso de back propagation
    :param queue: queue para devolver los resultados al proceso principal

    """
    # Se inicializa cada modelo:
    cnn = m(n=len(db.class_dict), weights=None if weight_init == 'random' else weight_init)

    # Se registran las métricas que se desean almacenar:
    if task_type == 'classification':
        cnn.register_metric(*list(conf.CLASSIFICATION_METRICS.values()))

        train, val = db.get_classification_dataset_generator(
            batch_size=conf.BATCH_SIZE, callback=cnn.get_preprocessing_func(), size=cnn.shape[:2]
        )

    elif task_type == 'segmentation':
        cnn.register_metric(*list(conf.SEGMENTATION_METRICS.values()))

        train, val = db.get_segmentation_dataset_generator(
            batch_size=conf.SEGMENTATION_BATCH_SIZE, callback=cnn.get_preprocessing_func(), size=conf.IMG_SHAPE
        )

    else:
        raise ValueError(f'task_type not incorporated')

    # Queue que servirá para recuparar las predicciones de cada modelo.
    name = cnn.__name__
    if frozen_layers != 'ALL':
        filename = f'{name}_FineTunning'
    else:
        filename = f'{name}_Scratch'

    # Se registran los callbacks del modelo:
    csv_filepath = get_path(c.model_log_dir, weight_init, frozen_layers, f'{filename}.csv')
    cnn.register_callback(
        early_stopping=EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True),
        log=CSVLogger(filename=csv_filepath, separator=';', append=True)
    )

    if frozen_layers == 'ALL':
        print(f'{"=" * 75}\nEntrenando {name} desde 0 con inicialización de pesos {weight_init}\n{"=" * 75}')

        t, e = cnn.train_from_scratch(train, val, conf.EPOCHS, Adam(conf.LEARNING_RATE))

        bulk_data(file=c.model_summary_train_csv, mode='a', cnn=name, process='Scratch', FT=frozen_layers,
                  weights=weight_init, time=t, epochs=e, trainable_layers=cnn.get_trainable_layers())
        print(f'{"=" * 75}\nEntrenamiento finalizado.\n{"=" * 75}')

    else:
        print(f'{"=" * 75}\nEntrenando {name} mediante transfer learning con inicialización de pesos de '
              f'{weight_init}. Número de capas a entrenar {frozen_layers}\n{"=" * 75}')

        print(f'{"-" * 75}\n\tEmpieza proceso de extract-features (warm up)\n{"-" * 75}')

        t, e = cnn.extract_features(train, val, conf.WARM_UP_EPOCHS, Adam(conf.WARM_UP_LEARNING_RATE))

        bulk_data(file=c.model_summary_train_csv, mode='a', cnn=name, process='ExtractFeatures', FT=frozen_layers,
                  weights=weight_init, time=t, epochs=e, trainable_layers=cnn.get_trainable_layers())
        print(f'{"-" * 75}\n\tEntrenamiento finalizado.\n{"-" * 75}')

        print(f'{"-" * 75}\n\tEmpieza proceso de fine-tunning\n{"-" * 75}')

        t, e = cnn.fine_tunning(train, val, conf.EPOCHS, Adam(conf.LEARNING_RATE), frozen_layers)

        bulk_data(file=c.model_summary_train_csv, mode='a', cnn=name, process='FineTunning', FT=frozen_layers,
                  weights=weight_init, time=t, epochs=e, trainable_layers=cnn.get_trainable_layers())
        print(f'{"-" * 75}\n\tEntrenamiento finalizado.\n{"-" * 75}')

        print(f'{"=" * 75}\nProceso de transfer learning finalizado\n{"=" * 75}')

    print(f'{"=" * 75}\nAlmacenando  modelo.\n{"=" * 75}')
    cnn.save_weights(dirname=get_path(c.model_store_cnn_dir, weight_init, frozen_layers), model_name=f"{name}.h5")
    print(f'{"=" * 75}\nModelo almacenado correctamente.\n{"=" * 75}')

    if task_type == 'classification':

        print(f'{"=" * 75}\nObteniendo predicciones del modelo {name}.\n{"=" * 75}')

        # Se generan las predicciones de entrenamiento y validación en formato de dataframe y se devuelven al proceso
        # ppal.
        q.put(
            pd.concat(
                objs=[
                    get_predictions(keras_model=cnn, data=train, add_columns={'TRAIN_VAL': 'train'}),
                    get_predictions(keras_model=cnn, data=val, add_columns={'TRAIN_VAL': 'val'})],
                ignore_index=True
        ))
        print(f'{"=" * 75}\nPredicciones finalizadas.\n{"=" * 75}')
    else:
        q.put(True)
