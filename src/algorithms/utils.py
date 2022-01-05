import pandas as pd
import numpy as np

import utils.config as conf

from multiprocessing import Queue
from typing import Union, io
from sklearn.metrics import roc_curve
from sklearn.utils import resample

from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import Iterator
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
    :param kwargs: columnas adicionales para añadir al dataframe devuelto. El key de kwargs será el nombre de la columna
                   y los values serán el valor asignado a cada columna.
    :return: dataframe con el path de la imagen, la clase verdadera (en caso de existir), la clase predicha y columnas
             definidas por kwargs.
    """

    # Se genera un orden aleatorio del datagenerator producido por on_epoch-end
    data.on_epoch_end()

    # se recupera el path de los archivos del dataset generator
    fnames = [data.filenames[i] for i in data.indexes]

    true_labels = []
    # En caso de que exista la clase verdadera, se recupera y se añade al dataset
    if hasattr(data, 'classes'):
        true_labels = [data.classes[i] for i in data.indexes]

    # Se define el tamaño de batch a 1 para predecir todas las observaciones
    data.set_batch_size(1)
    # Se predicen los datos y se recupera la probabilidad de la clase 1 (maligno)
    predictions = keras_model.predict(data)[:, 1]

    # Se crea el dataset final con los ficheros, prediccion y clase real en caso de existir
    if true_labels:
        dataset = pd.DataFrame({'PROCESSED_IMG': fnames, 'PREDICTION': predictions, 'IMG_LABEL': true_labels})
    else:
        dataset = pd.DataFrame({'PROCESSED_IMG': fnames, 'PREDICTION': predictions})

    # Se añaden columnas adicionales al dataset
    for col, value in kwargs.get('add_columns', {}).items():
        dataset.loc[:, col] = [value] * len(dataset)

    return dataset


def training_pipe(m: Model, db: BreastCancerDataset, q: Queue, c: conf.MODEL_FILES, task_type: str, fc: str = 'simple',
                  weight_init: Union[str, io] = None, frozen_layers: Union[str, int] = None) -> None:
    """
    Función utilizada para generar el pipeline de entrenamiento de cada modelo. Dado que tensorflow no libera cache
    al entrenar un modelo, se debe de llamar esta función a través de un thread o proceso paralelo.

    :param m: Red neuronal (objeto de la clase General Model) que contendrá cada algoritmo de dl
    :param db: Objeto BreastCancerDataset con las observaciones de los conjuntos de entrenamiento y validacion
    :param q: Queue para transmitir comunicar el resultado al thread principal.
    :param c: objeto Model files que contiene información sobre ls rutas de guardado de cada modelo
    :param task_type: admite los valores 'classification' o 'segmentation' para escoger el tipo de tarea a realizar
    :param weight_init: nombre o path de los pesos con los que inicializar el entrenamiento de un modelo.
    :param frozen_layers: número de capas a entrenar en cada modelo

    """
    # Se inicializa cada modelo:
    cnn = m(n=len(db.class_dict), weights=None if weight_init == 'random' else weight_init, top_fc=fc)

    # Se registran las métricas que se desean almacenar y se obtienen los conjuntos de train y validacion aplicando
    # el escalado propio de cada red y la función de preprocesado propia. El tamaño de batch se define a partir
    # de la hoja de configuraciones.
    if task_type == 'classification':
        cnn.register_metric(*list(conf.CLASSIFICATION_METRICS.values()))

        train, val = db.get_classification_dataset_generator(
            batch_size=cnn.BS_DICT[frozen_layers], callback=cnn.get_preprocessing_func(), size=cnn.shape[:2]
        )

        # train, val = db.get_classification_dataset_generator(
        #     batch_size=conf.BATCH_SIZE, callback=cnn.get_preprocessing_func(), size=cnn.shape[:2]
        # )

    elif task_type == 'segmentation':
        cnn.register_metric(*list(conf.SEGMENTATION_METRICS.values()))

        train, val = db.get_segmentation_dataset_generator(
            batch_size=conf.SEGMENTATION_BATCH_SIZE, callback=cnn.get_preprocessing_func(), size=conf.IMG_SHAPE
        )

    else:
        raise ValueError(f'task_type not incorporated')

    name = cnn.__name__
    if frozen_layers != 'ALL':
        filename = f'{name}_FineTunning'
    else:
        filename = f'{name}_Scratch'

    # Se registran los callbacks del modelo (Earlystopping y CSV logger del train de cada modelo)
    csv_filepath = get_path(c.model_log_dir, weight_init, frozen_layers, f'{filename}.csv')
    cnn.register_callback(
        early_stopping=EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True),
        log=CSVLogger(filename=csv_filepath, separator=';', append=True)
    )
    # En función de las capas congeladas y la inicializacion de los pesos se realiza el entrenamiento de cada
    # arquitectura siguiendo la siguiente lógica:
    #     - Si se entrenan todas las capas, se llama al metodo train_from_scratch.
    #     - En caso contrario, primero se entrenan únicamente las capas FC de cada modelo y posteriormente se entrena
    #       el modelo con las capas especificadas por el usuario.
    # por defecto el optimizador es Adam haciendo uso del learning rate identificado en la hoja de config.
    if frozen_layers == 'ALL':
        print(f'{"=" * 75}\nEntrenando {name} desde 0 con inicialización de pesos {weight_init}\n{"=" * 75}')
        # Entrenamiento desde 0
        t, e = cnn.train_from_scratch(train, val, conf.EPOCHS, Adam(conf.LEARNING_RATE))
        # Se registra el etrenamiento del modelo (tiempo, capas, inicialización) en un csv
        bulk_data(file=c.model_summary_train_csv, mode='a', cnn=name, process='Scratch', FT=frozen_layers,
                  weights=weight_init, time=t, epochs=e, trainable_layers=cnn.get_trainable_layers())
        print(f'{"=" * 75}\nEntrenamiento finalizado.\n{"=" * 75}')

    else:
        print(f'{"=" * 75}\nEntrenando {name} mediante transfer learning con inicialización de pesos de '
              f'{weight_init}. Número de capas a entrenar {frozen_layers}\n{"=" * 75}')

        print(f'{"-" * 75}\n\tEmpieza proceso de extract-features (warm up)\n{"-" * 75}')
        # Se realiza el extract features para entrenar los pesos de la capa FC
        t, e = cnn.extract_features(train, val, conf.WARM_UP_EPOCHS, Adam(conf.LEARNING_RATE))
        # Se registra el etrenamiento del modelo (tiempo, capas, inicialización) en un csv
        bulk_data(file=c.model_summary_train_csv, mode='a', cnn=name, process='ExtractFeatures', FT=frozen_layers,
                  weights=weight_init, time=t, epochs=e, trainable_layers=cnn.get_trainable_layers())
        print(f'{"-" * 75}\n\tEntrenamiento finalizado.\n{"-" * 75}')

        print(f'{"-" * 75}\n\tEmpieza proceso de fine-tunning\n{"-" * 75}')
        # Se entrena el modelo congelando las capas especificadas por el usuario
        t, e = cnn.fine_tunning(train, val, conf.EPOCHS, Adam(cnn.get_learning_rate()), frozen_layers)
        # Se registra el etrenamiento del modelo (tiempo, capas, inicialización) en un csv
        bulk_data(file=c.model_summary_train_csv, mode='a', cnn=name, process='FineTunning', FT=frozen_layers,
                  weights=weight_init, time=t, epochs=e, trainable_layers=cnn.get_trainable_layers())
        print(f'{"-" * 75}\n\tEntrenamiento finalizado.\n{"-" * 75}')

        print(f'{"=" * 75}\nProceso de transfer learning finalizado\n{"=" * 75}')

    # Se almacenan los pesos del modelo
    print(f'{"=" * 75}\nAlmacenando  modelo.\n{"=" * 75}')
    cnn.save_weights(dirname=get_path(c.model_store_cnn_dir, weight_init, frozen_layers), model_name=f"{name}.h5")
    print(f'{"=" * 75}\nModelo almacenado correctamente.\n{"=" * 75}')

    # En el caso de realizar una clasificación, se obtienen las predicciones de cada instancia.
    if task_type == 'classification':

        print(f'{"=" * 75}\nObteniendo predicciones del modelo {name}.\n{"=" * 75}')

        # Se generan las predicciones de entrenamiento y validación en formato de dataframe y se devuelven al proceso
        # ppal.
        q.put(
            pd.concat(
                objs=[
                    get_predictions(keras_model=cnn, data=train, add_columns={'TRAIN_VAL': 'train'}),
                    get_predictions(keras_model=cnn, data=val, add_columns={'TRAIN_VAL': 'val'})
                ],
                ignore_index=True
        ))
        print(f'{"=" * 75}\nPredicciones finalizadas.\n{"=" * 75}')
    else:
        q.put(True)


def optimize_threshold(true_labels: np.array, pred_labels: np.array) -> float:
    """
    Función utilizada para obtimizar el threshold a partir del estadístico J de Youden (maximizar tasa de tpr y tnr).
    :param true_labels: Vector con las clases verdaderas
    :param pred_labels: Vector con las clases predichas
    :return: threshold que maximiza la diferencia entre tpr y fpr = tpr + tnr - 1
    """

    try:
        fpr, tpr, thresholds = roc_curve(true_labels, pred_labels)

        return thresholds[argmax(tpr - fpr)]
    except Exception:
        return None


def apply_bootstrap(data: pd.DataFrame, true_col: str, pred_col: str, metric: callable, iters: int = 1000,
                    ci: float = 0.95, prop: float = 0.75, **kwargs) -> tuple:
    """
    Función utilizada para aplicar un bootstrap y obtener una métrica de actucación de un modelo.
    :param data: pandas dataframe con los datos verdaderos y predichos de cada instancia
    :param true_col: nombre de la columna del dataframe con los datos verdaderos
    :param pred_col: nombre de la columna del dataframe con los datos predidchos
    :param metric: callback sobre el cual aplicar la métrica
    :param iters: número de iteraciones para realizar el algoritmo de bootstrap
    :param ci: interalo de confianza para obtener la metrica
    :param prop: proporción del set de datos a tener en cuenta para aplicar el bootstrap
    :param kwargs: parámetros del callable metric
    :return: media del intervalo con sus respectivos limites (mínimo y máximo).
    """
    assert true_col in data.columns, f'{true_col} not in dataframe'
    assert pred_col in data.columns, f'{pred_col} not in dataframe'

    results = []
    for i in range(iters):

        sample = resample(data, n_samples=int(len(data) * prop))

        results.append(metric(sample[true_col].values.tolist(), sample[pred_col].values.tolist(), **kwargs))

    try:
        lower = max(0.0, np.percentile(results, ((1.0 - ci) / 2.0) * 100))
        upper = min(1.0, np.percentile(results, (ci + ((1.0 - ci) / 2.0)) * 100))
        return np.mean(results), lower, upper
    except TypeError:
        return None, None, None
