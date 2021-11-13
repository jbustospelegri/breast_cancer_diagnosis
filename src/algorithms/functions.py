import pandas as pd

import utils.config as conf

from multiprocessing import Queue
from typing import Union, io, Callable

from keras import models
from keras_preprocessing.image import DataFrameIterator
from tensorflow import one_hot
from tensorflow.python.keras.backend import argmax
from tensorflow.python.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, LambdaCallback
from tensorflow.python.keras.optimizer_v1 import Adam, SGD

from algorithms.cnns import GeneralModel
from algorithms.metrics import f1_score
from breast_cancer_dataset.datasets import BreastCancerDataset


from utils.functions import get_path


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


def training_pipe(model: Callable[..., GeneralModel], df: BreastCancerDataset, q: Queue, c: conf.MODEL_CONSTANTS,
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
    cnn = model(n_clases=len(df.class_dict), weights=None if weight_init == 'random' else weight_init)

    # Se registran las métricas que se desean almacenar:
    cnn.register_metric('AUC', 'Precision', 'Recall', f1_score)

    # Se registran los callbacks del modelo:
    cnn.register_callback(
        early_stopping=EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True),
        log_hyperparams=LambdaCallback(on_train_end=lambda logs: print(logs)),
        lr_reduce_on_plateau=ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=5)
    )

    name = cnn.__name__

    # Se recuperan los generadores de entrenamiento y validación en función del tamaño de entrada definido para cada
    # red y su función de preprocesado.
    train, val = df.get_dataset_generator(
        batch_size=conf.BATCH_SIZE, size=cnn.input_shape, preprocessing_function=cnn.preprocess_func,
        directory=c.MODEL_CONSTANTS.model_db_data_augm_dir
    )

    if weight_init == 'random':
        print(f'{"=" * 75}\nEntrenando {name} desde 0 con inicialización de pesos {weight_init}\n{"=" * 75}')
        cnn.register_callback(log=CSVLogger(
            filename=get_path(c.model_log_dir, f'{name}_{weight_init}_{frozen_layers}_scratch.csv'), separator=';'))
        cnn.train_from_scratch(train, val, conf.EPOCHS, conf.BATCH_SIZE, Adam(lr=conf.WARM_UP_LEARNING_RATE))
        print(f'{"=" * 75}\nEntrenamiento finalizado.\n{"=" * 75}')

    elif weight_init == 'imagenet':
        print(f'{"=" * 75}\nEntrenando {name} mediante transfer learning con inicialización de pesos de '
              f'{weight_init})\n{"=" * 75}')

        print(f'{"-" * 75}\n\tEmpieza proceso de extract-features (warm up)\n{"-" * 75}')
        cnn.register_callback(log=CSVLogger(
            filename=get_path(c.model_log_dir, f'{name}_{weight_init}_{frozen_layers}_ExtractFeatures.csv'),
            separator=';'))
        cnn.extract_features(train, val, conf.WARM_UP_EPOCHS, conf.BATCH_SIZE, Adam(lr=conf.WARM_UP_LEARNING_RATE))
        print(f'{"-" * 75}\n\tEntrenamiento finalizado.\n{"-" * 75}')

        print(f'{"-" * 75}\n\tEmpieza proceso de fine-tunning\n{"-" * 75}')
        cnn.register_callback(log=CSVLogger(
            filename=get_path(c.model_log_dir, f'{name}_{weight_init}_{frozen_layers}_FineTunning.csv'), separator=';'))
        cnn.fine_tunning(train, val, conf.EPOCHS, conf.BATCH_SIZE, SGD(lr=conf.LEARNING_RATE), frozen_layers)
        print(f'{"-" * 75}\n\tEntrenamiento finalizado.\n{"-" * 75}')

        print(f'{"=" * 75}\nProceso de transfer learning finalizado\n{"=" * 75}')

    print(f'{"=" * 75}\nAlmacenando  modelo.\n{"=" * 75}' )
    cnn.save_model(dirname=conf.MODEL_CONSTANTS.model_store_dir, model_name=f"{name}.h5")
    print(f'{"=" * 75}\nModelo almacenado correctamente.\n{"=" * 75}')

    print(f'{"=" * 75}\nObteniendo predicciones del modelo {name}.\n{"=" * 75}')
    class_dict_labels = {v: k for k, v in train.class_indices.items()}

    # Se generan las predicciones de entrenamiento y validación en formato de dataframe y se devuelven al proceso ppal.
    q.put(pd.concat(
        objs=[
            get_predictions(keras_model=cnn, data=train, class_labels=class_dict_labels,
                            add_columns=dict(mode='train')),
            get_predictions(keras_model=cnn, data=val, class_labels=class_dict_labels,
                            add_columns=dict(mode='val'))],
        ignore_index=True
    ))
    print(f'\n{"=" * 75}Predicciones finalizadas.\n{"=" * 75}')