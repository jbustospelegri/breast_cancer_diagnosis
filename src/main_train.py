from scripts.dataset import Dataset
from scripts.models import VGG16Model, Resnet50Model, InceptionV3Model, DenseNetModel, ModelEnsamble
from scripts.visualizacion_resultados import DataVisualizer

from utils.functions import get_predictions
from utils.config import (
    MODEL_SAVE_DATA_PATH, TRAIN_DATA_PATH, EDA_DATA_PATH, PREDICTIONS_PATH, LOGGING_DATA_PATH, VISUALIZATIONS_PATH,
    EXAMPLE_IMAGE_DATA_PATH
)

from keras.optimizers.adam_v2 import Adam
from multiprocessing import Process, Queue

import tensorflow
import pandas as pd
import os
import keras


def training_pipeline(df: Dataset, model_name: str, bs: int, n_epochs: int, name: str, opt: keras.optimizers,
                      queue: Queue):
    """
    Función utilizada para generar el pipeline de entrenamiento de cada modelo.

    :param df: objeto dataset con el cual obtener los dataframe iterators de entrenamiento y validacion.
    :param model_name: nombre del modelo para seleccionar qué modelo entrenar
    :param bs: tamaño del batch
    :param n_epochs: número de épocas
    :param name: nombre del test para almacenar el modelo y los logs generados
    :param opt: optimizador de gradiente descendiente a utilizar durante el proceso de back propagation
    :param queue: queue para devolver los resultados al proceso principal

    """

    # En función del parámetro name se escoge el modelo a entrenar
    if model_name == 'VGG16':
        cnn = VGG16Model(n_clases=df.n_clases, name='VGG16', test_name=name, get_model_structure=True)
    elif model_name == 'ResNet50':
        cnn = Resnet50Model(n_clases=df.n_clases, name='ResNet50', test_name=name, get_model_structure=True)
    elif model_name == 'InceptionV3':
        cnn = InceptionV3Model(n_clases=df.n_clases, name='InceptionV3', test_name=name, get_model_structure=True)
    elif model_name == 'DenseNet121':
        cnn = DenseNetModel(n_clases=df.n_clases, name='DenseNet121', test_name=name, get_model_structure=True)

    # Se recuperan los generadores de entrenamiento y validación en función del tamaño de entrada definido para cada
    # red y su función de preprocesado.
    train, val = df.get_dataset_generator(batch_size=bs, size=cnn.input_shape,
                                          preprocessing_function=cnn.preprocess_func)

    print('-' * 50 + f'\nEmpieza proceso de transfer learning para {cnn.name}')
    cnn.transfer_learning_pipe(train, val, n_epochs, bs, opt(learning_rate=1e-3))
    print('-' * 50 + f'\nFinalizado proceso transfer learning.\n' + '-' * 50)

    print('-' * 50 + f'\nEmpieza proceso de fine tunning {cnn.name}')
    cnn.train_from_scratch_pipe(train, val, n_epochs, bs, opt(learning_rate=1e-5))
    print('-' * 50 + f'\nFinalizado proceso de fine tunning.\n' + '-' * 50)

    print('-' * 50 + f'\nAlmacenando  modelo.\n' + '-' * 50)
    cnn.save_model(dirname=MODEL_SAVE_DATA_PATH, model_name=f"{cnn.name}_{name}.h5")
    print('-' * 50 + f'\nModelo almacenado correctamente.\n' + '-' * 50)

    print('-' * 50 + f'\nObteniendo predicciones del modelo {cnn.name}.\n' + '-' * 50)
    class_dict_labels = {v: k for k, v in train.class_indices.items()}

    # Se generan las predicciones de entrenamiento y validación en formato de dataframe y se devuelven al proceso ppal.
    queue.put(pd.concat(
        objs=[
            get_predictions(keras_model=cnn, data=train, class_labels=class_dict_labels,
                            add_columns=dict(mode='train')),
            get_predictions(keras_model=cnn, data=val, class_labels=class_dict_labels,
                            add_columns=dict(mode='val'))],
        ignore_index=True
    ))
    print('-' * 50 + f'\nPredicciones finalizadas.\n' + '-' * 50)


if __name__ == '__main__':

    # Se chequea la existencia de GPU's activas
    print("TF version   : ", tensorflow.__version__)
    # we'll need GPU!
    print("GPU available: ", tensorflow.config.list_physical_devices('GPU'))

    tensorflow.get_logger().setLevel('ERROR')
    tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

    # Parámetros de entrada que serán sustituidos por las variables del usuario
    test_name = 'prueba_incepcion_224'
    batch_size = 16
    epochs = 40
    optimizador = Adam
    predictions_file_csv = f'predictions_train_val_{test_name}.csv'

    # Constante en la que se almacena el modelo
    model_filepath = os.path.join(MODEL_SAVE_DATA_PATH, f'GradientBoostingClassifier_{test_name}.sav')

    # Se crea un objeto de la clase Dataset en el cual se obtendrá el set de datos de entrenamiento.
    data = Dataset(TRAIN_DATA_PATH, image_format='jpg')

    # Se obtiene el dataframe
    data.get_dataset()

    # Se representa la distribución de clases presentes en el dataframe
    data.get_class_distribution_from_dir(dirname=EDA_DATA_PATH, filename=f'dataset_distribution.jpg')

    # Se obtienen 5 ejemplos de cada clase
    data.get_class_examples(n_example=5, dirname=EDA_DATA_PATH, filename=f'data_examples.jpg')

    # Se obtiene el rango dinámico y los tamaños de cada clase.
    data.get_info_from_image(dirname=EDA_DATA_PATH, log_file=f'dataset_info.jpg')

    # Se obtiene un ejemplo de las transformaciones de data augmentation
    data.get_data_augmentation_examples(dirname=EDA_DATA_PATH, out_file='data_augmentation_example.jpg',
                                        example_imag=os.path.join(EXAMPLE_IMAGE_DATA_PATH, 'image_0.jpg'))

    # Se realiza la división del set de datos en train y validación
    data.split_dataset_on_train_val(train_prop=0.8, stratify=True)

    # Diccionario en el que se almacenarán las predicciones de cada modelo. Estas serán utilizadas para aplicar el
    # algorítmo de gradient boosting.
    model_predictions = {}

    # Debido a que tensorflow no libera el espacio de GPU hasta finalizar un proceso, cada modelo se entrenará en
    # un subproceso daemonico para evitar la sobrecarga de memoria.
    for model in ['VGG16', 'ResNet50', 'InceptionV3', 'DenseNet121']:

        # Queue que servirá para recuparar las predicciones de cada modelo.
        q = Queue()
        # Se rea el proceso
        p = Process(target=training_pipeline, args=(data, model, batch_size, epochs, test_name, optimizador, q))
        # Se lanza el proceso
        p.start()
        # Se recuperan los resultados. El metodo get es bloqueante hasta que se obtiene un resultado.
        predictions = q.get()
        # Se almacenan los resultados de cada modelo.
        model_predictions[model] = predictions

    print('-' * 50 + f'\nGeneradando combinación secuencial de clasificadores.\n' + '-' * 50)
    ensambler = ModelEnsamble()
    df_ensambler = data.df.rename(columns={'item_path': 'filenames', 'class': 'true_label', 'dataset': 'mode'})
    ensambler.train_model(
        train=df_ensambler[df_ensambler['mode'] == 'train'], val=df_ensambler[df_ensambler['mode'] == 'val'],
        dirname=PREDICTIONS_PATH, filename=predictions_file_csv, save_model=model_filepath, **model_predictions
    )
    print('-' * 50 + f'\nProceso de entrenamiento finalizado\n' + '-' * 50)

    print('-' * 50 + f'\nGeneradando visualización de resultados.\n' + '-' * 50)
    data_viz = DataVisualizer()

    print('-' * 50 + f'\nRepresentando accuracy, loss y f1_score para finetunning y transferLearning.\n' + '-' * 50)
    data_viz.get_model_logs_metrics(logs_dir=LOGGING_DATA_PATH, test_name=test_name, dirname=VISUALIZATIONS_PATH,
                                    out_filename=f'model_history_{test_name}',
                                    train_phases=['TransferLearning', 'FineTunning'])

    print('-' * 50 + f'\nRepresentando accuracy, loss y f1_score globales.\n' + '-' * 50)
    data_viz.get_model_logs_metrics(LOGGING_DATA_PATH, test_name, VISUALIZATIONS_PATH, f'model_history_{test_name}')

    print('-' * 50 + f'\nGenerando matrices de confusión de los modelos.\n' + '-' * 50)
    data_viz.plot_confusion_matrix(dirname=VISUALIZATIONS_PATH, out_file=f'confusion_matrix_{test_name}',
                                   input_file=os.path.join(PREDICTIONS_PATH, predictions_file_csv))

    print('-' * 50 + f'\nRecopilando métricas de Recall, Accuracy, F1_score y precisión.\n' + '-' * 50)
    data_viz.get_metrics_matrix(dirname=VISUALIZATIONS_PATH, out_file=f'model_metrics_{test_name}', class_metrics=False,
                                input_file=os.path.join(PREDICTIONS_PATH, predictions_file_csv))
    print('-' * 50 + f'\nVisualizaciones finalizadas\n' + '-' * 50)
