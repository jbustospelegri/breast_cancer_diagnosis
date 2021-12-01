from itertools import repeat

import tensorflow

from multiprocessing import Queue, Process

from algorithms.model_ensambling import GradientBoosting
from breast_cancer_dataset.database_generator import BreastCancerDataset
from algorithms.cnns import VGG16Model, InceptionV3Model, DenseNetModel, Resnet50Model
from algorithms.functions import training_pipe
from data_viz.visualizacion_resultados import DataVisualizer

from utils.config import MODEL_FILES, PREPROCESSING_CONFIG, XGB_CONFIG
from utils.functions import bulk_data, get_path, get_filename, get_dirname


if __name__ == '__main__':

    # Se chequea la existencia de GPU's activas
    print("TF version   : ", tensorflow.__version__)
    # we'll need GPU!
    print("GPU available: ", tensorflow.config.list_physical_devices('GPU'))

    # Parámetros de entrada que serán sustituidos por las variables del usuario
    experiment = 'CROPPED_IMAG'

    # Se setean las carpetas para almacenar las variables del modelo en función del experimento.
    model_config = MODEL_FILES
    model_config.set_model_name(name=experiment)

    # Se inicializa el procesado de las imagenes para los distintos datasets.
    db = BreastCancerDataset(excel_path=model_config.model_db_desc_csv)

    # Debido a que tensorflow no libera el espacio de GPU hasta finalizar un proceso, cada modelo se entrenará en
    # un subproceso daemonico para evitar la sobrecarga de memoria.
    for weight_init, frozen_layers in zip([*repeat('imagenet', 6), 'random'], ['0FT', '1FT', '2FT', '3FT', '4FT',
                                                                               'ALL', 'ALL']):
        # Diccionario en el que se almacenarán las predicciones de cada modelo. Estas serán utilizadas para aplicar el
        # algorítmo de gradient boosting.
        for cnn in [VGG16Model, InceptionV3Model, Resnet50Model, DenseNetModel]:
            q = Queue()

            # Se rea el proceso
            p = Process(target=training_pipe, args=(cnn, db, q, model_config, weight_init, frozen_layers))

            # Se lanza el proceso
            p.start()

            # Se recuperan los resultados. El metodo get es bloqueante hasta que se obtiene un resultado.
            predictions = q.get()

            # Se almacenan los resultados de cada modelo.
            path = get_path(model_config.model_predictions_cnn_dir, weight_init, frozen_layers,
                            f'{cnn.__name__.replace("Model", "")}.csv')
            bulk_data(path, **predictions.to_dict())

        # Se crea el gradient boosting
        print(f'{"-" * 75}\nGeneradando combinación secuencial de clasificadores.\n{"-" * 75}')
        ensambler = GradientBoosting(db=db.df)
        ensambler.train_model(
            cnn_predictions_dir=get_path(model_config.model_predictions_cnn_dir, weight_init, frozen_layers),
            save_model_dir=get_path(model_config.model_store_xgb_dir, XGB_CONFIG, weight_init, frozen_layers),
            xgb_predictions_dir=get_path(model_config.model_predictions_xgb_dir, XGB_CONFIG, weight_init, frozen_layers)
        )
        print('-' * 50 + f'\nProceso de entrenamiento finalizado\n' + '-' * 50)

    print(f'{"="* 75}\nGeneradando visualización de resultados.\n{"="* 75}')
    data_viz = DataVisualizer()

    print(f'{"-" * 75}\nRepresentando métricas del entrenamiento.\n{"-" * 75}')
    data_viz.get_model_logs_metrics(logs_dir=model_config.model_log_dir)

    print(f'{"-" * 75}\nRepresentando tiempos de entrenamiento.\n{"-" * 75}')
    data_viz.get_model_time_executions(summary_dir=model_config.model_summary_dir)

    print(f'{"-" * 75}\nGenerando matrices de confusión de los modelos.\n{"-" * 75}')
    data_viz.get_model_predictions_metrics(predictions_dir=model_config.model_predictions_dir)

    print(f'{"-" * 75}\nGenerando ejemplos de Data Augmentation del set de datos.\n{"-" * 75}')
    data_viz.get_data_augmentation_examples()

    print(f'{"-" * 75}\nGenerando análisis EDA del set de datos.\n{"-" * 75}')
    data_viz.get_eda_from_df()
