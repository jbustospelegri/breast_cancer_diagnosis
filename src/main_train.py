import tensorflow
import warnings

from itertools import repeat
from multiprocessing import Queue, Process

from breast_cancer_dataset.database_generator import BreastCancerDataset
from algorithms.classification import VGG16Model, InceptionV3Model, DenseNetModel, Resnet50Model
from algorithms.segmentation import UnetVGG16Model, UnetDenseNetModel, UnetInceptionV3Model, UnetResnet50Model
from algorithms.model_ensambling import RandomForest
from algorithms.utils import training_pipe
from data_viz.visualizacion_resultados import DataVisualizer

from utils.config import MODEL_FILES, ENSEMBLER_CONFIG
from utils.functions import bulk_data, get_path

warnings.filterwarnings('ignore')


if __name__ == '__main__':

    # Se chequea la existencia de GPU's activas
    print("TF version   : ", tensorflow.__version__)
    # we'll need GPU!
    print("GPU available: ", tensorflow.config.list_physical_devices('GPU'))

    # Parámetros de entrada que serán sustituidos por las variables del usuario

    # Los valores posibles son segmentation, classification
    task_type = 'classification'
    # Los valores disponibles son PATCHES, COMPLETE_IMAGE
    experiment = 'PATCHES'
    # Nombre del experimento
    experiment_name = 'EJEC_COMPLETE_IMG'

    available_models = {
        'classification': [InceptionV3Model, DenseNetModel, Resnet50Model, VGG16Model],
        'segmentation': [UnetVGG16Model, UnetDenseNetModel, UnetInceptionV3Model, UnetResnet50Model]
    }

    # Se setean las carpetas para almacenar las variables del modelo en función del experimento.
    model_config = MODEL_FILES
    model_config.set_model_name(name=experiment_name)

    # Se inicializa el procesado de las imagenes para los distintos datasets.
    db = BreastCancerDataset(
        xlsx_io=model_config.model_db_desc_csv, img_type=experiment, task_type=task_type, test_dataset=['MIAS']
    )

    # Se generarán algunos ejemplos de la base de datos
    data_viz = DataVisualizer(config=model_config, img_type=experiment)

    print(f'{"-" * 75}\nGenerando ejemplos de Data Augmentation del set de datos.\n{"-" * 75}')
    data_viz.get_data_augmentation_examples()

    print(f'{"-" * 75}\nGenerando análisis EDA del set de datos.\n{"-" * 75}')
    data_viz.get_eda_from_df()

    print(f'{"-" * 75}\nGenerando imagenes de ejemplo de preprocesado del set de datos.\n{"-" * 75}')
    # data_viz.get_preprocessing_examples()

    # Debido a que tensorflow no libera el espacio de GPU hasta finalizar un proceso, cada modelo se entrenará en
    # un subproceso daemonico para evitar la sobrecarga de memoria.
    for weight_init, frozen_layers in zip([*repeat('imagenet', 6), 'random'], ['ALL', '0FT', '1FT', '2FT', '3FT', '4FT',
                                                                               'ALL']):
        for cnn in available_models[task_type]:
            q = Queue()

            # Se rea el proceso
            p = Process(target=training_pipe, args=(cnn, db, q, model_config, task_type, weight_init, frozen_layers))

            # Se lanza el proceso
            p.start()

            # Se recuperan los resultados. El metodo get es bloqueante hasta que se obtiene un resultado.
            predictions = q.get()

            if task_type == 'classification':

                # Se almacenan los resultados de cada modelo.
                path = get_path(model_config.model_predictions_cnn_dir, weight_init, frozen_layers,
                                f'{cnn.__name__.replace("Model", "")}.csv')
                bulk_data(path, **predictions.to_dict())

    if task_type == 'classification':
        # Se crea el grandom forest
        print(f'{"-" * 75}\nGeneradando combinación secuencial de clasificadores.\n{"-" * 75}')
        ensambler = RandomForest(db=db.df)
        ensambler.train_model(
            cnn_predictions_dir=get_path(model_config.model_predictions_cnn_dir),
            save_model_dir=get_path(model_config.model_store_ensembler_dir, ENSEMBLER_CONFIG),
            out_predictions_dir=get_path(model_config.model_predictions_ensembler_dir, ENSEMBLER_CONFIG)
        )

        print(f'{"-" * 50}\nProceso de entrenamiento finalizado\n{"-" * 50}')

    print(f'{"="* 75}\nGeneradando visualización de resultados.\n{"="* 75}')

    print(f'{"-" * 75}\nRepresentando métricas del entrenamiento.\n{"-" * 75}')
    data_viz.get_model_logs_metrics(logs_dir=model_config.model_log_dir)

    print(f'{"-" * 75}\nRepresentando tiempos de entrenamiento.\n{"-" * 75}')
    data_viz.get_model_time_executions(summary_dir=model_config.model_summary_dir)

    print(f'{"-" * 75}\nGenerando matrices de confusión de los modelos.\n{"-" * 75}')
    data_viz.get_model_predictions_metrics(
        cnn_predictions_dir=model_config.model_predictions_cnn_dir,
        ensembler_predictions_dir=model_config.model_predictions_ensembler_dir
    )
