from itertools import repeat

from breast_cancer_dataset.datasets import BreastCancerDataset
from data_viz.visualizacion_resultados import DataVisualizer
from algorithms.model_ensambling import GradientBoosting
from algorithms.cnns import InceptionV3Model, Resnet50Model, DenseNetModel, VGG16Model

from utils.config import MODEL_CONSTANTS, PREPROCESSING_CONFIG

from multiprocessing import Queue, Process

from algorithms.functions import training_pipe


if __name__ == '__main__':

    # Parámetros de entrada que serán sustituidos por las variables del usuario
    experiment = 'complete_imag'

    # Se setean las carpetas para almacenar las variables del modelo en función del experimento.
    model_config = MODEL_CONSTANTS
    model_config.set_model_name(name=experiment)

    # Se inicializa el procesado de las imagenes para los distintos datasets.
    db = BreastCancerDataset(preprocesing_conf=PREPROCESSING_CONFIG)

    # Debido a que tensorflow no libera el espacio de GPU hasta finalizar un proceso, cada modelo se entrenará en
    # un subproceso daemonico para evitar la sobrecarga de memoria.
    for weight_init, frozen_layers in zip(['random', *repeat('imagenet', 6)], ['ALL', '0FT', '1FT', '2FT', '3FT', '4FT',
                                                                               'ALL']):

        # Diccionario en el que se almacenarán las predicciones de cada modelo. Estas serán utilizadas para aplicar el
        # algorítmo de gradient boosting.
        model_predictions = {}

        for cnn in [InceptionV3Model, Resnet50Model, DenseNetModel, VGG16Model]:

            # Queue que servirá para recuparar las predicciones de cada modelo.
            q = Queue()

            # Se rea el proceso
            p = Process(target=training_pipe, args=(cnn, db, q, model_config, weight_init, frozen_layers), daemon=True)

            # Se lanza el proceso
            p.start()

            # Se recuperan los resultados. El metodo get es bloqueante hasta que se obtiene un resultado.
            predictions = q.get()

            # Se almacenan los resultados de cada modelo.
            model_predictions[cnn] = predictions

        print(f'{"-" * 75}\nGeneradando combinación secuencial de clasificadores.\n{"-" * 75}')

        # hacer un bulking de las predicciones a un csv

        # ensambler = GradientBoosting()
        # df_ensambler = db.df.rename(
        #     columns={'preprocessing_filepath': 'filenames', 'img_label': 'true_label', 'dataset': 'mode'}
        # ).copy()
        # ensambler.train_model(
        #     train=df_ensambler[df_ensambler['mode'] == 'train'], val=df_ensambler[df_ensambler['mode'] == 'val'],
        #     filename=model_cte.model_predictions_filepath, model_dirname=model_cte.model_dirname,
        #     **model_predictions
        # )
        # print('-' * 50 + f'\nProceso de entrenamiento finalizado\n' + '-' * 50)
    #
    # print('-' * 50 + f'\nGeneradando visualización de resultados.\n' + '-' * 50)
    # data_viz = DataVisualizer()
    #
    # print('-' * 50 + f'\nRepresentando accuracy, loss y f1_score para finetunning y transferLearning.\n' + '-' * 50)
    # data_viz.get_model_logs_metrics(
    #     logs_dir=model_cte.model_logs_dirname, test_name=experiment, dirname=model_cte.model_results_dirname,
    #     out_filename=f'model_history_{experiment}', train_phases=['TransferLearning', 'FineTunning'])
    #
    # print('-' * 50 + f'\nRepresentando accuracy, loss y f1_score globales.\n' + '-' * 50)
    # data_viz.get_model_logs_metrics(
    #     model_cte.model_logs_dirname, experiment, model_cte.model_results_dirname, f'model_history_{experiment}')
    #
    # print('-' * 50 + f'\nGenerando matrices de confusión de los modelos.\n' + '-' * 50)
    # data_viz.plot_confusion_matrix(
    #     dirname=model_cte.model_results_dirname, out_file=f'confusion_matrix_{experiment}',
    #     input_file=model_cte.model_predictions_filepath
    # )
    #
    # print('-' * 50 + f'\nRecopilando métricas de Recall, Accuracy, F1_score y precisión.\n' + '-' * 50)
    # data_viz.get_metrics_matrix(
    #     dirname=model_cte.model_results_dirname, out_file=f'model_metrics_{experiment}', class_metrics=False,
    #     input_file=model_cte.model_predictions_filepath
    # )
    # print('-' * 50 + f'\nVisualizaciones finalizadas\n' + '-' * 50)
