from algorithms.models import VGG16Model, Resnet50Model, InceptionV3Model, DenseNetModel, ModelEnsamble
from data_viz.dataset import TestDataset

from utils.config import PREDICTIONS_PATH, MODEL_SAVE_DATA_PATH, TEST_DATA_PATH, CLASS_LABELS
from utils.functions import f1_score, get_predictions

from keras.models import load_model

import os


if __name__ == '__main__':

    # Parametros a introducir por el usuario. Deben coincidir con los de tentrenamiento
    batch_size = 16
    stored_models_test_name = 'prueba_3'

    # Constantes del entrenamiento
    predictions = {}
    filename_predictions = f'test_{stored_models_test_name}.csv'
    ensamble_model_filepath = os.path.join(MODEL_SAVE_DATA_PATH,
                                           f'GradientBoostingClassifier_{stored_models_test_name}.sav')
    models = [
        VGG16Model(n_clases=6, name='VGG16', test_name=stored_models_test_name, get_model_structure=False),
        Resnet50Model(n_clases=6, name='ResNet50', test_name=stored_models_test_name, get_model_structure=False),
        InceptionV3Model(n_clases=6, name='InceptionV3', test_name=stored_models_test_name, get_model_structure=False),
        DenseNetModel(n_clases=6, name='DenseNet121', test_name=stored_models_test_name, get_model_structure=False),
    ]

    print('-' * 50 + f'\nGenerando predicciones\n' + '-' * 50)

    # Se iteran los modelos para ir generando las predicciones.
    for cnn in models:

        # Se recupera el path del modelo almacenado
        path_model = os.path.join(MODEL_SAVE_DATA_PATH, f'{cnn.name}_{stored_models_test_name}.h5')

        print('-' * 50 + f'\nObteniendo set de datos de {TEST_DATA_PATH}\n' + '-' * 50)
        data = TestDataset(path=TEST_DATA_PATH)
        data.get_dataset()
        test_datagen = data.get_dataset_generator(
            batch_size=batch_size, size=cnn.input_shape, preproces_callback=cnn.preprocess_func
        )
        print('-' * 50 + f'\nSet de datos cargado correctamente\n' + '-' * 50)

        print('-' * 50 + f'\nCargando modelo {path_model}\n' + '-' * 50)
        model = load_model(path_model, custom_objects={'f1_score': f1_score})
        print('-' * 50 + f'\nModelo cargado correctamente\n' + '-' * 50)

        print('-' * 50 + f'\nGenerando predicciones de {cnn.name}\n' + '-' * 50)
        predictions[cnn.name] = get_predictions(keras_model=model, data=test_datagen, class_labels=CLASS_LABELS,
                                                save_csv=False)
        print('-' * 50 + f'\nPredicciones generadas correctamente\n' + '-' * 50)

    # Se recupera el modelo de gradient boosting para realizar el ensamblado
    model_ensambler = ModelEnsamble(ensamble_model_filepath)

    print('-' * 50 + f'\nAlmacenando predicciones en {os.path.join(PREDICTIONS_PATH, filename_predictions)}\n' +
          '-' * 50)
    model_ensambler.predict(dirname=PREDICTIONS_PATH, filename=filename_predictions, data=test_datagen,
                            return_model_predictions=True, **predictions)
    print('-' * 50 + f'\nProceso de predicción finalizado correctamente\n' + '-' * 50)
