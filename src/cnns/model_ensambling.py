import os
import pickle
from itertools import product

import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.image import Iterator
from sklearn.ensemble import GradientBoostingClassifier
from typing import io

from src.utils.config import SEED, XGB_COLS, XGB_CONFIG, N_ESTIMATORS, MAX_DEPTH, MODEL_FILES
from src.utils.functions import get_path, bulk_data, search_files, get_filename


class GradientBoosting:

    __name__ = 'GradientBoosting'

    def __init__(self, db: pd.DataFrame = None):
        self.db = db
        self.model_gb = GradientBoostingClassifier(max_depth=N_ESTIMATORS, n_estimators=MAX_DEPTH, random_state=SEED)

    def load_model(self, model_filepath: io):
        assert os.path.exists(model_filepath), f"File doesn't exists {model_filepath}"
        self.model_gb = pickle.load(open(model_filepath, 'rb'))

    @staticmethod
    def get_dataframe_from_kwargs(data: pd.DataFrame, **model_inputs):
        """
        Función utilizada para generar un set de datos unificado a partir de las predicciones generadas por cada modelo
        :param data: dataframe con el nombre de archivo
        :param model_inputs: kwargs cuya key será el nombre del modelo que genera las predicciones y cuyo value será
                             un dataframe formado por las predicciones (columna predictions) y el filename de cada
                             observación.
        :return: dataframe unificado
        """
        for model, df in model_inputs.items():
            data = pd.merge(
                left=data,
                right=df.set_index('filename').rename(columns={'predictions': model})[[model]],
                right_index=True,
                left_index=True,
                how='left'
            )
        return data

    def train_model(self, cnn_predictions_dir: io, xgb_predictions_dir: io, save_model_dir: io):
        """
        Función utilizada para generar el algorítmo de gradient boosting a partir de las predicciones generadas por cada
        modelo.
        :param train: Dataframe que contendrá los identificadores (filenames) del conjunto del set de entrenamiento y
                      una columna 'mode' cuyo valor sera train. Además deberá contener la columna true_label con las
                      clases verdaderas de cada observación
        :param cnn_predictions_dir: nombre del archivo en el que se almacenarán las predicciones del modelo
        :param val:  Dataframe que contendrá los identificadores (filenames) del conjunto del set de validación y
                     una columna 'mode' cuyo valor sera val. Además deberá contener la columna true_label con las
                      clases verdaderas de cada observación
        :param model_dirname: nombre del archivo con el que se guardará el modelo.
        :param models_ensamble: kwargs que contendrá como key el nombre del modelo y como values los valores devueltos
                                por el método predict de cada modelo
        """

        # En caso de existir dataset de validación, se concatena train y val en un dataset único. En caso contrario,
        # se recupera unicamente el set de datos de train
        data = self.db[['PROCESSED_IMG', 'IMG_LABEL', 'TRAIN_VAL', *XGB_COLS[XGB_CONFIG]]].copy()
        cols = XGB_COLS[XGB_CONFIG].copy()
        for file in search_files(cnn_predictions_dir, 'csv', in_subdirs=False):
            model_name = get_filename(file)
            df = pd.read_csv(file, sep=';')[['PROCESSED_IMG', 'PREDICTION']]
            df_dumy = pd.concat(
                objs=[
                    pd.DataFrame(
                        data=[['0', '0', '0']],
                        columns=['PROCESSED_IMG',
                                 *[f'{n}_{l}' for n, l in list(product([model_name], data.IMG_LABEL.unique()))]]
                    ),
                    pd.get_dummies(df.rename(columns={'PREDICTION': model_name}), columns=[model_name]).astype(str)
                ],
                ignore_index=True
            ).ffill()
            cols += [f'{n}_{l}' for n, l in list(product([model_name], data.IMG_LABEL.unique()))]
            data = pd.merge(left=data, right=df_dumy, on='PROCESSED_IMG', how='left')

        # generación del conjunto de datos de train para gradient boosting
        data.dropna(how='any', inplace=True)
        train_x, train_y = data.loc[data.TRAIN_VAL == 'train', cols], data.loc[data.TRAIN_VAL == 'train', 'IMG_LABEL']

        # entrenamiento del modelo
        self.model_gb.fit(train_x, np.reshape(train_y.values, -1))

        # se almacenan las predicciones
        data_csv = data[['PROCESSED_IMG', 'IMG_LABEL', 'TRAIN_VAL']].\
            assign(PREDICTION=self.model_gb.predict(data[cols]))

        bulk_data(file=get_path(xgb_predictions_dir, f'{self.__name__}.csv'), **data_csv.to_dict())

        # se almacena el modelo en caso de que el usuario haya definido un nombre de archivo
        pickle.dump(self.model_gb, open(get_path(save_model_dir, f'{self.__name__}.sav'), 'wb'))

    def predict(self, dirname: str, filename: str, data: Iterator, return_model_predictions: bool = False, **kwargs):
        """
        Función utilizada para realizar la predicción del algorítmo de graadient boosting a partir de las predicciones
        del conjunto de redes convolucionales

        :param dirname: directorio en el que se almacenará el log de predicciones
        :param filename: nombre del archivo en el que se almacenará el log de predicciones
        :param data: dataframe que contiene el nombre de cada imagen en una columna llamada filenames
        :param return_model_predictions: booleano que permite recuperar en el log de predicciones, las predicciones
                                         individuales de cada red neuronal convolucional
        :param input_models: kwargs que contendrá como key el nombre del modelo y como values los valores devueltos
                             por el método predict de cada modelo de red neuronal convolucional
        """

        # Se genera un dataframe con los directorios de las imagenes a predecir
        gb_dataset = pd.DataFrame(index=data.filenames)
        gb_dataset.index.name = 'image'

        # Se unifica el set de datos obteniendo las predicciones de cada modelo representadas por input_models
        df = self.get_dataframe_from_kwargs(gb_dataset, **kwargs)

        # Se añaden las predicciones
        df.loc[:, 'label'] = self.model_gb.predict(pd.get_dummies(df[kwargs.keys()]))

        # se escribe el log de errores con las predicciones individuales de cada arquitectura de red o únicamente las
        # generadas por gradient boosting
        if return_model_predictions:
            df.to_csv(get_path(dirname, filename), sep=';')
        else:
            df[['label']].to_csv(get_path(dirname, filename), sep=';')
