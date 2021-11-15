import os
import pickle

import pandas as pd
import numpy as np

from keras_preprocessing.image import DataFrameIterator
from sklearn.ensemble import GradientBoostingClassifier
from typing import io
from itertools import product


from utils.config import SEED
from utils.functions import get_path


class GradientBoosting:

    def __init__(self, model_path: io = None):
        if os.path.exists(model_path or ''):
            self.model_gb = pickle.load(open(model_path, 'rb'))
        else:
            self.model_gb = GradientBoostingClassifier(max_depth=3, n_estimators=20, random_state=SEED)

    @staticmethod
    def get_dataframe_from_models(data: pd.DataFrame, **model_inputs):
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

    def train_model(self, train: pd.DataFrame, filename: io, val: pd.DataFrame = None, model_dirname: io = '',
                    **models_ensamble):
        """
        Función utilizada para generar el algorítmo de gradient boosting a partir de las predicciones generadas por cada
        modelo.
        :param train: Dataframe que contendrá los identificadores (filenames) del conjunto del set de entrenamiento y
                      una columna 'mode' cuyo valor sera train. Además deberá contener la columna true_label con las
                      clases verdaderas de cada observación
        :param filename: nombre del archivo en el que se almacenarán las predicciones del modelo
        :param val:  Dataframe que contendrá los identificadores (filenames) del conjunto del set de validación y
                     una columna 'mode' cuyo valor sera val. Además deberá contener la columna true_label con las
                      clases verdaderas de cada observación
        :param model_dirname: nombre del archivo con el que se guardará el modelo.
        :param models_ensamble: kwargs que contendrá como key el nombre del modelo y como values los valores devueltos
                                por el método predict de cada modelo
        """

        # En caso de existir dataset de validación, se concatena train y val en un dataset único. En caso contrario,
        # se recupera unicamente el set de datos de train
        if val is not None:
            gb_dataset = pd.concat(objs=[train.set_index('filenames'), val.set_index('filenames')], ignore_index=False)
        else:
            gb_dataset = train

        # se asigna el nombre del indice
        gb_dataset.index.name = 'file'

        # Se unifica el set de datos obteniendo las predicciones de cada modelo representadas por models ensamble
        df = self.get_dataframe_from_models(gb_dataset, **models_ensamble)

        # generación del conjunto de datos de train para gradient boosting
        train_gb_x = pd.concat(
            objs=[
                pd.DataFrame(columns=[f'{m}_{c}' for m, c in product(models_ensamble.keys(), CLASS_LABELS.values())]),
                pd.get_dummies(df[df['mode'].str.lower() == 'train'][list(models_ensamble.keys())])
            ], ignore_index=False
        ).fillna(0)
        train_gb_y = df[df['mode'].str.lower() == 'train'][['true_label']]

        # generación del conjunto de datos de validación
        if val is not None:
            val_gb_x = pd.concat(
                objs=[
                    pd.DataFrame(
                        columns=[f'{m}_{c}' for m, c in product(models_ensamble.keys(), CLASS_LABELS.values())]),
                    pd.get_dummies(df[df['mode'].str.lower() == 'val'][list(models_ensamble.keys())])
                ], ignore_index=False
            ).fillna(0)

        # entrenamiento del modelo
        self.model_gb.fit(train_gb_x, np.reshape(train_gb_y.values, -1))

        # Se define un dataframe con las columnas resultantes:
        df_predict = pd.concat(
            objs=[
                pd.DataFrame(columns=[f'{m}_{c}' for m, c in product(models_ensamble.keys(), CLASS_LABELS.values())]),
                pd.get_dummies(train_gb_x),
                pd.get_dummies(val_gb_x)
            ], ignore_index=False
        ).fillna(0)

        # se añade al dataset original, las predicciones del modelo de gradient boosting
        df.loc[:, self.__class__.__name__] = df_predict.assign(LABEL=self.model_gb.predict(df_predict)).LABEL

        # se almacenan las predicciones
        df.reset_index().to_csv(filename, sep=';', index=False)

        # se almacena el modelo en caso de que el usuario haya definido un nombre de archivo
        if model_dirname:
            pickle.dump(self.model_gb, open(get_path(model_dirname, 'GradientBoosterClassifier.sav'), 'wb'))

    def predict(self, dirname: str, filename: str, data: DataFrameIterator, return_model_predictions: bool = False,
                **input_models):
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
        df = self.get_dataframe_from_models(gb_dataset, **input_models)

        # Se añaden las predicciones
        df.loc[:, 'label'] = self.model_gb.predict(pd.get_dummies(df[input_models.keys()]))

        # se escribe el log de errores con las predicciones individuales de cada arquitectura de red o únicamente las
        # generadas por gradient boosting
        if return_model_predictions:
            df.to_csv(get_path(dirname, filename), sep=';')
        else:
            df[['label']].to_csv(get_path(dirname, filename), sep=';')
