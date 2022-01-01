import os
import pickle
from itertools import product, repeat

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from typing import io

from breast_cancer_dataset.base import Dataloder
from utils.config import SEED, ENSEMBLER_COLS, ENSEMBLER_CONFIG
from utils.functions import get_path, bulk_data, search_files, get_filename


class RandomForest:
    __name__ = 'RandomForest'
    PARAMETERS_GRID = {
        'n_estimators': np.arange(50, 500, 50),
        'max_depth': np.arange(0, 10)
    }

    def __init__(self, db: Dataloder = None):
        self.db = db
        self.clf = None

    def load_model(self, model_filepath: io):
        assert os.path.isfile(model_filepath), f"File doesn't exists {model_filepath}"
        self.clf = pickle.load(open(model_filepath, 'rb'))

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
                right=df.set_index('PROCESSED_IMG').rename(columns={'PREDICTION': model})[[model]],
                right_index=True,
                left_index=True,
                how='left'
            )

        return data

    def train_model(self, cnn_predictions_dir: io, out_predictions_dir: io, save_model_dir: io):
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
        data = self.db.loc[self.db.TRAIN_VAL != 'test',
                           ['PROCESSED_IMG', 'IMG_LABEL', 'TRAIN_VAL', *ENSEMBLER_COLS[ENSEMBLER_CONFIG]]].copy()
        data.loc[:, 'LABEL'] = data.IMG_LABEL.map({k: v for v, k in enumerate(sorted(data.IMG_LABEL.unique()))})

        merge_list = []
        for weight, frozen_layers in zip([*repeat('imagenet', 6), 'random'],
                                         ['ALL', '0FT', '1FT', '2FT', '3FT', '4FT', 'ALL']):

            l = []
            for file in search_files(get_path(cnn_predictions_dir, weight, frozen_layers, create=False), ext='csv'):
                l.append(
                    pd.read_csv(file, sep=';')[['PROCESSED_IMG', 'PREDICTION']]. \
                        assign(WEIGHTS=weight, FT=frozen_layers, CNN=get_filename(file))
                )

            merge_list.append(
                pd.merge(left=data, right=pd.concat(l, ignore_index=True), on='PROCESSED_IMG', how='left')
            )

        all_data = pd.concat(merge_list, ignore_index=True)

        # Se escoge el mejor modelo en función de las metricas AUC de validacion
        cnn_selection = all_data.groupby(['CNN', 'FT', 'WEIGHTS', 'TRAIN_VAL'], as_index=False).apply(
            lambda x: pd.Series({'AUC': roc_auc_score(x.LABEL, x.PREDICTION)})
        )

        selected_cnns = cnn_selection[cnn_selection.TRAIN_VAL == 'val'].sort_values('AUC', ascending=False). \
            groupby('CNN', as_index=False).first()

        selected_cnns.to_csv(get_path(save_model_dir, 'Selected CNN Report.csv'), sep=';', index=False, decimal=',')

        final_list = []
        for _, row in selected_cnns.iterrows():
            final_list.append(
                all_data[(all_data.CNN == row.CNN) & (all_data.FT == row.FT) & (all_data.WEIGHTS == row.WEIGHTS)]
            )

        final_df = pd.concat(final_list, ignore_index=True). \
            set_index(['PROCESSED_IMG', 'LABEL', 'IMG_LABEL', 'TRAIN_VAL', *ENSEMBLER_COLS[ENSEMBLER_CONFIG], 'CNN']) \
            ['PREDICTION'].unstack().reset_index()

        # generación del conjunto de datos de train para gradient boosting
        data.dropna(how='any', inplace=True)
        cols = [*ENSEMBLER_COLS[ENSEMBLER_CONFIG], *all_data.CNN.unique().tolist()]
        x, y = final_df.loc[:, cols], final_df.loc[:, 'LABEL']

        clf = GridSearchCV(
            estimator=RandomForestRegressor(random_state=SEED),
            param_grid=self.PARAMETERS_GRID,
            scoring='roc_auc',
            cv=PredefinedSplit(test_fold=np.where(final_df.TRAIN_VAL == 'train', -1, 0))
        )
        clf.fit(x, y)

        data_csv = final_df[['PROCESSED_IMG', 'TRAIN_VAL', 'IMG_LABEL']].assign(PREDICTION=clf.predict(final_df[cols]))

        bulk_data(file=get_path(out_predictions_dir, f'{self.__name__}.csv'), **data_csv.to_dict())

        self.clf = clf.best_estimator_

        # se almacena el modelo en caso de que el usuario haya definido un nombre de archivo
        pickle.dump(clf.best_estimator_, open(get_path(save_model_dir, f'{self.__name__}.sav'), 'wb'))

    def predict(self, data: Dataloder, **kwargs):
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
        assert self.clf, 'Please,load model using load_model method'
        dataset = pd.DataFrame(index=data.filenames)
        dataset.index.name = 'PROCESSED_IMG'

        # Se unifica el set de datos obteniendo las predicciones de cada modelo representadas por input_models
        df = self.get_dataframe_from_kwargs(dataset, **kwargs)

        df_encoded = pd.get_dummies(df[kwargs.keys()])
        for c in [col for col in df_encoded.columns if col not in self.clf.feature_names_in_]:
            df_encoded.loc[:, c] = 0

        # Se añaden las predicciones
        df.loc[:, 'PATHOLOGY'] = self.clf.predict(df_encoded)

        # se escribe el log de errores con las predicciones individuales de cada arquitectura de red o únicamente las
        # generadas por gradient boosting
        return df.reset_index()[['PROCESSED_IMG', 'PATHOLOGY']]
