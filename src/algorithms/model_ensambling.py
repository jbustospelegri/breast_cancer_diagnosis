import os
import pickle
from itertools import repeat

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from typing import io

from breast_cancer_dataset.base import Dataloder
from utils.config import SEED, ENSEMBLER_COLS, ENSEMBLER_CONFIG
from utils.functions import get_path, bulk_data, search_files, get_filename


class RandomForest:
    """

        Clase para realizar el model ensembling de las predicciones de cada red neuronal

    """
    __name__ = 'RandomForest'
    PARAMETERS_GRID = {
        'n_estimators': np.arange(50, 500, 50),
        'max_depth': np.arange(0, 10)
    }

    def __init__(self, db: Dataloder = None):
        self.db = db
        self.clf = None

    def load_model(self, model_filepath: io):
        """
        Función que recupera un modelo almacenado
        :param model_filepath: ruta del modelo
        """
        assert os.path.isfile(model_filepath), f"File doesn't exists {model_filepath}"
        self.clf = pickle.load(open(model_filepath, 'rb'))

    @staticmethod
    def get_dataframe_from_kwargs(data: pd.DataFrame, **model_inputs):
        """
        Función utilizada para generar un set de datos unificado a partir de las predicciones generadas por cada modelo
        :param data: dataframe con una columna (PROCESSED_IMG) que contiene el nombre de archivo
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
        Función utilizada para generar el algoritmo de Random Forest a partir de las predicciones generadas por cada
        modelo.
        :param cnn_predictions_dir: directorio que contiene las predicciones de cada red neuronal
        :param out_predictions_dir: nombre del archivo en el que se almacenarán las predicciones del modelo
        :param save_model_dir:  carpeta para guardar el algoritmo de random forest
        """

        # Se obtiene información de las observaciones a partir del dataset genrado por database_generator.py. Ademas,
        # se pueden añadir columnas extras a partir del parámetro ENSEMBLER_COLS. Las columnas obtenidas del dataset
        # seran el file de la imagen (PROCESSED_IMG), su etiqueta (IMG_LABEL) y a qué conjunto de datos pertenece
        # (TRAIN_VAL)
        data = self.db.loc[self.db.TRAIN_VAL != 'test',
                           ['PROCESSED_IMG', 'IMG_LABEL', 'TRAIN_VAL', *ENSEMBLER_COLS[ENSEMBLER_CONFIG]]].copy()
        # Se obtiene el mapping de cada clase con un numero
        data.loc[:, 'LABEL'] = data.IMG_LABEL.map({k: v for v, k in enumerate(sorted(data.IMG_LABEL.unique()))})

        # Se iteran las predicciones de cada red neuronal para crear un único dataframe con las predicciones de cada
        # red para cada observacion
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

        # Se alamcenan los mejores modelos
        selected_cnns.to_csv(get_path(save_model_dir, 'Selected CNN Report.csv'), sep=';', index=False, decimal=',')

        # Se filtran los mejores modelos del dataframe original para crear el conjunto de entrenamiento del random
        # forest
        final_list = []
        for _, row in selected_cnns.iterrows():
            final_list.append(
                all_data[(all_data.CNN == row.CNN) & (all_data.FT == row.FT) & (all_data.WEIGHTS == row.WEIGHTS)]
            )

        final_df = pd.concat(final_list, ignore_index=True). \
            set_index(['PROCESSED_IMG', 'LABEL', 'IMG_LABEL', 'TRAIN_VAL', *ENSEMBLER_COLS[ENSEMBLER_CONFIG], 'CNN']) \
            ['PREDICTION'].unstack().reset_index()

        # generación del conjunto de datos de train para random forest.
        data.dropna(how='any', inplace=True)
        cols = [*ENSEMBLER_COLS[ENSEMBLER_CONFIG], *all_data.CNN.unique().tolist()]
        x, y = final_df.loc[:, cols], final_df.loc[:, 'LABEL']

        # Grid search de parámetros. El cross validation se aplicara utilizando el set de validación
        clf = GridSearchCV(
            estimator=RandomForestRegressor(random_state=SEED),
            param_grid=self.PARAMETERS_GRID,
            scoring='roc_auc',
            cv=PredefinedSplit(test_fold=np.where(final_df.TRAIN_VAL == 'train', -1, 0))
        )
        clf.fit(x, y)

        # Se almacenan las predicciones del random forest
        data_csv = final_df[['PROCESSED_IMG', 'TRAIN_VAL', 'IMG_LABEL']].assign(PREDICTION=clf.predict(final_df[cols]))

        bulk_data(file=get_path(out_predictions_dir, f'{self.__name__}.csv'), **data_csv.to_dict())

        # Se almacena el mejor modelo obtenido por el grid search.
        self.clf = clf.best_estimator_

        # se almacena el modelo en un archivo .sac
        pickle.dump(clf.best_estimator_, open(get_path(save_model_dir, f'{self.__name__}.sav'), 'wb'))

    def predict(self, data: Dataloder, threshold: float = 0.5, **kwargs):
        """
        Función utilizada para realizar la predicción del algorítmo de random forest a partir de las predicciones
        del conjunto de redes convolucionales

        :param data: objeto dataloader que contiene los datos de las predicciones a realizar
        :param threshold: humbral para considerar una muestra maligna o benigna
        :param kwargs: diccionario en que cada key será el nombre del modelo de deep learning utilizado y los values
                       sus correspondientes predicciones
        """

        # Se genera un dataframe con los paths de las imagenes a predecir como índices
        assert self.clf, 'Please,load model using load_model method'
        dataset = pd.DataFrame(index=data.filenames)
        dataset.index.name = 'PROCESSED_IMG'

        # Se unifica el set de datos obteniendo las predicciones de cada modelo representadas por input_models
        df = self.get_dataframe_from_kwargs(dataset, **kwargs)

        # Se añaden las predicciones
        df.loc[:, 'MALIGNANT_PROBABILITY'] = self.clf.predict(df[self.clf.feature_names_in_])

        # Se mete el threshold
        df.loc[:, 'PATHOLOGY'] = np.where(df.MALIGNANT_PROBABILITY > threshold, 'MALIGNANT', 'BENIGN')

        # se retorna la predicción juntamente con la probabilidad de cancer maligno
        return df.reset_index()[['PROCESSED_IMG', 'PATHOLOGY', 'MALIGNANT_PROBABILITY']]
