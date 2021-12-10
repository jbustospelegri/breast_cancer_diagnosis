import json
import logging
import os

import pandas as pd
import numpy as np

from typing import Callable
from tensorflow.keras.preprocessing.image import ImageDataGenerator, Iterator
from sklearn.model_selection import train_test_split

from src.breast_cancer_dataset.databases.cbis_ddsm import DatasetCBISDDSM, DatasetCBISDDSMCrop
from src.breast_cancer_dataset.databases.inbreast import DatasetINBreast, DatasetINBreastCrop
from src.breast_cancer_dataset.databases.mias import DatasetMIAS, DatasetMIASCrop
from src.utils.config import (
    MODEL_FILES, SEED, DATA_AUGMENTATION_FUNCS, TRAIN_DATA_PROP, PREPROCESSING_FUNCS, PATCH_SIZE, PREPROCESSING_CONFIG,
    EXPERIMENT, IMG_SHAPE
)


class BreastCancerDataset:

    DBS = {
            'COMPLETE_IMAGE': [DatasetCBISDDSM, DatasetMIAS, DatasetINBreast],
            'PATCHES': [DatasetCBISDDSMCrop, DatasetMIASCrop, DatasetINBreastCrop],
            'MASK': []
        }

    def __init__(self, get_class: bool = True, split_data: bool = True, excel_path: str = ''):

        if not os.path.isfile(excel_path):
            self.df = self.get_data_from_databases()
            self.split_dataset(train_prop=TRAIN_DATA_PROP if split_data else 1, stratify=True)
            self.bulk_data_desc_to_files(df=self.df)
        else:
            self.df = pd.read_excel(excel_path, dtype=object, index_col=None)

        if get_class:
            self.class_dict = {x: l for x, l in enumerate(self.df.IMG_LABEL.unique())}
        else:
            self.class_dict = {}

    def get_data_from_databases(self) -> pd.DataFrame:

        df_list = []
        for database in self.DBS[EXPERIMENT]:

            # Se inicializa la base de datos
            db = database()

            db.start_pipeline()

            df_list.append(db.df_desc)

        return pd.concat(objs=df_list, ignore_index=True)

    def get_dataset_generator(self, batch_size: int, preproces_func: Callable, size: tuple = IMG_SHAPE) -> \
            (Iterator, Iterator):
        """

        Función que permite recuperar un dataframe iterator para entrenamiento y para validación.

        :param directory: booleano que permite almacenar las imagenes generadas por el ImageDataGenerator en la carpeta
                          Data/03_OUTPUT/DATA AUGMENTATION.
        :param batch_size: tamaño de batch con el que se crearán los iteradores.
        :param size: tamaño de la imagen que servirá de input para los iteradores. Si la imagen tiene un tamaño distinto
                     se aplicará un resize aplicando la tecnica de interpolación lanzcos. Por defecto es 224, 224.
        :param preproces_func: función de preprocesado a aplicar a las imagenes leidas una vez aplicadas las
                                       técnicas de data augmentation.
        :return: dataframeIterator de validación y de tran.
        """

        # Se crea una configuración por defecto para crear los dataframe iterators. En esta, se leerán los paths de las
        # imagenes a partir de una columna llamada 'preprocessing_filepath' y la clase de cada imagen estará
        # representada por la columna 'img_label'. Para ajustar el tamaño de la imagen al tamaño definido por el
        # usuario mediante input, se utilizará la técnica de interpolación lanzcos. Por otra parte, para generar una
        # salida one hot encoding en función de la clase de cada muestra, se parametriza class_mode como 'categorical'.
        params = dict(
            x_col='PREPROCESSED_IMG',
            y_col='IMG_LABEL',
            target_size=size,
            interpolation='lanczos',
            shufle=True,
            seed=SEED,
            batch_size=batch_size,
            class_mode='categorical',
            directory=None,
        )

        # Parametrización del generador de entrenamiento. Las imagenes de entrenamiento recibirán un conjunto de
        # modificaciones aleatorias con el objetivo de aumentar el set de datos de entrenamiento y evitar de esta forma
        # el over fitting.
        train_datagen = ImageDataGenerator(
            **DATA_AUGMENTATION_FUNCS, fill_mode='constant', cval=0, preprocessing_function=preproces_func
        )

        # Parametrización del generador de validación. Las imagenes de validación exclusivamente se les aplicará la
        # técnica de preprocesado subministrada por el usuario.
        val_datagen = ImageDataGenerator(preprocessing_function=preproces_func)

        # Para evitar entrecruzamientos de imagenes entre train y validación a partir del atributo shuffle=True, cada
        # generador se aplicará sobre una muestra disjunta del set de datos representada mediante la columna dataset.

        # Se chequea que existen observaciones de entrenamiento para poder crear el dataframeiterator.
        if len(self.df[self.df.TRAIN_VAL == 'train']) == 0:
            train_df_iter = None
            logging.warning('No existen registros para generar un generador de train. Se retornará None')
        else:
            train_df_iter = train_datagen.flow_from_dataframe(dataframe=self.df[self.df.TRAIN_VAL == 'train'], **params)

        # Se chequea que existen observaciones de validación para poder crear el dataframeiterator.
        if len(self.df[self.df.TRAIN_VAL == 'val']) == 0:
            val_df_iter = None
            logging.warning('No existen registros para generar un generador de validación. Se retornará None')
        else:
            val_df_iter = val_datagen.flow_from_dataframe(dataframe=self.df[self.df.TRAIN_VAL == 'val'], **params)

        return train_df_iter, val_df_iter

    def get_segmentation_dataset_generator(self, batch_size: int, preproces_func: Callable, size: tuple = IMG_SHAPE) \
            -> (Iterator, Iterator):
        """

        Función que permite recuperar un dataframe iterator para entrenamiento y para validación.

        :param directory: booleano que permite almacenar las imagenes generadas por el ImageDataGenerator en la carpeta
                          Data/03_OUTPUT/DATA AUGMENTATION.
        :param batch_size: tamaño de batch con el que se crearán los iteradores.
        :param size: tamaño de la imagen que servirá de input para los iteradores. Si la imagen tiene un tamaño distinto
                     se aplicará un resize aplicando la tecnica de interpolación lanzcos. Por defecto es 224, 224.
        :param preproces_func: función de preprocesado a aplicar a las imagenes leidas una vez aplicadas las
                                       técnicas de data augmentation.
        :return: dataframeIterator de validación y de tran.
        """

        # Se crea una configuración por defecto para crear los dataframe iterators. En esta, se leerán los paths de las
        # imagenes a partir de una columna llamada 'preprocessing_filepath' y la clase de cada imagen estará
        # representada por la columna 'img_label'. Para ajustar el tamaño de la imagen al tamaño definido por el
        # usuario mediante input, se utilizará la técnica de interpolación lanzcos. Por otra parte, para generar una
        # salida one hot encoding en función de la clase de cada muestra, se parametriza class_mode como 'categorical'.
        params = dict(
            target_size=size,
            interpolation='lanczos',
            shufle=True,
            seed=SEED,
            batch_size=batch_size,
            class_mode=None,
            directory=None,
        )

        # Parametrización del generador de entrenamiento. Las imagenes de entrenamiento recibirán un conjunto de
        # modificaciones aleatorias con el objetivo de aumentar el set de datos de entrenamiento y evitar de esta forma
        # el over fitting.
        train_datagen = ImageDataGenerator(
            **DATA_AUGMENTATION_FUNCS, fill_mode='constant', cval=0, preprocessing_function=preproces_func
        )

        # Parametrización del generador de validación. Las imagenes de validación exclusivamente se les aplicará la
        # técnica de preprocesado subministrada por el usuario.
        val_datagen = ImageDataGenerator(preprocessing_function=preproces_func)

        # Para evitar entrecruzamientos de imagenes entre train y validación a partir del atributo shuffle=True, cada
        # generador se aplicará sobre una muestra disjunta del set de datos representada mediante la columna dataset.

        # Se chequea que existen observaciones de entrenamiento para poder crear el dataframeiterator.
        if len(self.df[self.df.TRAIN_VAL == 'train']) == 0:
            train_df_iter_img, train_df_iter_mask = None, None
            logging.warning('No existen registros para generar un generador de train. Se retornará None')
        else:
            train_df_iter_img = train_datagen.flow_from_dataframe(
                dataframe=self.df[self.df.TRAIN_VAL == 'train'], x_col='PREPROCESSED_IMG', **params
            )
            train_df_iter_mask = train_datagen.flow_from_dataframe(
                dataframe=self.df[self.df.TRAIN_VAL == 'train'], x_col='MASK_IMG', **params
            )

        # Se chequea que existen observaciones de validación para poder crear el dataframeiterator.
        if len(self.df[self.df.TRAIN_VAL == 'val']) == 0:
            val_df_iter_img = None
            val_df_iter_mask = None
            logging.warning('No existen registros para generar un generador de validación. Se retornará None')
        else:
            val_df_iter_img = val_datagen.flow_from_dataframe(
                dataframe=self.df[self.df.TRAIN_VAL == 'val'], x_col='PREPROCESSED_IMG', **params
            )
            val_df_iter_mask = val_datagen.flow_from_dataframe(
                dataframe=self.df[self.df.TRAIN_VAL == 'val'], x_col='MASK_IMG', **params
            )

        return zip(train_df_iter_img, train_df_iter_mask), zip(val_df_iter_img, val_df_iter_mask)

    def split_dataset(self, train_prop: float, stratify: bool = True):
        """
        Función que permite dividir el dataset en un subconjunto de entrenamiento y otro de validación. La división
        puede ser estratificada.
        Para realizar la división se creará una columna nueva en el atributo desc_df indicando a que subconjunto de
        datos pertenece cada observación.

        :param train_prop: proporción del total de observaciones del dataset con los que se creará el conjunto de train
        :param stratify: booleano que determina si la división debe ser estratificada o no.
        """

        # Se confirma que el valor de train_prop no sea superior a 1
        assert train_prop < 1, 'Proporción de datos de validación superior al 100%'

        # Se filtran los datos en función de si se desea obtener el conjunto de train y val
        train_x, _, _, _ = train_test_split(
            self.df.PREPROCESSED_IMG, self.df.IMG_LABEL, random_state=SEED, train_size=train_prop,
            stratify=self.df.IMG_LABEL if stratify else None
        )

        # Se asigna el valor de 'train' a aquellas imagenes (representadas por sus paths) que estén presentes en train_x
        # en caso contrario, se asignará el valor 'val'.
        self.df.loc[:, 'TRAIN_VAL'] = np.where(self.df.PREPROCESSED_IMG.isin(train_x), 'train', 'val')

    @staticmethod
    def bulk_data_desc_to_files(df: pd.DataFrame) -> None:
        """
        Función escribe el feedback del dataset en la carpeta de destino de la conversión.
        """
        print(f'{"-" * 75}\n\tBulking data to {MODEL_FILES.model_db_desc_csv}\n{"-" * 75}')
        df.to_excel(MODEL_FILES.model_db_desc_csv, index=False)

        print(f'\tBulking preprocessing functions to {MODEL_FILES.model_db_processing_info_file}\n{"-" * 75}')
        with open(MODEL_FILES.model_db_processing_info_file, 'w') as out:
            json.dump(PREPROCESSING_FUNCS[PREPROCESSING_CONFIG], out, indent=4)
