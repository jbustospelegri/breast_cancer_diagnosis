import json
import logging
import os

import cv2
import pandas as pd
import numpy as np

from typing import Callable, io
from albumentations import Lambda, Compose
from tensorflow import cast, float32
from tensorflow.keras.preprocessing.image import Iterator
from sklearn.model_selection import train_test_split

from breast_cancer_dataset.base import SegmentationDataset, Dataloder, ClassificationDataset
from breast_cancer_dataset.databases.cbis_ddsm import DatasetCBISDDSM, DatasetCBISDDSMCrop
from breast_cancer_dataset.databases.inbreast import DatasetINBreast, DatasetINBreastCrop
from breast_cancer_dataset.databases.mias import DatasetMIAS, DatasetMIASCrop
from utils.config import (
    MODEL_FILES, SEED, CLASSIFICATION_DATA_AUGMENTATION_FUNCS, TRAIN_DATA_PROP, PREPROCESSING_FUNCS,
    PREPROCESSING_CONFIG, IMG_SHAPE, SEGMENTATION_DATA_AUGMENTATION_FUNCS, PATCH_SIZE
)
from preprocessing.image_processing import resize_img


class BreastCancerDataset:

    DBS = {
            'COMPLETE_IMAGE': [DatasetCBISDDSM, DatasetINBreast, DatasetMIAS],
            'PATCHES': [DatasetCBISDDSMCrop, DatasetINBreastCrop, DatasetMIASCrop]
        }

    def __init__(
            self,
            img_type: str,
            task_type: str,
            get_class: bool = True,
            split_data: bool = True,
            xlsx_io: io = '',
            test_dataset: list = None
    ):

        if img_type not in list(self.DBS.keys()):
            raise ValueError(f'img_type param {img_type} not implemented')
        if task_type not in ['classification', 'segmentation']:
            raise ValueError(f'task_type param {task_type} not implemented')

        if split_data:
            train_proportion = TRAIN_DATA_PROP
        else:
            train_proportion = 1

        self.img_type = img_type
        self.task_type = task_type

        if not os.path.isfile(xlsx_io):
            self.df = self.get_data_from_databases()
            self.split_dataset(train_prop=train_proportion, strat_cols=['IMG_LABEL', 'DATASET'], test_id=test_dataset)
            self.bulk_data_desc_to_files(df=self.df)
        else:
            self.df = pd.read_excel(xlsx_io, dtype=object, index_col=None)

        if get_class:
            self.class_dict = {x: l for x, l in enumerate(self.df.IMG_LABEL.unique())}
        else:
            self.class_dict = {}

    def get_data_from_databases(self) -> pd.DataFrame:

        df_list = []
        for database in self.DBS[self.img_type]:

            # Se inicializa la base de datos
            db = database()

            db.start_pipeline()

            df_list.append(db.df_desc[db.DF_COLS])

        return pd.concat(objs=df_list, ignore_index=True)

    def get_dataset_generator(self, **kwargs):
        if self.task_type == 'classification':
            return self.get_classification_dataset_generator(**kwargs)
        else:
            return self.get_segmentation_dataset_generator(**kwargs)

    def get_classification_dataset_generator(self, batch_size: int, callback: Callable, size: tuple = PATCH_SIZE) -> \
            (Iterator, Iterator):
        """

        Funci??n que permite recuperar un dataframe iterator para entrenamiento y para validaci??n.

        :param batch_size: tama??o de batch con el que se crear??n los iteradores.
        :param size: tama??o de la imagen que servir?? de input para los iteradores. Si la imagen tiene un tama??o distinto
                     se aplicar?? un resize aplicando la tecnica de interpolaci??n lanzcos. Por defecto es 224, 224.
        :param callback: funci??n de preprocesado a aplicar a las imagenes leidas una vez aplicadas las
                                       t??cnicas de data augmentation.
        :return: dataframeIterator de validaci??n y de tran.
        """

        # Se crea una configuraci??n por defecto para crear los dataframe iterators. En esta, se leer??n los paths de las
        # imagenes a partir de una columna llamada 'preprocessing_filepath' y la clase de cada imagen estar??
        # representada por la columna 'img_label'. Para ajustar el tama??o de la imagen al tama??o definido por el
        # usuario mediante input, se utilizar?? la t??cnica de interpolaci??n lanzcos. Por otra parte, para generar una
        # salida one hot encoding en funci??n de la clase de cada muestra, se parametriza class_mode como 'categorical'.
        train_augmentations = Compose([
            *list(CLASSIFICATION_DATA_AUGMENTATION_FUNCS.values()),
            Lambda(
                image=lambda x, **kgs: resize_img(x, height=size[0], width=size[1], interpolation=cv2.INTER_LANCZOS4),
                name='image resizing'
            ),
            Lambda(image=lambda x, **kwargs: cast(x, float32), name='floating point conversion'),
            Lambda(image=callback, name='cnn processing function')
        ])

        val_augmentations = Compose([
            Lambda(
                image=lambda x, **kgs: resize_img(x, height=size[0], width=size[1], interpolation=cv2.INTER_LANCZOS4),
                name='image resizing'
            ),
            Lambda(image=lambda x, **kwargs: cast(x, float32), name='floating point conversion'),
            Lambda(image=callback, name='cnn processing function')
        ])

        # Para evitar entrecruzamientos de imagenes entre train y validaci??n a partir del atributo shuffle=True, cada
        # generador se aplicar?? sobre una muestra disjunta del set de datos representada mediante la columna dataset.

        # Se chequea que existen observaciones de entrenamiento para poder crear el dataframeiterator.
        if len(self.df[self.df.TRAIN_VAL == 'train']) == 0:
            train_dataloader = None
            logging.warning('No existen registros para generar un generador de train. Se retornar?? None')
        else:
            # Dataset for train images
            train_df = ClassificationDataset(
                self.df[self.df.TRAIN_VAL == 'train'], 'PROCESSED_IMG', 'IMG_LABEL', train_augmentations
            )
            train_dataloader = Dataloder(train_df, batch_size=batch_size, shuffle=True, seed=SEED)

        # Se chequea que existen observaciones de validaci??n para poder crear el dataframeiterator.
        if len(self.df[self.df.TRAIN_VAL == 'val']) == 0:
            valid_dataloader = None
            logging.warning('No existen registros para generar un generador de validaci??n. Se retornar?? None')
        else:
            # Dataset for validation images
            val_df = ClassificationDataset(
                self.df[self.df.TRAIN_VAL == 'val'], 'PROCESSED_IMG', 'IMG_LABEL', val_augmentations
            )
            valid_dataloader = Dataloder(val_df, batch_size=batch_size, shuffle=True, seed=SEED)

        return train_dataloader, valid_dataloader

    def get_segmentation_dataset_generator(self, batch_size: int, callback: Callable, size: tuple = IMG_SHAPE) \
            -> (Iterator, Iterator):
        """

        Funci??n que permite recuperar un dataframe iterator para entrenamiento y para validaci??n.

        :param batch_size: tama??o de batch con el que se crear??n los iteradores.
        :param size: tama??o de la imagen que servir?? de input para los iteradores. Si la imagen tiene un tama??o distinto
                     se aplicar?? un resize aplicando la tecnica de interpolaci??n lanzcos. Por defecto es 224, 224.
        :param callback: funci??n de preprocesado a aplicar a las imagenes leidas una vez aplicadas las
                                       t??cnicas de data augmentation.
        :return: dataframeIterator de validaci??n y de tran.
        """

        # Se crea una configuraci??n por defecto para crear los dataframe iterators. En esta, se leer??n los paths de las
        # imagenes a partir de una columna llamada 'preprocessing_filepath' y la clase de cada imagen estar??
        # representada por la columna 'img_label'. Para ajustar el tama??o de la imagen al tama??o definido por el
        # usuario mediante input, se utilizar?? la t??cnica de interpolaci??n lanzcos. Por otra parte, para generar una
        # salida one hot encoding en funci??n de la clase de cada muestra, se parametriza class_mode como 'categorical'.

        train_augmentations = Compose([
            *list(SEGMENTATION_DATA_AUGMENTATION_FUNCS.values()),
            Lambda(
                image=lambda x, **kgs: resize_img(x, height=size[0], width=size[1], interpolation=cv2.INTER_LANCZOS4),
                name='image resizing'
            ),
            Lambda(
                mask=lambda x, **kgs: resize_img(x, height=size[0], width=size[1], interpolation=cv2.INTER_NEAREST),
                name='mask resizing'
            ),
            Lambda(mask=lambda x, **kwargs: np.float32(x.round().clip(0, 1)), name='normalize mask'),
            Lambda(image=lambda x, **kwargs: cast(x, float32), name='image floating point conversion'),
            Lambda(image=callback, name='cnn processing function')
        ])

        val_augmentations = Compose([
            Lambda(
                image=lambda x, **kgs: resize_img(x, height=size[0], width=size[1], interpolation=cv2.INTER_LANCZOS4),
                name='image resizing'
            ),
            Lambda(
                mask=lambda x, **kgs: resize_img(x, height=size[0], width=size[1], interpolation=cv2.INTER_NEAREST),
                name='mask resizing'
            ),
            Lambda(mask=lambda x, **kwargs: np.float32(x.round().clip(0, 1)), name='normalize mask'),
            Lambda(image=lambda x, **kwargs: cast(x, float32), name='image floating point conversion'),
            Lambda(image=callback, name='cnn processing function')
        ])

        # Para evitar entrecruzamientos de imagenes entre train y validaci??n a partir del atributo shuffle=True, cada
        # generador se aplicar?? sobre una muestra disjunta del set de datos representada mediante la columna dataset.

        # Se chequea que existen observaciones de entrenamiento para poder crear el dataframeiterator.
        if len(self.df[self.df.TRAIN_VAL == 'train']) == 0:
            train_dataloader = None
            logging.warning('No existen registros para generar un generador de train. Se retornar?? None')
        else:
            # Dataset for train images
            train_df = SegmentationDataset(
                self.df[self.df.TRAIN_VAL == 'train'], 'PROCESSED_IMG', 'PROCESSED_MASK', train_augmentations
            )
            train_dataloader = Dataloder(train_df, batch_size=batch_size, shuffle=True, seed=SEED)

        # Se chequea que existen observaciones de validaci??n para poder crear el dataframeiterator.
        if len(self.df[self.df.TRAIN_VAL == 'val']) == 0:
            valid_dataloader = None
            logging.warning('No existen registros para generar un generador de validaci??n. Se retornar?? None')
        else:
            # Dataset for validation images
            val_df = SegmentationDataset(
                self.df[self.df.TRAIN_VAL == 'val'], 'PROCESSED_IMG', 'PROCESSED_MASK', val_augmentations
            )
            valid_dataloader = Dataloder(val_df, batch_size=batch_size, shuffle=True, seed=SEED)

        return train_dataloader, valid_dataloader

    def split_dataset(self, train_prop: float, strat_cols: list = None, test_id: list = None):
        """
        Funci??n que permite dividir el dataset en un subconjunto de entrenamiento y otro de validaci??n. La divisi??n
        puede ser estratificada.
        Para realizar la divisi??n se crear?? una columna nueva en el atributo desc_df indicando a que subconjunto de
        datos pertenece cada observaci??n.

        :param train_prop: proporci??n del total de observaciones del dataset con los que se crear?? el conjunto de train
        :param strat_cols: lista de columnas pertenecientes al pandas dataframe para realizar el estratficado de
                           observaciones al realizar el split en train y validacion.
        :param test_id: lista de valores para la columna DATASET que pertenecer??n a la partici??n de test
        """

        if test_id is None:
            test_id = []

        # Se confirma que el valor de train_prop no sea superior a 1
        assert train_prop < 1, 'Proporci??n de datos de validaci??n superior al 100%'

        # Se filtran los datasets del campo test_id para no incorporarlos en el split de train y validacion
        df = self.df[~self.df.DATASET.isin(test_id)].copy()

        if strat_cols is None:
            stratify = None
        else:
            stratify = df[strat_cols].apply(lambda r: '_'.join(r.values.astype(str)), axis=1)

        # Se filtran los datos en funci??n de si se desea obtener el conjunto de train y val
        train_x, _, _, _ = train_test_split(
            df.PROCESSED_IMG, df.IMG_LABEL, random_state=SEED, train_size=train_prop, stratify=stratify
        )

        # Se asigna el valor de 'train' a aquellas imagenes (representadas por sus paths) que est??n presentes en train_x
        # en caso contrario, se asignar?? el valor 'val'.
        self.df.loc[:, 'TRAIN_VAL'] = np.where(
            self.df.DATASET.isin(test_id), 'test', np.where(self.df.PROCESSED_IMG.isin(train_x), 'train', 'val')
        )

    @staticmethod
    def bulk_data_desc_to_files(df: pd.DataFrame) -> None:
        """
        Funci??n escribe el feedback del dataset en la carpeta de destino de la conversi??n.
        """
        print(f'{"-" * 75}\n\tBulking data to {MODEL_FILES.model_db_desc_csv}\n{"-" * 75}')
        df.to_excel(MODEL_FILES.model_db_desc_csv, index=False)

        print(f'\tBulking preprocessing functions to {MODEL_FILES.model_db_processing_info_file}\n{"-" * 75}')
        with open(MODEL_FILES.model_db_processing_info_file, 'w') as out:
            json.dump(PREPROCESSING_FUNCS[PREPROCESSING_CONFIG], out, indent=4)
