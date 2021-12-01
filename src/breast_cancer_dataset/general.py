import json
import logging
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import io, Callable
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, Iterator
from sklearn.model_selection import train_test_split

from breast_cancer_dataset.cbis_ddsm import DatasetCBISDDSM, DatasetCBISDDSMCrop
from breast_cancer_dataset.inbreast import DatasetINBreast
from breast_cancer_dataset.mias import DatasetMIAS, DatasetMIASCrop
from data_viz.functions import create_countplot, plot_image
from utils.config import (
    MODEL_FILES, SEED, DATA_AUGMENTATION_FUNCS, TRAIN_DATA_PROP, PREPROCESSING_FUNCS, IMG_SHAPE, PREPROCESSING_CONFIG,
    EXPERIMENT, DF_COLS
)
from utils.functions import get_filename, get_path


class BreastCancerDataset:

    DBS = {
            'COMPLETE_IMAGE': [DatasetCBISDDSM, DatasetMIAS, DatasetINBreast],
            'PATCHES': [DatasetCBISDDSMCrop], # , DatasetMIASCrop],
            'MASK': []
        }

    def __init__(self, get_class: bool = True, split_data: bool = True, excel_path: str = ''):

        if not os.path.isfile(excel_path):
            self.df = self.get_data_from_databases()
            self.split_dataset(train_prop=TRAIN_DATA_PROP if split_data else 1, stratify=True)
            self.bulk_data_desc_to_files(df=self.df)
            self.get_eda_from_df(dirname=MODEL_FILES.model_viz_eda_dir)
        else:
            self.df = pd.read_excel(
                excel_path, header=None, skiprows=1, names=[*DF_COLS, 'TRAIN_VAL'], dtype=object, index_col=None
            )

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

    def get_dataset_generator(self, batch_size: int, preprocessing_function: Callable, directory: io = None,
                              size: tuple = (IMG_SHAPE, IMG_SHAPE)) -> (Iterator, Iterator):
        """

        Función que permite recuperar un dataframe iterator para entrenamiento y para validación.

        :param directory: booleano que permite almacenar las imagenes generadas por el ImageDataGenerator en la carpeta
                          Data/03_OUTPUT/DATA AUGMENTATION.
        :param batch_size: tamaño de batch con el que se crearán los iteradores.
        :param size: tamaño de la imagen que servirá de input para los iteradores. Si la imagen tiene un tamaño distinto
                     se aplicará un resize aplicando la tecnica de interpolación lanzcos. Por defecto es 224, 224.
        :param preprocessing_function: función de preprocesado a aplicar a las imagenes leidas una vez aplicadas las
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
            **DATA_AUGMENTATION_FUNCS, fill_mode='constant', cval=0, preprocessing_function=preprocessing_function
        )

        # Se plotea las transformaciones que sufre una imagen en caso de indicarse el parámetro directory
        if directory:
            self.get_data_augmentation_examples(
                out_filepath=directory,
                example_imag=self.df.iloc[random.sample(self.df.index.tolist(), 1)[0]].PREPROCESSED_IMG
            )

        # Parametrización del generador de validación. Las imagenes de validación exclusivamente se les aplicará la
        # técnica de preprocesado subministrada por el usuario.
        val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

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

    @staticmethod
    def get_data_augmentation_examples(out_filepath: io, example_imag: io) -> None:
        """
        Función que permite generar un ejemplo de cada tipo de data augmentation aplicado
        :param out_filepath: ruta del archivo de imagen a generar
        :param example_imag: nombre de una muestra de ejemplo sobre la que se aplicarán las transformaciones propias del
                             data augmentation
        """

        # Se lee la imagen del path de ejemplo
        image = load_img(example_imag)
        # Se transforma la imagen a formato array
        image = img_to_array(image)
        # Se añade una dimensión para obtener el dato de forma (1, width, height, channels)
        image_ori = np.expand_dims(image, axis=0)

        # Figura y subplots de matplotlib. Debido a que existen 4 transformaciones de data augmentation, se creará un
        # grid con 5 columnas que contendrán cada ejemplo de transformación y la imagen original
        elements = len(DATA_AUGMENTATION_FUNCS.keys())
        cols = 3
        rows = elements // cols + elements % cols
        fig = plt.figure(figsize=(15, 4 * rows))

        # Se representa la imagen original en el primer subplot.
        plot_image(img=image_ori, title='Imagen Original', ax_=fig.add_subplot(rows, cols, 1))

        # Se iteran las transformaciones
        for i, (k, v) in enumerate(DATA_AUGMENTATION_FUNCS.items(), 2):

            # Se crea al datagenerator con exclusivamente la transformación a aplicar.
            datagen = ImageDataGenerator(**{k: v}, fill_mode='constant', cval=0)
            # Se recupera la imagen transformada mediante next() del método flow del objeto datagen
            plot_image(img=next(datagen.flow(image_ori)), title=k, ax_=fig.add_subplot(rows, cols, i))

        # Se ajusta la figura
        fig.tight_layout()

        # Se almacena la figura
        plt.savefig(get_path(out_filepath, f'{get_filename(example_imag)}.png'))

    def get_eda_from_df(self, dirname: io) -> None:
        """
        Función que permite representar graficamente el número de observaciones y la proporción de cada una de las
        clases presentes en un dataet. La clase de cada observción debe estar almacenada en una columna cuyo
        nombre sea "class".

        :param dirname: directorio en el que se almacenará la imagen.
        """

        print(f'{"-" * 75}\n\tGenerando análisis del dataset\n{"-" * 75}')
        title = 'Distribución clases según orígen'
        file = get_path(dirname, f'{title}.png')
        create_countplot(x='DATASET', hue='IMG_LABEL', data=self.df, title=title, file=file)

        title = 'Distribución clases'
        file = get_path(dirname, f'{title}.png')
        create_countplot(x='IMG_LABEL', data=self.df, title=title, file=file)

        title = 'Distribución clases segun train-val'
        file = get_path(dirname, f'{title}.png')
        create_countplot(x='TRAIN_VAL', hue='IMG_LABEL', data=self.df, title=title, file=file, norm=True)

        title = 'Distribución clases segun patología'
        file = get_path(dirname, f'{title}.png')
        create_countplot(x='ABNORMALITY_TYPE', hue='IMG_LABEL', data=self.df, title=title, file=file, norm=True)
        print(f'{"-" * 75}\n\tAnálisis del dataset finalizado en {dirname}\n{"-" * 75}')
