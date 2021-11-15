import json
import logging
import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import io, Callable, List
from keras_preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array, DataFrameIterator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_viz.functions import create_countplot, plot_image
from utils.config import (
    CBIS_DDSM_DB_PATH, CBIS_DDSM_CALC_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TRAIN, CBIS_DDSM_MASS_CASE_DESC_TEST,
    CBIS_DDSM_MASS_CASE_DESC_TRAIN, CBIS_DDSM_CONVERTED_DATA_PATH, CBIS_DDSM_PREPROCESSED_DATA_PATH,
    MIAS_DB_PATH, MIAS_CASE_DESC, MIAS_CONVERTED_DATA_PATH, MIAS_PREPROCESSED_DATA_PATH,
    INBREAST_DB_PATH, INBREAST_CASE_DESC, INBREAST_CONVERTED_DATA_PATH, INBREAST_PREPROCESSED_DATA_PATH,
    MODEL_FILES, SEED, DATA_AUGMENTATION_FUNCS, TRAIN_DATA_PROP, PREPROCESSING_FUNCS
)
from utils.functions import get_filename, get_dirname, get_path, search_files
from utils.image_processing import convert_img, image_processing

INFO_COLS = [
    'DATASET', 'BREAST', 'BREAST_VIEW', 'BREAST_DENSITY', 'ABNORMALITY_TYPE', 'IMG_TYPE', 'RAW_IMG',
    'CONVERTED_IMG', 'PREPROCESSED_IMG', 'IMG_LABEL'
]


class BreastCancerDataset:

    def __init__(self, preprocesing_conf):
        self.databases = [DatasetMIAS, DatasetINBreast, DatasetCBISDDSM]
        self.df = self.__get_data_from_databases(preprocesing_conf)
        self.class_dict = {x: l for x, l in enumerate(self.df.IMG_LABEL.unique())}
        self.__split_dataset(train_prop=TRAIN_DATA_PROP, stratify=True)
        self.__get_eda_from_df(dir=MODEL_FILES.model_viz_eda_dir)
        self.__bulk_data_desc_to_files(df=self.df, conf=preprocesing_conf)

    def get_dataset_generator(self, batch_size: int, size: tuple = (224, 224), directory: io = None,
                              preprocessing_function: Callable = None):
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
            directory=None
        )

        # Parametrización del generador de entrenamiento. Las imagenes de entrenamiento recibirán un conjunto de
        # modificaciones aleatorias con el objetivo de aumentar el set de datos de entrenamiento y evitar de esta forma
        # el over fitting.
        train_datagen = ImageDataGenerator(**DATA_AUGMENTATION_FUNCS, preprocessing_function=preprocessing_function)

        # Se plotea las transformaciones que sufre una imagen en caso de indicarse el parámetro directory
        if directory:
            self.__get_data_augmentation_examples(
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

    def __get_data_from_databases(self, preprocessing_conf: str) -> pd.DataFrame:

        l = []
        for database in self.databases:

            # Se inicializa la base de datos
            db = database()

            # # Se realiza la conversión de formato dicom a png.
            # db.convert_images_format()
            #
            # # Se realiza el preprocesado de las fotografías.
            # db.preproces_images(conf=preprocessing_conf, show_example=True)

            l.append(db.df_desc)

        return pd.concat(objs=l, ignore_index=True)

    def __split_dataset(self, train_prop: float, stratify: bool = False):
        """
        Función que permite dividir el dataset en un subconjunto de entrenamiento y otro de validación. La división
        puede ser estratificada.
        Para realizar la división se creará una columna nueva en el atributo desc_df indicando a que subconjunto de
        datos pertenece cada observación.

        :param train_prop: proporción del total de observaciones del dataset con los que se creará el conjunto de train
        :param stratify: booleano que determina si la división debe ser estratificada o no.
        :param img_type: indica el tipo de imágenes a tener en cuenta para realizar la división: FULL o CROP.
        """

        # Se confirma que el valor de train_prop no sea superior a 1
        assert train_prop < 1, 'Proporción de datos de validación superior al 100%'

        # Se filtran los datos en función de si se desea obtener el conjunto de train y val
        train_x, _, _, _ = train_test_split(
            self.df.PREPROCESSED_IMG, self.df.IMG_LABEL, random_state=SEED,
            stratify=self.df.IMG_LABEL if stratify else None
        )

        # Se asigna el valor de 'train' a aquellas imagenes (representadas por sus paths) que estén presentes en train_x
        # en caso contrario, se asignará el valor 'val'.
        self.df.loc[:, 'TRAIN_VAL'] = np.where(self.df.PREPROCESSED_IMG.isin(train_x), 'train', 'val')

    @staticmethod
    def __bulk_data_desc_to_files(df: pd.DataFrame, conf: str) -> None:
        """
        Función escribe el feedback del dataset en la carpeta de destino de la conversión.
        """
        print(f'{"-" * 75}\n\tBulking data to {MODEL_FILES.model_db_desc_csv}\n{"-" * 75}')
        df.to_excel(MODEL_FILES.model_db_desc_csv, index=False)

        print(f'\tBulking preprocessing functions to {MODEL_FILES.model_db_processing_info_file}\n{"-" * 75}')
        with open(MODEL_FILES.model_db_processing_info_file, 'w') as out:
            json.dump(PREPROCESSING_FUNCS[conf], out, indent=4)

    @staticmethod
    def __get_data_augmentation_examples(out_filepath: io, example_imag: io) -> None:
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
        fig = plt.figure(figsize=(15, 5 * rows))

        # Se representa la imagen original en el primer subplot.
        plot_image(img=image_ori, title='Imagen Original', ax_=fig.add_subplot(rows, cols, 1))

        # Se iteran las transformaciones
        for i, (k, v) in enumerate(DATA_AUGMENTATION_FUNCS.items(), 2):

            # Se crea al datagenerator con exclusivamente la transformación a aplicar.
            datagen = ImageDataGenerator(**{k: v})
            # Se recupera la imagen transformada mediante next() del método flow del objeto datagen
            plot_image(img=next(datagen.flow(image_ori)), title=k, ax_=fig.add_subplot(rows, cols, i))

        # Se ajusta la figura
        fig.tight_layout()

        # Se almacena la figura
        plt.savefig(get_path(out_filepath, f'{get_filename(example_imag)}.png'))

    def __get_eda_from_df(self, dir: io) -> None:
        """
        Función que permite representar graficamente el número de observaciones y la proporción de cada una de las
        clases presentes en un dataet. La clase de cada observción debe estar almacenada en una columna cuyo
        nombre sea "class".

        :param dir: directorio en el que se almacenará la imagen.
        :param filename: nombre de la imagen.
        """

        print(f'{"-" * 75}\n\tGenerando análisis del dataset\n{"-" * 75}')
        title = 'Distribución clases según orígen'
        create_countplot(x='DATASET', hue='IMG_LABEL', data=self.df, title=title, file=get_path(dir, f'{title}.png'))

        title = 'Distribución clases'
        create_countplot(x='IMG_LABEL', data=self.df, title=title, file=get_path(dir, f'{title}.png'))

        title = 'Distribución clases segun train-val'
        file = get_path(dir, f'{title}.png')
        create_countplot(x='TRAIN_VAL', hue='IMG_LABEL', data=self.df, title=title, file=file, norm=True)
        print(f'{"-" * 75}\n\tAnálisis del dataset finalizado en {dir}\n{"-" * 75}')


class DatasetCBISDDSM:

    def __init__(
            self,
            ori_dir: io = CBIS_DDSM_DB_PATH,
            ori_extension: str = 'dcm',
            dest_extension: str = 'png',
            converted_dir: io = CBIS_DDSM_CONVERTED_DATA_PATH,
            procesed_dir: io = CBIS_DDSM_PREPROCESSED_DATA_PATH,
            database_info_file_paths: List[io] = None
    ):

        if database_info_file_paths is None:
            database_info_file_paths = [CBIS_DDSM_MASS_CASE_DESC_TRAIN, CBIS_DDSM_MASS_CASE_DESC_TEST,
                                        CBIS_DDSM_CALC_CASE_DESC_TRAIN, CBIS_DDSM_CALC_CASE_DESC_TEST]
        else:
            for p in database_info_file_paths:
                assert os.path.exists(p), f"Directory {p} doesn't exists."

        assert os.path.exists(ori_dir), f"Directory {ori_dir} doesn't exists."

        self.ori_extension = ori_extension
        self.dest_extension = dest_extension
        self.ori_dir = ori_dir
        self.conversion_dir = converted_dir
        self.procesed_dir = procesed_dir
        self.database_info_file_paths = database_info_file_paths
        self.df_desc = self.get_data_from_info_files()

    def get_data_from_info_files(self) -> pd.DataFrame:
        """
        Función que creará un dataframe con información del dataset CBIS-DDSM. Para cada imagen, se pretende recuperar
        la tipología (clase) detectada, el path de origen de la imagen y su nombre, el tipo de patología (massa o
        calcificación) y el nombre del estudio que contiene el identificador del paciente, el seno sobre el cual se ha
        realizado la mamografía, el tipo de mamografía (CC o MLO) y un índice para aquellas imagenes recortadas o
        mascaras. Adicionalmente, se indicarán los paths de destino para la conversión y el procesado de las imagenes.

        :return: Pandas dataframe con las columnas especificadas en la descripción
        """

        # Se crea una lista que contendrá la información de los archivos csv del set de datos
        l = []

        # Se iteran los csv con información del set de datos para unificaros
        print(f'{"-" * 70}\n\tGetting information from database {self.__class__.__name__}\n{"-" * 70}')
        for path in self.database_info_file_paths:
            l.append(pd.read_csv(path))
        df = pd.concat(objs=l, ignore_index=True)

        # Se obtienen las imagenes full de cada tipología.
        df.drop_duplicates(subset='image file path', inplace=True)

        # Se crea una columna con información acerca de qué dataset se trata.
        df.loc[:, 'DATASET'] = self.__class__.__name__

        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        df.loc[:, 'BREAST'] = df['left or right breast']

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO
        df.loc[:, 'BREAST_VIEW'] = df['image view']

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(df['abnormality type'] == 'calcification', 'CALC', 'MASS')

        # Se crea la columna IMG_TYPE que indicará si se trata de una imagen completa (FULL) o bien de una imagen
        # recortada (CROP) o mascara (MASK). En este caso, todas las imagenes son FULL
        df.loc[:, 'IMG_TYPE'] = 'FULL'

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno
        df.loc[:, 'BREAST_DENSITY'] = df.breast_density

        # Se crea la columna ID para poder linkar la información del excel con la información de las imagenes
        # almacenadas en la carpeta RAW. En este caso, se utilizará la jerarquía de carpetas en la que se almacena
        # cada imagen
        df.loc[:, 'ID'] = df['image file path'].apply(lambda x: get_dirname(x))

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGN' y 'MALIGNANT'. Se excluyen los casos de
        # patologias 'benign without callback'.
        df.loc[:, 'IMG_LABEL'] = df.pathology

        # Se descartan aquellas imagenes que pertenecen a la clase Benign without callback debido a que tal vez no
        # contienen ningúna pat
        print(f'\tExcluding {len(df[df.IMG_LABEL == "BENIGN_WITHOUT_CALLBACK"])} Benign without callback cases')
        df.drop(index=df[df.IMG_LABEL == "BENIGN_WITHOUT_CALLBACK"].index, inplace=True)

        # Se descartarán aquellas imagenes completas que presenten más de una tipología. (por ejemplo, el seno presenta
        # una zona benigna y otra maligna).
        duplicated_tags = df.groupby(['image file path']).IMG_LABEL.nunique()
        print(f'\tExcluding {len(duplicated_tags[duplicated_tags > 1])} images for ambiguous pathologys')
        df.drop(index=df[df.ID.isin(duplicated_tags[duplicated_tags > 1].index.tolist())].index, inplace=True)

        # Se recuperan los paths de las imagenes almacenadas con el formato específico (por defecto dcm) en la carpeta
        # de origen (por defecto INBREAST_DB_PATH)
        db_files_df = pd.DataFrame(data=search_files(file=self.ori_dir, ext=self.ori_extension), columns=['RAW_IMG'])

        # Se procesa la columna ori path para poder lincar cada path con los datos del excel. Para ello, se
        # obtiene la jerarquía de directorios de cada imagen.
        db_files_df.loc[:, 'ID'] = db_files_df.RAW_IMG.apply(lambda x: "/".join(get_dirname(x).split(os.sep)[-3:]))

        # Se crea la columna RAW_IMG con el path de la imagen original
        df_def = pd.merge(left=df, right=db_files_df, on='ID', how='left')

        # Se crea la clumna PREPROCESSED_IMG en la que se volcarán las imagenes preprocesadas
        df_def.loc[:, 'PREPROCESSED_IMG'] = df_def.apply(
            lambda x: get_path(
                self.procesed_dir, x.IMG_LABEL, x.IMG_TYPE, f'{x.ID.split("/")[0]}.{self.dest_extension}'
            ), axis=1
        )

        # Se crea la clumna CONVERTED_IMG en la que se volcarán las imagenes convertidas de formato
        df_def.loc[:, 'CONVERTED_IMG'] = df_def.apply(
            lambda x: get_path(
                self.conversion_dir, x.IMG_LABEL, x.IMG_TYPE, f'{x.ID.split("/")[0]}.{self.dest_extension}'
            ), axis=1
        )

        print(f'\t{len(df_def.RAW_IMG.unique())} image paths available in database')

        return df_def[INFO_COLS]

    def convert_images_format(self) -> None:
        """
        Función para convertir las imagenes del formato de origen al formato de destino.
        """

        print(f'{"-" * 75}\n\tStarting conversion of file format: {len(self.df_desc.RAW_IMG.unique())} '
              f'{self.ori_extension} files finded.')

        # Se crea el iterador con los argumentos necesarios para realizar la función a través de un multiproceso.
        arg_iter = [(row.RAW_IMG, row.CONVERTED_IMG) for _, row in self.df_desc.iterrows()]

        # Se crea un pool de multihilos para realizar la tarea de conversión de forma paralelizada.
        with Pool(processes=cpu_count() - 2) as pool:
            results = tqdm(pool.imap(convert_img, arg_iter), total=len(arg_iter), desc='conversion dcm to png')
            tuple(results)

        # Se recuperan las imagenes modificadas y se crea un dataframe
        converted_imgs = list(search_files(file=self.conversion_dir, ext=self.dest_extension))
        print(f"\tConverted {len(converted_imgs)} images to {self.dest_extension} format.\n{'-' * 75}")

    def preproces_images(self, conf: str, show_example: bool = False) -> None:
        """
        Función utilizara para realizar el preprocesado de las imagenes completas.

        :param show_example: booleano para almacenar 5 ejemplos aleatorios en la carpeta de resultados para la
        prueba realizada.
        """
        full_img_df = self.df_desc.assign(example_dir=None).copy()

        if show_example:
            photos = random.sample(full_img_df.index.tolist(), 5)
            full_img_df.loc[photos, 'example_dir'] = full_img_df.loc[photos, :].apply(
                lambda x: get_path(MODEL_FILES.model_viz_preprocesing_dir, x.DATASET, get_filename(x.PREPROCESSED_IMG)),
                axis=1
            )

        conv_imgs = list(search_files(file=self.conversion_dir, ext=self.dest_extension))
        print(f'{"-" * 75}\n\tStarting preprocessing of {len(conv_imgs)} images')

        args = [(conf, row.CONVERTED_IMG, row.PREPROCESSED_IMG, row.example_dir) for _, row in full_img_df.iterrows()]

        with Pool(processes=cpu_count() - 2) as pool:
            results = tqdm(pool.imap(image_processing, args), total=len(args), desc='preprocessing full images')
            tuple(results)

        # Se recuperan las imagenes modificadas y se crea un dataframe
        proc_imgs = list(search_files(file=self.procesed_dir, ext=self.dest_extension))
        print(f'\tProcessed {len(proc_imgs)} images.\n{"-" * 75}')


class DatasetINBreast(DatasetCBISDDSM):

    def __init__(self):
        super().__init__(
            ori_dir=INBREAST_DB_PATH,
            converted_dir=INBREAST_CONVERTED_DATA_PATH,
            procesed_dir=INBREAST_PREPROCESSED_DATA_PATH,
            database_info_file_paths=[INBREAST_CASE_DESC]
        )

    def get_data_from_info_files(self) -> pd.DataFrame:
        """

        :return:
        """

        # Se crea una lista que contendrá la información de los archivos csv del set de datos
        l = []

        # Se iteran los csv con información del set de datos para unificaros
        print(f'{"-" * 70}\n\tGetting information from database {self.__class__.__name__}\n{"-" * 70}')
        for path in self.database_info_file_paths:
            l.append(pd.read_excel(path, skipfooter=2))
        df = pd.concat(objs=l, ignore_index=True)

        # Se crea una columna con información acerca de qué dataset se trata.
        df.loc[:, 'DATASET'] = self.__class__.__name__

        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        df.loc[:, 'BREAST'] = np.where(df.Laterality == 'R', 'RIGHT', 'LEFT')

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO
        df.loc[:, 'BREAST_VIEW'] = df.View

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(df['Mass '] == 'X', 'MASS', np.where(df.Micros == 'X', 'CALC', None))

        # Se crea la columna IMG_TYPE que indicará si se trata de una imagen completa (FULL) o bien de una imagen
        # recortada (CROP) o mascara (MASK). En este caso, todas las imagenes son FULL
        df.loc[:, 'IMG_TYPE'] = 'FULL'

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno
        df.loc[:, 'BREAST_DENSITY'] = df.ACR

        # Se crea la columna ID para poder linkar la información del excel con la información de las imagenes
        # almacenadas en la carpeta RAW. En este caso, se utilizará el campo File Name
        df.loc[:, 'ID'] = df['File Name'].astype(str)

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGNA' y 'MALIGNA' en función de la puntición
        # asignada por BIRADS. Los codigos 2 se asignan a la clase benigna; los 4b, 4c, 5 y 6 a la clase maligna.
        # Se excluyen los códigos 0 (Incompletos), 3 (Probablemente benigno), 4a (Probablemente maligno (2-9%).
        df.loc[:, 'IMG_LABEL'] = np.where(
            df['Bi-Rads'].astype(str).isin(['2']), 'BENIGN',
            np.where(df['Bi-Rads'].astype(str).isin(['4b', '4c', '5', '6']), 'MALIGNANT', None)
        )

        # Se suprimen los casos que no contienen ninguna patología
        print(f'\tExcluding {len(df[df.IMG_LABEL.isnull()].index.drop_duplicates())} samples without pathologies.')
        df.drop(index=df[df.IMG_LABEL.isnull()].index, inplace=True)

        # Se descartarán aquellas imagenes completas que presenten más de una tipología. (por ejemplo, el seno presenta
        # una zona benigna y otra maligna).
        duplicated_tags = df.groupby(['ID']).IMG_LABEL.nunique()
        print(f'\tExcluding {len(duplicated_tags[duplicated_tags > 1])} images for ambiguous pathologys')
        df.drop(index=df[df.ID.isin(duplicated_tags[duplicated_tags > 1].index.tolist())].index, inplace=True)

        # Se recuperan los paths de las imagenes almacenadas con el formato específico (por defecto dcm) en la carpeta
        # de origen (por defecto INBREAST_DB_PATH)
        db_files_df = pd.DataFrame(
            data=glob(os.path.join(self.ori_dir, '**', f'*.{self.ori_extension}'), recursive=True),
            columns=['RAW_IMG']
        )

        # Se procesa la columna ori path para poder lincar cada path con los datos del excel. Para ello, se separa
        # los nombres de cara archivo a partir del símbolo _ y se obtiene la primera posición.
        db_files_df.loc[:, 'ID'] = db_files_df.RAW_IMG.apply(lambda x: get_filename(x).split('_')[0])

        # Se crea la columna RAW_IMG con el path de la imagen original
        df_def = pd.merge(left=df, right=db_files_df, on='ID', how='left')

        print(f'\t{len(df_def.RAW_IMG.unique())} image paths available in database')

        # Se crea la clumna PREPROCESSED_IMG en la que se volcarán las imagenes preprocesadas
        df_def.loc[:, 'PREPROCESSED_IMG'] = df_def.apply(
            lambda x: get_path(self.procesed_dir, x.IMG_LABEL, x.IMG_TYPE,
                               f'{get_filename(x.RAW_IMG)}.{self.dest_extension}'), axis=1
        )

        # Se crea la clumna CONVERTED_IMG en la que se volcarán las imagenes convertidas de formato
        df_def.loc[:, 'CONVERTED_IMG'] = df_def.apply(
            lambda x: get_path(self.conversion_dir, x.IMG_LABEL, x.IMG_TYPE,
                                   f'{get_filename(x.RAW_IMG)}.{self.dest_extension}'), axis=1
        )

        return df_def[INFO_COLS]


class DatasetMIAS(DatasetCBISDDSM):

    def __init__(self):
        super().__init__(
            ori_dir=MIAS_DB_PATH,
            ori_extension='pgm',
            converted_dir=MIAS_CONVERTED_DATA_PATH,
            procesed_dir=MIAS_PREPROCESSED_DATA_PATH,
            database_info_file_paths=[MIAS_CASE_DESC]
        )

    def get_data_from_info_files(self) -> pd.DataFrame:

        # Se obtiene la información del fichero de texto descriptivo del dataset
        l = []
        # Se iteran los csv con información del set de datos para unificaros
        print(f'{"-" * 70}\n\tGetting information from database {self.__class__.__name__}\n{"-" * 70}')
        for path in self.database_info_file_paths:
            l.append(
                pd.read_csv(
                    path, sep=' ', skiprows=102, skipfooter=2, engine='python',
                    names=['ID', 'BREAST_TISSUE', 'ABNORMALITY_TYPE', 'PATHOLOGY', 'X_CORD', 'Y_CORD', 'RAD']
                )
            )
        df = pd.concat(objs=l, ignore_index=True)

        # Se crea una columna con información acerca de qué dataset se trata.
        df.loc[:, 'DATASET'] = self.__class__.__name__

        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        # En este caso, no se dispone de dicha información
        df.loc[:, 'BREAST'] = None

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO. No se dispone de esta
        # información
        df.loc[:, 'BREAST_VIEW'] = None

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(df.ABNORMALITY_TYPE == 'CALC', 'CALC', 'MASS')

        # Se crea la columna IMG_TYPE que indicará si se trata de una imagen completa (FULL) o bien de una imagen
        # recortada (CROP) o mascara (MASK). En este caso, todas las imagenes son FULL
        df.loc[:, 'IMG_TYPE'] = 'FULL'

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno. Para ello se mapean los valores:
        # - 'F' (Fatty): 1
        # - 'G' (Fatty-Glandular): 2
        # - 'D' (Dense-Glandular): 3
        df.loc[:, 'BREAST_DENSITY'] = df.BREAST_TISSUE.map(defaultdict(lambda: None, {'F': '1', 'G': '2', 'D': '3'}))

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGNA' y 'MALIGNA'.
        df.loc[:, 'IMG_LABEL'] = df.PATHOLOGY.map(defaultdict(lambda: None, {'B': 'BENIGN', 'M': 'MALIGNANT'}))

        # Se suprimen los casos que no contienen ninguna patología
        print(f'\tExcluding {len(df[df.IMG_LABEL.isnull()].index.drop_duplicates())} samples without pathologies.')
        df.drop(index=df[df.IMG_LABEL.isnull()].index, inplace=True)

        # Se descartarán aquellas imagenes completas que presenten más de una tipología. (por ejemplo, el seno presenta
        # una zona benigna y otra maligna).
        duplicated_tags = df.groupby(['ID']).IMG_LABEL.nunique()
        print(f'\tExcluding {len(duplicated_tags[duplicated_tags > 1])} images for ambiguous pathologys')
        df.drop(index=df[df.ID.isin(duplicated_tags[duplicated_tags > 1].index.tolist())].index, inplace=True)

        # Se recuperan los paths de las imagenes almacenadas con el formato específico (por defecto dcm) en la carpeta
        # de origen (por defecto INBREAST_DB_PATH)
        db_files_df = pd.DataFrame(data=search_files(self.ori_dir, self.ori_extension), columns=['RAW_IMG'])

        # Se procesa la columna ori path para poder lincar cada path con los datos del excel. Para ello, se separa
        # los nombres de cara archivo a partir del símbolo _ y se obtiene la primera posición.
        db_files_df.loc[:, 'ID'] = db_files_df.RAW_IMG.apply(lambda x: get_filename(x))

        # Se crea la columna RAW_IMG con el path de la imagen original
        df_def = pd.merge(left=df, right=db_files_df, on='ID', how='left')

        print(f'\t{len(df_def.RAW_IMG.unique())} image paths available in database')

        # Se crea la clumna PREPROCESSED_IMG en la que se volcarán las imagenes preprocesadas
        df_def.loc[:, 'PREPROCESSED_IMG'] = df_def.apply(
            lambda x: get_path(self.procesed_dir, x.IMG_LABEL, x.IMG_TYPE,
                               f'{get_filename(x.RAW_IMG)}.{self.dest_extension}'), axis=1
        )

        # Se crea la clumna CONVERTED_IMG en la que se volcarán las imagenes convertidas de formato
        df_def.loc[:, 'CONVERTED_IMG'] = df_def.apply(
            lambda x: get_path(self.conversion_dir, x.IMG_LABEL, x.IMG_TYPE,
                                   f'{get_filename(x.RAW_IMG)}.{self.dest_extension}'), axis=1
        )

        return df_def[INFO_COLS]


class TestDataset:

    def __init__(self, path: str, image_format: str = 'jpg'):
        self.path = path
        self.image_format = image_format

    def __repr__(self):
        return f'Dataset formado por {len(self.df)} registros.'

    def get_dataset(self):
        """

        Función que genera un dataframe con las imagenes almacenadas en la carpeta definida en self.path.
        Este dataframe estará formado por tres columnas: 'item_path' con el directorio de cada imagen; 'class': no
        contendrá información pero se incluye por temas de herencia; 'dataset': contendrá el valor fijo 'Test'.

        """

        elements = []

        print('-' * 50 + f'\nLeyendo directorio de imagenes {self.path} para generar datataset')

        # Se itera sobre el directorio de imagenes obteniendo cada imagen
        for file in search_files(self.path, f'*{self.image_format}'):

            elements.append([file, np.nan, 'Test'])

        # Dataframe que almacenará el nombre de clase, número de imagenes y un ejemplo de cada categoria.
        self.df = pd.DataFrame(columns=['item_path', 'class', 'dataset'], data=elements)

        print(f'Dataset Leido correctamente - {len(self.df)} Imagenes encontradas para test')

    def get_dataset_generator(self, batch_size: int, size: tuple = (224, 224), preproces_callback: Callable = None) \
            -> DataFrameIterator:
        """

        Función que permite recuperar un dataframe iterator para test.

        :param directory: booleano que permite almacenar las imagenes generadas por el ImageDataGenerator en la carpeta
                          Data/01_procesed.
        :param batch_size: tamaño de batch con el que se crearán los iteradores.
        :param size: tamaño de la imagen que servirá de input para los iteradores. Si la imagen tiene un tamaño distinto
                     se aplicará un resize aplicando la tecnica de interpolación lanzcos. Por defecto es 224, 224.
        :param preproces_callback: función de preprocesado a aplicar a las imagenes leidas una vez aplicadas las
                                       técnicas de data augmentation.
        :return: dataframeIterator de test.
        """

        # Se comprueba que existe el dataframe en self.df
        assert self.df is not None, 'Dataframe vacío. Ejecute la función get_dataset()'

        # Se crea un datagenerator en el cual únicamente se aplicará la función de preprocesado introducida por el
        # usuario.
        datagen = ImageDataGenerator(
            preprocessing_function=preproces_callback
        )

        # Se devuelve un dataframe iterator con la misma configuración que la clase padre Dataset a excepción de que en
        # este caso, la variable class_mode es None debido a que el dataframe iterator deberá contener exclusivamente
        # información de las observaciones y no sus clases.
        return datagen.flow_from_dataframe(
            dataframe=self.df,
            x_col='item_path',
            y_col='class',
            target_size=size,
            interpolation='lanczos',
            shufle=False,
            seed=SEED,
            batch_size=batch_size,
            class_mode=None,
        )