import os
import random

import pandas as pd
import numpy as np

from multiprocessing import Pool, cpu_count
from collections import defaultdict
from tqdm import tqdm

from breast_cancer_dataset.base import GeneralDataBase
from preprocessing.image_processing import crop_image_pipeline
from utils.config import MIAS_DB_PATH, MIAS_CONVERTED_DATA_PATH, MIAS_PREPROCESSED_DATA_PATH, MIAS_CASE_DESC, IMG_SHAPE
from utils.functions import get_filename, search_files, get_path, get_patch_from_center


class DatasetMIAS(GeneralDataBase):

    __name__ = 'MIAS'

    def __init__(self):
        super(DatasetMIAS, self).__init__(
            ori_dir=MIAS_DB_PATH, ori_extension='pgm', dest_extension='png', converted_dir=MIAS_CONVERTED_DATA_PATH,
            procesed_dir=MIAS_PREPROCESSED_DATA_PATH, database_info_file_paths=[MIAS_CASE_DESC]
        )

    def get_df_from_info_files(self) -> pd.DataFrame:

        # Se obtiene la información del fichero de texto descriptivo del dataset
        l = []
        # Se iteran los csv con información del set de datos para unificaros
        print(f'{"-" * 70}\n\tGetting information from database {self.__name__} ({self.IMG_TYPE})\n{"-" * 70}')
        for path in self.database_info_file_paths:
            l.append(
                pd.read_csv(
                    path, sep=' ', skiprows=102, skipfooter=2, engine='python',
                    names=['File Name', 'BREAST_TISSUE', 'ABNORMALITY_TYPE', 'PATHOLOGY', 'X_CORD', 'Y_CORD', 'RAD']
                )
            )
        df = pd.concat(objs=l, ignore_index=True)

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGNA' y 'MALIGNA'.
        df.loc[:, 'IMG_LABEL'] = df.PATHOLOGY.map(defaultdict(lambda: None, {'B': 'BENIGN', 'M': 'MALIGNANT'}))

        # Se procesa la columna RAD estableciendo un tamaño minimo de IMG_SHAPE / 2
        df.loc[:, 'RAD'] = df.RAD.apply(lambda x: np.max([float(IMG_SHAPE / 2), float(x)]))

        # Debido a que el sistema de coordenadas se centra en el borde inferior izquierdo se deben de modificar las
        # coordenadas Y para adecuarlas al sistema de coordenadas centrado en el borde superior izquerdo.
        df.loc[:, 'Y_CORD'] = 1024 - pd.to_numeric(df.Y_CORD, downcast='float', errors='coerce')

        return df

    def add_extra_columns(self, df: pd.DataFrame):

        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        # En este caso, no se dispone de dicha información
        df.loc[:, 'BREAST'] = None

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO. No se dispone de esta
        # información
        df.loc[:, 'BREAST_VIEW'] = None

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(
            df.ABNORMALITY_TYPE.isin(['CIRC', 'SPIC', 'MISC', 'ASYM']), 'MASS', None
        )

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno. Para ello se mapean los valores:
        # - 'F' (Fatty): 1
        # - 'G' (Fatty-Glandular): 2
        # - 'D' (Dense-Glandular): 3
        df.loc[:, 'BREAST_DENSITY'] = df.BREAST_TISSUE.map(defaultdict(lambda: None, {'F': '1', 'G': '2', 'D': '3'}))

        # Se crea la columna de ID
        df.loc[:, 'ID'] = df['File Name']

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

        # Se recuperan los paths de las imagenes almacenadas con el formato específico (por defecto dcm) en la carpeta
        # de origen (por defecto INBREAST_DB_PATH)
        db_files_df = pd.DataFrame(data=search_files(self.ori_dir, self.ori_extension), columns=['RAW_IMG'])

        # Se procesa la columna ori path para poder lincar cada path con los datos del excel. Para ello, se separa
        # los nombres de cara archivo a partir del símbolo _ y se obtiene la primera posición.
        db_files_df.loc[:, 'File Name'] = db_files_df.RAW_IMG.apply(lambda x: get_filename(x))

        # Se crea la columna RAW_IMG con el path de la imagen original
        df_def = pd.merge(left=df, right=db_files_df, on='File Name', how='left')

        print(f'\t{len(df_def)} image paths available in database')

        # Se crea la clumna PREPROCESSED_IMG en la que se volcarán las imagenes preprocesadas
        df_def.loc[:, 'PREPROCESSED_IMG'] = df_def.apply(
            lambda x: get_path(self.procesed_dir, x.IMG_LABEL, x.IMG_TYPE, f'{x.ID}.{self.dest_extension}'), axis=1
        )

        # Se crea la clumna CONVERTED_IMG en la que se volcarán las imagenes convertidas de formato
        df_def.loc[:, 'CONVERTED_IMG'] = df_def.apply(
            lambda x: get_path(self.conversion_dir, x.IMG_LABEL, x.IMG_TYPE, f'{x.ID}.{self.dest_extension}'), axis=1
        )

        return df_def[self.DF_COLS]


class DatasetMIASCrop(DatasetMIAS):

    IMG_TYPE: str = 'CROP'
    DF_COLS = [
        'ID', 'DATASET', 'BREAST', 'BREAST_VIEW', 'BREAST_DENSITY', 'IMG_TYPE', 'RAW_IMG', 'CONVERTED_IMG',
        'PREPROCESSED_IMG', 'X_MAX', 'Y_MAX', 'X_MIN', 'Y_MIN', 'IMG_LABEL'
    ]

    def add_extra_columns(self, df: pd.DataFrame):
        super(DatasetMIASCrop, self).add_extra_columns(df)

        # A partir de las coordenadas y el radio, se obtienen los extremos de un cuadrado para realizar los recortes.
        get_patch_from_center(df=df)

        # Debido a que una imagen puede contener más de un recorte, se modifica la columna de ID para tener un identifi
        # cador unico
        df.loc[:, 'ID'] = df.ID + '_' + df.groupby('ID').cumcount().astype(str)

    def clean_dataframe(self):
        print(f'\tExcluding {len(self.df_desc[self.df_desc[["X_MAX", "Y_MAX", "X_MIN", "Y_MIN"]].isna().any(axis=1)])} '
              f'images without pathology localization.')
        self.df_desc.drop(
            index=self.df_desc[self.df_desc[["X_MAX", "Y_MAX", "X_MIN", "Y_MIN"]].isna().any(axis=1)].index,
            inplace=True
        )

    def preproces_images(self, show_example: bool = False) -> None:
        """
        Función utilizara para realizar el preprocesado de las imagenes completas.

        :param show_example: booleano para almacenar 5 ejemplos aleatorios en la carpeta de resultados para la
        prueba realizada.
        """

        preprocessed_imgs = pd.DataFrame(
            data=search_files(file=f'{self.conversion_dir}{os.sep}**{os.sep}{self.IMG_TYPE}',
                              ext=self.dest_extension),
            columns=['CONVERTED_IMG']
        )
        print(f'{"-" * 75}\n\tStarting preprocessing of {len(preprocessed_imgs)} images')

        args = [(row.CONVERTED_IMG, row.PREPROCESSED_IMG, row.X_MAX, row.Y_MAX, row.X_MIN, row.Y_MIN) for _, row in
                self.df_desc.iterrows()]

        with Pool(processes=cpu_count() - 2) as pool:
            results = tqdm(pool.imap(crop_image_pipeline, args), total=len(args), desc='preprocessing crop images')
            tuple(results)

        # Se recuperan las imagenes modificadas y se crea un dataframe
        proc_imgs = list(
            search_files(file=f'{self.procesed_dir}{os.sep}**{os.sep}{self.IMG_TYPE}', ext=self.dest_extension)
        )
        print(f'\tProcessed {len(proc_imgs)} images.\n{"-" * 75}')
