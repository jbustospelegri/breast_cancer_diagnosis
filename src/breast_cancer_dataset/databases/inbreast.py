import os
from typing import io

import pandas as pd
import numpy as np
import plistlib

from skimage.draw import polygon
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from src.breast_cancer_dataset.base import GeneralDataBase
from src.preprocessing.image_processing import crop_image_pipeline
from src.utils.config import (
    INBREAST_DB_PATH, INBREAST_CONVERTED_DATA_PATH, INBREAST_PREPROCESSED_DATA_PATH, INBREAST_CASE_DESC,
    INBREAST_DB_XML_ROI_PATH, PATCH_SIZE
)
from src.utils.functions import search_files, get_filename, get_path, get_patch_from_center


class DatasetINBreast(GeneralDataBase):

    __name__ = 'INBreast'

    def __init__(self):
        super().__init__(
            ori_dir=INBREAST_DB_PATH, ori_extension='dcm', dest_extension='png',
            converted_dir=INBREAST_CONVERTED_DATA_PATH, procesed_dir=INBREAST_PREPROCESSED_DATA_PATH,
            database_info_file_paths=[INBREAST_CASE_DESC]
        )

    def get_df_from_info_files(self) -> pd.DataFrame:
        # Se crea una lista que contendrá la información de los archivos csv del set de datos
        l = []

        # Se iteran los csv con información del set de datos para unificaros
        print(f'{"-" * 70}\n\tGetting information from database {self.__name__} ({self.IMG_TYPE})\n{"-" * 70}')
        for path in self.database_info_file_paths:
            l.append(pd.read_excel(path, skipfooter=2))
        df = pd.concat(objs=l, ignore_index=True)

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGNA' y 'MALIGNA' en función de la puntición
        # asignada por BIRADS. Los codigos 2 se asignan a la clase benigna; los 4b, 4c, 5 y 6 a la clase maligna.
        # Se excluyen los códigos 0 (Incompletos), 3 (Probablemente benigno), 4a (Probablemente maligno (2-9%).
        # noinspection PyTypeChecker
        df.loc[:, 'IMG_LABEL'] = np.where(
            df['Bi-Rads'].astype(str).isin(['2']), 'BENIGN',
            np.where(df['Bi-Rads'].astype(str).isin(['4b', '4c', '5', '6']), 'MALIGNANT', None)
        )

        # Se exluyen aquellos casos en los que no haya una patología de masa
        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(df['Mass '] == 'X', 'MASS', None)

        return df

    def add_extra_columns(self, df: pd.DataFrame):
        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        df.loc[:, 'BREAST'] = np.where(df.Laterality == 'R', 'RIGHT', 'LEFT')

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO
        df.loc[:, 'BREAST_VIEW'] = df.View

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno
        df.loc[:, 'BREAST_DENSITY'] = df.ACR

        # Se crea la columna de ID
        df.loc[:, 'ID'] = df['File Name'].astype(str)

        return df

    def process_dataframe(self, df: pd.DataFrame, f: callable = lambda x: int(get_filename(x).split('_')[0])) \
            -> pd.DataFrame:
        return super(DatasetINBreast, self).process_dataframe(df=df, get_id_func=f)


class DatasetINBreastCrop(DatasetINBreast):

    IMG_TYPE: str = 'CROP'
    DF_COLS = [
        'ID', 'DATASET', 'BREAST', 'BREAST_VIEW', 'BREAST_DENSITY', 'ABNORMALITY_TYPE', 'IMG_TYPE', 'RAW_IMG',
        'CONVERTED_IMG', 'PREPROCESSED_IMG', 'X_MAX', 'Y_MAX', 'X_MIN', 'Y_MIN', 'IMG_LABEL'
    ]

    def add_extra_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(DatasetINBreastCrop, self).add_extra_columns(df)
        df = pd.merge(left=df, right=self.get_inbreast_roi(), on='File Name', how='left')

        # Debido a que una imagen puede contener más de un recorte, se modifica la columna de ID para tener un identifi
        # cador unico
        df.loc[:, 'ID'] = df.ID + '_' + df.groupby('ID').cumcount().astype(str)

        return df

    def preproces_images(self, args: list = None, func: callable = crop_image_pipeline) -> None:
        """
        Función utilizara para realizar el preprocesado de las imagenes completas.

        :param show_example: booleano para almacenar 5 ejemplos aleatorios en la carpeta de resultados para la
        prueba realizada.
        """
        super(DatasetINBreastCrop, self).preproces_images(
            args=[
                (r.CONVERTED_IMG, r.PREPROCESSED_IMG, r.X_MAX, r.Y_MAX, r.X_MIN, r.Y_MIN) for _, r in
                self.df_desc.iterrows()
            ], func=func
        )

    def clean_dataframe(self):
        print(f'\tExcluding {len(self.df_desc[self.df_desc[["X_MAX", "Y_MAX", "X_MIN", "Y_MIN"]].isna().any(axis=1)])} '
              f'images without pathology localization.')
        self.df_desc.drop(
            index=self.df_desc[self.df_desc[["X_MAX", "Y_MAX", "X_MIN", "Y_MIN"]].isna().any(axis=1)].index,
            inplace=True
        )

    @staticmethod
    def get_inbreast_roi():
        """
        This function loads a osirix xml region as a binary numpy array for INBREAST
        dataset
        @mask_path : Path to the xml file
        @imshape : The shape of the image as an array e.g. [4084, 3328]
        return: numpy array where positions in the roi are assigned a value of 1.
        """

        def load_point(point_string):
            x, y = tuple([int(round(float(num), 0)) for num in point_string.strip('()').split(',')])
            return y, x

        l = []
        for path in search_files(file=INBREAST_DB_XML_ROI_PATH, ext='xml', in_subdirs=False):
            plist_dict = plistlib.load(open(path, 'rb'), fmt=plistlib.FMT_XML)['Images'][0]
            for roi in plist_dict['ROIs']:
                if roi['Name'] in ['Mass']:
                    p = pd.DataFrame(data=[load_point(point) for point in roi['Point_px']], columns=['Y', 'X'])
                    l.append(pd.DataFrame(
                        data=[[int(get_filename(path)), p.X.max(), p.X.min(), p.Y.max(), p.Y.min()]],
                        columns=['File Name', 'X_MAX', 'X_MIN', 'Y_MAX', 'Y_MIN'])
                    )

        df = pd.concat(l, ignore_index=True)

        df.loc[:, 'RAD'] = df.apply(
            lambda x: round(max([(x.X_MAX - x.X_MIN), (x.Y_MAX - x.Y_MIN), PATCH_SIZE]) / 2), axis=1
        )

        for axis in ['X', 'Y']:
            df.loc[:, f'{axis}_CORD'] = round(df[[f'{axis}_MAX', f'{axis}_MIN']].sum(axis=1) / 2)

        get_patch_from_center(df=df)

        return df[['File Name', 'X_MAX', 'Y_MAX', 'X_MIN', 'Y_MIN']]


class DatasetINBreastSegmentation(DatasetINBreast):

    IMG_TYPE: str = 'Segmentation'
    DF_COLS = [
        'ID', 'DATASET', 'BREAST', 'BREAST_VIEW', 'BREAST_DENSITY', 'ABNORMALITY_TYPE', 'IMG_TYPE', 'RAW_IMG',
        'CONVERTED_IMG', 'PREPROCESSED_IMG', 'IMG_LABEL'
    ]

    def add_extra_columns(self, df: pd.DataFrame):
        super(DatasetINBreast, self).add_extra_columns(df)
        for col in ['X_MAX', 'Y_MAX', 'X_MIN', 'Y_MIN']:
            df.loc[:, col] = None

    def clean_dataframe(self):
        super(DatasetINBreast, self).clean_dataframe()
        self.df_desc = self.df_desc.groupby('CONVERTED_IMG', as_index=False).first()

    @staticmethod
    def load_inbreast_mask(mask_path: io, imshape: tuple):
        """
        This function loads a osirix xml region as a binary numpy array for INBREAST
        dataset
        @mask_path : Path to the xml file
        @imshape : The shape of the image as an array e.g. [4084, 3328]
        return: numpy array where positions in the roi are assigned a value of 1.
        """

        def load_point(point_string):
            x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
            return y, x

        mask = np.zeros(imshape)
        with open(mask_path, 'rb') as mask_file:
            plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
            assert plist_dict['NumberOfROIs'] == len(plist_dict['ROIs'])
            for roi in plist_dict['ROIs']:
                assert roi['NumberOfPoints'] == len(roi['Point_px'])
                points = [load_point(point) for point in roi['Point_px']]
                if len(points) <= 2:
                    for point in points:
                        mask[int(point[0]), int(point[1])] = 1
                else:
                    x, y = zip(*points)
                    x, y = np.array(x), np.array(y)
                    poly_x, poly_y = polygon(x, y, shape=imshape)
                    mask[poly_x, poly_y] = 1
        return mask
