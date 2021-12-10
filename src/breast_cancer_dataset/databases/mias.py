from typing import io, List

import cv2
import pandas as pd
import numpy as np

from collections import defaultdict

from breast_cancer_dataset.base import GeneralDataBase
from preprocessing.image_processing import crop_image_pipeline, full_image_pipeline, get_mias_roi_mask
from utils.config import MIAS_DB_PATH, MIAS_CONVERTED_DATA_PATH, MIAS_PREPROCESSED_DATA_PATH, MIAS_CASE_DESC, PATCH_SIZE
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
                    names=['FILE_NAME', 'BREAST_TISSUE', 'ABNORMALITY_TYPE', 'PATHOLOGY', 'X_CORD', 'Y_CORD', 'RAD']
                )
            )
        df = pd.concat(objs=l, ignore_index=True)

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGNA' y 'MALIGNA'.
        df.loc[:, 'IMG_LABEL'] = df.PATHOLOGY.map(defaultdict(lambda: None, {'B': 'BENIGN', 'M': 'MALIGNANT'}))

        # Se corrige el formato del círculo
        df.loc[:, 'ORI_RAD'] = pd.to_numeric(df.RAD, downcast='integer', errors='coerce')

        # Se procesa la columna RAD estableciendo un tamaño minimo de IMG_SHAPE / 2
        df.loc[:, 'RAD'] = df.RAD.apply(lambda x: np.max([float(PATCH_SIZE / 2), float(x)]))

        # Debido a que el sistema de coordenadas se centra en el borde inferior izquierdo se deben de modificar las
        # coordenadas Y para adecuarlas al sistema de coordenadas centrado en el borde superior izquerdo.
        df.loc[:, 'Y_CORD'] = 1024 - pd.to_numeric(df.Y_CORD, downcast='integer', errors='coerce')

        # Se corrige el formato de las coordenadas
        df.loc[:, 'X_CORD'] = pd.to_numeric(df.X_CORD, downcast='integer', errors='coerce')

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(
            df.ABNORMALITY_TYPE.isin(['CIRC', 'SPIC', 'MISC', 'ASYM']), 'MASS', None
        )

        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        # En este caso, no se dispone de dicha información
        df.loc[:, 'BREAST'] = None

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO. No se dispone de esta
        # información
        df.loc[:, 'BREAST_VIEW'] = None

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno. Para ello se mapean los valores:
        # - 'F' (Fatty): 1
        # - 'G' (Fatty-Glandular): 2
        # - 'D' (Dense-Glandular): 3
        df.loc[:, 'BREAST_DENSITY'] = df.BREAST_TISSUE.map(defaultdict(lambda: None, {'F': '1', 'G': '2', 'D': '3'}))

        # Se crea la columna de ID
        df.loc[:, 'ID'] = df['FILE_NAME']

        return df


class DatasetMIASCrop(DatasetMIAS):

    IMG_TYPE: str = 'CROP'
    DF_COLS = [
        'ID', 'DATASET', 'BREAST', 'BREAST_VIEW', 'BREAST_DENSITY', 'IMG_TYPE', 'FILE_NAME', 'RAW_IMG', 'CONVERTED_IMG',
        'PREPROCESSED_IMG', 'X_MAX', 'Y_MAX', 'X_MIN', 'Y_MIN', 'IMG_LABEL'
    ]

    def add_extra_columns(self, df: pd.DataFrame):

        # A partir de las coordenadas y el radio, se obtienen los extremos de un cuadrado para realizar los recortes.
        get_patch_from_center(df=df)

        # Debido a que una imagen puede contener más de un recorte, se modifica la columna de ID para tener un identifi
        # cador unico
        df.loc[:, 'ID'] = df.ID + '_' + df.groupby('ID').cumcount().astype(str)

        return df

    def clean_dataframe(self):
        print(f'\tExcluding {len(self.df_desc[self.df_desc[["X_MAX", "Y_MAX", "X_MIN", "Y_MIN"]].isna().any(axis=1)])} '
              f'images without pathology localization.')
        self.df_desc.drop(
            index=self.df_desc[self.df_desc[["X_MAX", "Y_MAX", "X_MIN", "Y_MIN"]].isna().any(axis=1)].index,
            inplace=True
        )

    def preproces_images(self, args: list = None, func: callable = crop_image_pipeline) -> None:
        """
        Función utilizara para realizar el preprocesado de las imagenes completas.

        :param show_example: booleano para almacenar 5 ejemplos aleatorios en la carpeta de resultados para la
        prueba realizada.
        """
        super(DatasetMIASCrop, self).preproces_images(
            args=[
                (r.CONVERTED_IMG, r.PREPROCESSED_IMG, r.X_MAX, r.Y_MAX, r.X_MIN, r.Y_MIN) for _, r in
                self.df_desc.iterrows()
            ], func=func
        )


class DatasetMIASMask(DatasetMIAS):

    DF_COLS = [
        'ID', 'DATASET', 'BREAST', 'BREAST_VIEW', 'BREAST_DENSITY', 'IMG_TYPE', 'FILE_NAME', 'RAW_IMG', 'CONVERTED_IMG',
        'PREPROCESSED_IMG', 'X_CORD', 'Y_CORD', 'ORI_RAD', 'IMG_LABEL'
    ]

    def process_dataframe(self, df: pd.DataFrame, get_id_func: callable = lambda x: get_filename(x)) -> pd.DataFrame:
        df = super(DatasetMIAS, self).process_dataframe(df=df, get_id_func=get_id_func)
        df.loc[:, 'ID'] = df.ID + '_' + df.groupby(['ID', 'IMG_LABEL']).cumcount().astype(str)
        df.loc[:, 'CONVERTED_MASK_IMG'] = df.apply(
            lambda x: get_path(self.conversion_dir, x.IMG_LABEL, 'MASK', f'{x.ID}.png'), axis=1
        )
        df.loc[:, 'MASK_IMG'] = df.apply(
            lambda x: get_path(self.procesed_dir, x.IMG_LABEL, 'MASK', f'{x.FILE_NAME}.png'), axis=1)

        return df

    def convert_images_format(self, func: callable = get_mias_roi_mask, args: List = None, txt: str = '') -> None:
        super(DatasetMIAS, self).convert_images_format()

        # Crear las mascaras de directo
        args = list(set([
            (x.CONVERTED_MASK_IMG, x.X_CORD, x.Y_CORD, x.ORI_RAD) for _, x in
            self.df_desc[~self.df_desc[['X_CORD', 'Y_CORD']].isnull().any(axis=1)].iterrows()
        ]))
        super(DatasetMIAS, self).convert_images_format(func=func, args=args)

    def preproces_images(self, args: list = None, func: callable = full_image_pipeline) -> None:

        args = [
            (x.CONVERTED_IMG, x.PREPROCESSED_IMG, False, x.MASK_IMG, x.CONVERTED_MASK_IMG) for _, x in
            self.df_desc.groupby(['CONVERTED_IMG', 'PREPROCESSED_IMG', 'MASK_IMG'], as_index=False).\
                agg({'CONVERTED_MASK_IMG': lambda x: x.tolist()}).iterrows()
        ]
        super(DatasetMIASMask, self).preproces_images(args=args, func=func)
