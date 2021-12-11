from typing import List

import pandas as pd
import numpy as np

from src.breast_cancer_dataset.base import GeneralDataBase
from src.preprocessing.image_processing import crop_image_pipeline
from src.preprocessing.mask_conversion import get_inbreast_roi_mask
from src.utils.config import (
    INBREAST_DB_PATH, INBREAST_CONVERTED_DATA_PATH, INBREAST_PREPROCESSED_DATA_PATH, INBREAST_CASE_DESC,
    CROP_CONFIG, CROP_PARAMS
)
from src.utils.functions import get_filename, get_path


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

        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        df.loc[:, 'BREAST'] = np.where(df.Laterality == 'R', 'RIGHT', 'LEFT')

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO
        df.loc[:, 'BREAST_VIEW'] = df.View

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno
        df.loc[:, 'BREAST_DENSITY'] = df.ACR

        # Se crea la columna de ID
        df.loc[:, 'ID'] = df['File Name'].astype(str)

        # Se crea la columna FILE_NAME
        df.loc[:, 'FILE_NAME'] = df['File Name'].astype(str)

        return df

    def get_raw_files(self, df: pd.DataFrame, f: callable = lambda x: int(get_filename(x).split('_')[0])) \
            -> pd.DataFrame:
        return super(DatasetINBreast, self).get_raw_files(df=df, get_id_func=f)

    def get_image_mask(self, func: callable = get_inbreast_roi_mask, args: List = None):

        args = list(set([(x.CONVERTED_IMG, x.FILE_NAME, x.CONVERTED_MASK) for _, x in self.df_desc.iterrows()]))
        super(DatasetINBreast, self).get_image_mask(func=func, args=args)


class DatasetINBreastCrop(DatasetINBreast):

    IMG_TYPE: str = get_path('CROP', CROP_CONFIG)

    def preproces_images(self, args: list = None, func: callable = crop_image_pipeline) -> None:
        """
        Función utilizara para realizar el preprocesado de las imagenes completas.

        :param show_example: booleano para almacenar 5 ejemplos aleatorios en la carpeta de resultados para la
        prueba realizada.
        """

        p = CROP_PARAMS[CROP_CONFIG]
        args = list(set([(
            x.CONVERTED_IMG, x.PROCESSED_IMG, x.CONVERTED_MASK, p['N_BACKGROUND'], p['N_ROI'], p['OVERLAP'], p['MARGIN'])
            for _, x in self.df_desc.iterrows()
        ]))

        super(DatasetINBreastCrop, self).preproces_images(args=args, func=func)
        super(DatasetINBreastCrop, self).get_roi_imgs()

    def delete_observations(self):
        pass
