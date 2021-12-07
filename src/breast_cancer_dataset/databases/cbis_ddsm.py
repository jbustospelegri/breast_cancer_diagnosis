import os

import numpy as np
import pandas as pd


from breast_cancer_dataset.base import GeneralDataBase
from preprocessing.image_processing import crop_image_pipeline
from utils.config import (
    CBIS_DDSM_DB_PATH, CBIS_DDSM_CONVERTED_DATA_PATH, CBIS_DDSM_PREPROCESSED_DATA_PATH, CBIS_DDSM_MASS_CASE_DESC_TRAIN,
    CBIS_DDSM_MASS_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TRAIN,
)
from utils.functions import get_dirname


class DatasetCBISDDSM(GeneralDataBase):

    __name__ = 'CBIS-DDSM'
    ID_COL: str = 'image file path'

    def __init__(self):
        super(DatasetCBISDDSM, self).__init__(
            ori_dir=CBIS_DDSM_DB_PATH, ori_extension='dcm', dest_extension='png',
            converted_dir=CBIS_DDSM_CONVERTED_DATA_PATH, procesed_dir=CBIS_DDSM_PREPROCESSED_DATA_PATH,
            database_info_file_paths=[CBIS_DDSM_MASS_CASE_DESC_TRAIN, CBIS_DDSM_MASS_CASE_DESC_TEST,
                                      CBIS_DDSM_CALC_CASE_DESC_TRAIN, CBIS_DDSM_CALC_CASE_DESC_TEST]
        )

    def get_df_from_info_files(self) -> pd.DataFrame:

        # Se crea una lista que contendrá la información de los archivos csv del set de datos
        l = []

        # Se iteran los csv con información del set de datos para unificaros
        print(f'{"-" * 70}\n\tGetting information from database {self.__name__} ({self.IMG_TYPE})\n{"-" * 70}')
        for path in self.database_info_file_paths:
            l.append(pd.read_csv(path))

        df = pd.concat(objs=l, ignore_index=True)

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGN' y 'MALIGNANT'. Se excluyen los casos de
        # patologias 'benign without callback'.
        df.loc[:, 'IMG_LABEL'] = df.pathology.where(df.pathology != 'BENIGN_WITHOUT_CALLBACK', np.nan)

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(df['abnormality type'] == 'calcification', 'CALC', 'MASS')

        # Se obtienen las imagenes full de cada tipología.
        df.drop_duplicates(subset=self.ID_COL, inplace=True)

        return df

    def add_extra_columns(self, df: pd.DataFrame):
        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        df.loc[:, 'BREAST'] = df['left or right breast']

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO
        df.loc[:, 'BREAST_VIEW'] = df['image view']

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno
        df.loc[:, 'BREAST_DENSITY'] = df.breast_density

        # Se crea la columna filename para realizar el join
        df.loc[:, 'File Name'] = df[self.ID_COL].apply(lambda x: get_dirname(x))

        # Se crea una columna identificadora
        df.loc[:, 'ID'] = df[self.ID_COL].apply(lambda x: get_dirname(x).split("/")[0])

        return df

    def process_dataframe(self, df: pd.DataFrame, f: callable = lambda x: "/".join(get_dirname(x).split(os.sep)[-3:])) \
            -> pd.DataFrame:
        """
        Función que creará un dataframe con información del dataset CBIS-DDSM. Para cada imagen, se pretende recuperar
        la tipología (clase) detectada, el path de origen de la imagen y su nombre, el tipo de patología (massa o
        calcificación) y el nombre del estudio que contiene el identificador del paciente, el seno sobre el cual se ha
        realizado la mamografía, el tipo de mamografía (CC o MLO) y un índice para aquellas imagenes recortadas o
        mascaras. Adicionalmente, se indicarán los paths de destino para la conversión y el procesado de las imagenes.

        :return: Pandas dataframe con las columnas especificadas en la descripción
        """
        return super(DatasetCBISDDSM, self).process_dataframe(df=df, get_id_func=f)


class DatasetCBISDDSMCrop(DatasetCBISDDSM):

    IMG_TYPE: str = 'CROP'
    ID_COL: str = 'cropped image file path'
    PREPROCESS_FUNC = lambda: crop_image_pipeline
    DF_COLS = [
        'ID', 'DATASET', 'BREAST', 'BREAST_VIEW', 'BREAST_DENSITY', 'ABNORMALITY_TYPE', 'IMG_TYPE', 'RAW_IMG',
        'CONVERTED_IMG', 'PREPROCESSED_IMG', 'X_MAX', 'Y_MAX', 'X_MIN', 'Y_MIN', 'IMG_LABEL'
    ]

    def add_extra_columns(self, df: pd.DataFrame):
        df = super(DatasetCBISDDSMCrop, self).add_extra_columns(df)
        for col in ['X_MAX', 'Y_MAX', 'X_MIN', 'Y_MIN']:
            df.loc[:, col] = None
        return df

    def clean_dataframe(self):
        super(DatasetCBISDDSMCrop, self).clean_dataframe()
        self.df_desc = self.df_desc.groupby('CONVERTED_IMG', as_index=False).first()

    def preproces_images(self, args: list = None, func: callable = crop_image_pipeline) -> None:
        """
        Función utilizara para realizar el preprocesado de las imagenes completas.

        :param show_example: booleano para almacenar 5 ejemplos aleatorios en la carpeta de resultados para la
        prueba realizada.
        """
        super(DatasetCBISDDSMCrop, self).preproces_images(
            args=[
                (r.CONVERTED_IMG, r.PREPROCESSED_IMG, r.X_MAX, r.Y_MAX, r.X_MIN, r.Y_MIN) for _, r in
                self.df_desc.iterrows()
            ], func=func
        )


class DatasetCBISSDDMask(DatasetCBISDDSM):

    IMG_TYPE: str = 'MASK'
    ID_COL: str = 'ROI mask file path'
    BINARY: bool = True

    def clean_dataframe(self):
        super(DatasetCBISSDDMask, self).clean_dataframe()
        self.df_desc = self.df_desc.groupby('CONVERTED_IMG', as_index=False).first()