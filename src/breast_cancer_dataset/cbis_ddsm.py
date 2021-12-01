import os
import numpy as np
import pandas as pd

from breast_cancer_dataset.base import GeneralDataBase
from utils.config import (
    CBIS_DDSM_DB_PATH, CBIS_DDSM_CONVERTED_DATA_PATH, CBIS_DDSM_PREPROCESSED_DATA_PATH, CBIS_DDSM_MASS_CASE_DESC_TRAIN,
    CBIS_DDSM_MASS_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TRAIN,
)
from utils.functions import get_dirname, search_files, get_path


class DatasetCBISDDSM(GeneralDataBase):

    __name__ = 'CBIS-DDSM'
    IMG_TYPE: str = 'FULL'
    ID_COL: str = 'image file path'
    BINARY: bool = False

    def __init__(self):
        super().__init__(
            ori_dir=CBIS_DDSM_DB_PATH, ori_extension='dcm', dest_extension='png',
            converted_dir=CBIS_DDSM_CONVERTED_DATA_PATH, procesed_dir=CBIS_DDSM_PREPROCESSED_DATA_PATH,
            database_info_file_paths=[CBIS_DDSM_MASS_CASE_DESC_TRAIN, CBIS_DDSM_MASS_CASE_DESC_TEST,
                                      CBIS_DDSM_CALC_CASE_DESC_TRAIN, CBIS_DDSM_CALC_CASE_DESC_TEST]
        )

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Función que creará un dataframe con información del dataset CBIS-DDSM. Para cada imagen, se pretende recuperar
        la tipología (clase) detectada, el path de origen de la imagen y su nombre, el tipo de patología (massa o
        calcificación) y el nombre del estudio que contiene el identificador del paciente, el seno sobre el cual se ha
        realizado la mamografía, el tipo de mamografía (CC o MLO) y un índice para aquellas imagenes recortadas o
        mascaras. Adicionalmente, se indicarán los paths de destino para la conversión y el procesado de las imagenes.

        :return: Pandas dataframe con las columnas especificadas en la descripción
        """

        # Se crea la columna ID para poder linkar la información del excel con la información de las imagenes
        # almacenadas en la carpeta RAW. En este caso, se utilizará la jerarquía de carpetas en la que se almacena
        # cada imagen
        df.loc[:, 'ID'] = df[self.ID_COL].apply(lambda x: get_dirname(x))

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

        return df_def[self.DF_COLS]

    def get_df_from_info_files(self) -> pd.DataFrame:

        # Se crea una lista que contendrá la información de los archivos csv del set de datos
        l = []

        # Se iteran los csv con información del set de datos para unificaros
        print(f'{"-" * 70}\n\tGetting information from database {self.__name__} ({self.IMG_TYPE})\n{"-" * 70}')
        for path in self.database_info_file_paths:
            l.append(pd.read_csv(path))

        df = pd.concat(objs=l, ignore_index=True)

        # Se descartan aquellas imagenes que pertenecen a la clase Benign without callback debido a que tal vez no
        # contienen ningúna pat
        print(f'\tExcluding {len(df[df.pathology == "BENIGN_WITHOUT_CALLBACK"])} Benign without callback cases')
        df.drop(index=df[df.pathology == "BENIGN_WITHOUT_CALLBACK"].index, inplace=True)

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGN' y 'MALIGNANT'. Se excluyen los casos de
        # patologias 'benign without callback'.
        df.loc[:, 'IMG_LABEL'] = df.pathology

        # Se obtienen las imagenes full de cada tipología.
        df.drop_duplicates(subset=self.ID_COL, inplace=True)

        return df

    @staticmethod
    def add_extra_columns(df: pd.DataFrame):
        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        df.loc[:, 'BREAST'] = df['left or right breast']

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO
        df.loc[:, 'BREAST_VIEW'] = df['image view']

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(df['abnormality type'] == 'calcification', 'CALC', 'MASS')

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno
        df.loc[:, 'BREAST_DENSITY'] = df.breast_density


class DatasetCBISDDSMCrop(DatasetCBISDDSM):

    IMG_TYPE: str = 'CROP'
    ID_COL: str = 'cropped image file path'
    BINARY: bool = False

    def clean_dataframe(self):
        super().clean_dataframe()
        self.df_desc = self.df_desc.groupby('CONVERTED_IMG', as_index=False).first()


class DatasetCBISSDDMMask(DatasetCBISDDSM):

    IMG_TYPE: str = 'MASK'
    ID_COL: str = 'mask image file path'
    BINARY: bool = True

    def clean_dataframe(self):
        super().clean_dataframe()
        self.df_desc = self.df_desc.groupby('CONVERTED_IMG', as_index=False).first()