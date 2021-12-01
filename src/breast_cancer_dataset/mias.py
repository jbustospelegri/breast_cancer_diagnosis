import pandas as pd
import numpy as np

from collections import defaultdict

from breast_cancer_dataset.base import GeneralDataBase
from preprocessing.image_processing import crop_image_pipeline
from utils.config import MIAS_DB_PATH, MIAS_CONVERTED_DATA_PATH, MIAS_PREPROCESSED_DATA_PATH, MIAS_CASE_DESC, DF_COLS
from utils.functions import get_filename, search_files, get_path


class DatasetMIAS(GeneralDataBase):

    __name__ = 'MIAS'

    def __init__(self):
        super().__init__(
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
                    names=['ID', 'BREAST_TISSUE', 'ABNORMALITY_TYPE', 'PATHOLOGY', 'X_CORD', 'Y_CORD', 'RAD']
                )
            )
        df = pd.concat(objs=l, ignore_index=True)

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGNA' y 'MALIGNA'.
        df.loc[:, 'IMG_LABEL'] = df.PATHOLOGY.map(defaultdict(lambda: None, {'B': 'BENIGN', 'M': 'MALIGNANT'}))

        # Se suprimen los casos que no contienen ninguna patología
        print(f'\tExcluding {len(df[df.IMG_LABEL.isnull()].index.drop_duplicates())} samples without pathologies.')
        df.drop(index=df[df.IMG_LABEL.isnull()].index, inplace=True)

        self.add_extra_columns(df)

        return df

    @staticmethod
    def add_extra_columns(df: pd.DataFrame):

        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        # En este caso, no se dispone de dicha información
        df.loc[:, 'BREAST'] = None

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO. No se dispone de esta
        # información
        df.loc[:, 'BREAST_VIEW'] = None

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(df.ABNORMALITY_TYPE == 'CALC', 'CALC', 'MASS')

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno. Para ello se mapean los valores:
        # - 'F' (Fatty): 1
        # - 'G' (Fatty-Glandular): 2
        # - 'D' (Dense-Glandular): 3
        df.loc[:, 'BREAST_DENSITY'] = df.BREAST_TISSUE.map(defaultdict(lambda: None, {'F': '1', 'G': '2', 'D': '3'}))

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

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

        return df_def[DF_COLS]


class DatasetMIASCrop(DatasetMIAS):

    IMG_TYPE: str = 'CROP'
    BINARY: bool = False
    PREPROCESS_FUNC = crop_image_pipeline

    @staticmethod
    def add_extra_columns(df: pd.DataFrame):
        super().add_extra_columns(df)
        df.loc[:, 'XY_MAX'] = (0, 0)
        df.loc[:, 'XY_MIN'] = (0, 0)

    def clean_dataframe(self):
        super().clean_dataframe()
        print(f'\tExcluding {len(self.df_desc[self.df_desc.XY_MAX.isnull()])} images without pathology localization.')
        self.df_desc.drop(index=self.df_desc[self.df_desc.XY_MAX.isnull()])
        self.df_desc = self.df_desc.groupby('CONVERTED_IMG', as_index=False).first()
