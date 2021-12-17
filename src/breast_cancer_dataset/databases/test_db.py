import pandas as pd
import numpy as np

from typing import List, io
from collections import defaultdict

from preprocessing.image_conversion import convert_img
from preprocessing.image_processing import crop_image_pipeline
from preprocessing.mask_generator import get_test_mask
from utils.config import TEST_CONVERTED_DATA_PATH, TEST_PREPROCESSED_DATA_PATH, CROP_CONFIG, CROP_PARAMS
from utils.functions import get_path, search_files, get_filename


class DatasetTest():

    __name__ = 'TEST_SET'
    XLSX_COLS = ['ID', 'FILE_PATH', 'X_CORD', 'Y_CORD', 'RAD']
    CONVERSION_PATH = TEST_CONVERTED_DATA_PATH
    PROCESED_PATH = TEST_PREPROCESSED_DATA_PATH
    PROCESSING_PARAMS = CROP_PARAMS[CROP_CONFIG]

    def __init__(self, xlsx_io: io):

        self.df = self.get_df_from_info_files(path=xlsx_io)

    def get_df_from_info_files(self, path) -> pd.DataFrame:

        # Se iteran los csv con información del set de datos para unificaros
        print(f'{"=" * 70}\n\tGetting information from database {self.__name__}\n{"=" * 70}')

        # Lectura del excel con la información del dataset
        df_pre = pd.read_excel(path)

        # Se comprueban que las columnas son correctas
        if not all([c in self.XLSX_COLS for c in df_pre.columns]):
            raise ValueError(
                f'Incorrect column names. Please check excel contains next column values: {", ".join(self.XLSX_COLS)}'
            )

        # Se crea el dataset con las columnas deseadas
        df = df_pre[self.XLSX_COLS].copy()

        # Se comprueba que el id es único
        if any(df.ID.value_counts() > 1):
            raise ValueError(f'Duplicated id values found in excel')

        # Se realiza la conversión de datos para las columnas de análisis.
        for col in ['X_CORD', 'Y_CORD', 'RAD']:
            df.loc[:, col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')

        # Aquellos datos con posicionamiento invalido son eliminadas
        incorrect = df[df[['X_CORD', 'Y_CORD', 'RAD']].isna().any(axis=1)]
        assert len(incorrect) == 0, f'Deleting {len(incorrect)} incorrect values found in X_CORD, Y_CORD and RAD.'
        df.drop(index=incorrect.index, inplace=True)

        # Se crea la clumna CONVERTED_IMG en la que se volcarán las imagenes convertidas de formato
        df.loc[:, 'CONVERTED_IMG'] = df.apply(lambda x: get_path(self.CONVERSION_PATH, 'FULL', f'{x.ID}.png'), axis=1)

        df.loc[:, 'CONVERTED_MASK'] = df.apply(lambda x: get_path(self.CONVERSION_PATH, 'MASK', f'{x.ID}.png'), axis=1)

        # Se crea la clumna PROCESSED_IMG en la que se volcarán las imagenes preprocesadas
        df.loc[:, 'PROCESSED_IMG'] = df.apply(lambda x: get_path(self.PROCESED_PATH, f'{x.ID}.png'), axis=1)

        return df

    def convert_images_format(self):
        for arg in list(set([(row.FILE_NAME, row.CONVERTED_IMG) for _, row in self.df.iterrows()])):
            convert_img(arg)

    def preprocess_image(self):

        self.get_image_mask()

        for arg in list(set([(
                x.CONVERTED_IMG, x.PROCESSED_IMG, x.CONVERTED_MASK, self.PROCESSING_PARAMS['N_BACKGROUND'],
                self.PROCESSING_PARAMS['N_ROI'], self.PROCESSING_PARAMS['OVERLAP'], self.PROCESSING_PARAMS['MARGIN'])
            for _, x in self.df.iterrows()
        ])):
            crop_image_pipeline(arg)

    def get_image_mask(self):

        for arg in [(x.CONVERTED_MASK, x.X_CORD, x.Y_CORD, x.RAD) for _, x in self.df.iterrows()]:
            get_test_mask(arg)

        croped_imgs = pd.DataFrame(data=search_files(self.PROCESED_PATH, ext='png', in_subdirs=False), columns=['FILE'])

        croped_imgs.loc[:, 'FILE_NAME'] = croped_imgs.FILE.apply(lambda x: "_".join(get_filename(x).split('_')[1:-1]))

        self.df = pd.merge(left=self.df, right=croped_imgs, on=['FILE_NAME'], how='left')

        # Se suprimen los casos que no contienen ningún recorte
        print(f'\tDeleting {len(self.df[self.df.FILE.isnull()])} samples without cropped regions')
        self.df.drop(index=self.df[self.df.FILE.isnull()].index, inplace=True)
