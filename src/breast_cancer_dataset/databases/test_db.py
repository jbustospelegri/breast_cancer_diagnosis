import os
import cv2

import pandas as pd

from typing import io
from tensorflow import cast, float32
from albumentations import Compose, Lambda

from breast_cancer_dataset.base import ClassificationDataset, Dataloder
from preprocessing.image_conversion import convert_img
from preprocessing.image_processing import crop_image_pipeline, resize_img
from preprocessing.mask_generator import get_test_mask
from user_interface.utils import ControledError
from utils.config import TEST_CONVERTED_DATA_PATH, TEST_PREPROCESSED_DATA_PATH, CROP_CONFIG, CROP_PARAMS
from utils.functions import get_path, search_files, get_filename
from user_interface.signals_interface import SignalProgressBar, SignalLogging


class DatasetTest:

    __name__ = 'TEST_SET'
    XLSX_COLS = ['ID', 'FILE_PATH', 'X_CORD', 'Y_CORD', 'RAD']
    CONVERSION_PATH = TEST_CONVERTED_DATA_PATH
    PROCESED_PATH = TEST_PREPROCESSED_DATA_PATH
    PROCESSING_PARAMS = CROP_PARAMS[CROP_CONFIG]

    def __init__(self, xlsx_io: io, signal: SignalLogging, out_path: io):

        self.signal_log = signal
        self.out_path = out_path
        self.df = self.get_df_from_info_files(path=xlsx_io)

    def get_df_from_info_files(self, path) -> pd.DataFrame:

        # Lectura del excel con la información del dataset
        df = pd.read_excel(path, dtype=str)

        # Se comprueban que las columnas son correctas
        if not all([c in df.columns for c in self.XLSX_COLS]):
            raise ControledError(f'Incorrect column names. Please check excel contains next '
                                 f'column values: {", ".join(self.XLSX_COLS)}')

        self.XLSX_COLS = df.columns.values.tolist()

        # Se comprueba que el id es único
        if any(df.ID.value_counts() > 1):
            raise ControledError(f'Duplicated id values found in excel')

        # Se realiza la conversión de datos para las columnas de análisis.
        for col in ['X_CORD', 'Y_CORD', 'RAD']:
            df.loc[:, col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')

        # Aquellos datos con posicionamiento invalido son eliminadas
        incorrect = df[df[['X_CORD', 'Y_CORD', 'RAD']].isna().any(axis=1)]
        if len(incorrect) > 0:
            self.signal_log.log(
                f'Incorrect data found!\nDeleted {len(incorrect)} incorrect values found in X_CORD, Y_CORD and RAD.' + \
                'Deleted files are:\n\t- File:' + "\n\t- File:".join(incorrect.ID.values.tolist())
            )
        df.drop(index=incorrect.index, inplace=True)
        
        # Aquellos datos con posicionamiento invalido son eliminadas
        incorrect = df[~ df.FILE_PATH.apply(lambda x: os.path.isfile(x))]
        if len(incorrect) > 0:
            self.signal_log.log(
                f'Incorrect data found!\n{len(incorrect)} FILE_PATHS does not exists.' + \
                'Deleted files are:\n\t- File:' + "\n\t- File:".join(incorrect.ID.values.tolist())
            )
        df.drop(index=incorrect.index, inplace=True)

        return df

    def convert_images_format(self, signal: SignalProgressBar, min_value: int, max_value: int):

        bar_range = max_value - min_value

        # Se crea la columna CONVERTED_IMG en la que se volcarán las imagenes convertidas de formato
        self.df.loc[:, 'CONVERTED_IMG'] = self.df.apply(lambda x: get_path(self.CONVERSION_PATH, 'FULL', f'{x.ID}.png'),
                                                        axis=1)

        for i, arg in enumerate(list(set(
                [(row.FILE_PATH, row.CONVERTED_IMG, False, self.out_path) for _, row in self.df.iterrows()])), 1):
            txt = f'Converting image {i} of {len(self.df)}'
            signal.emit_update_label_and_progress_bar(min_value + round(bar_range * i / len(self.df)), txt)
            convert_img(arg)

    def preprocess_image(self, signal: SignalProgressBar, min_value: int, max_value: int):

        self.df.loc[:, 'CONVERTED_MASK'] = self.df.apply(
            lambda x: get_path(self.CONVERSION_PATH, 'MASK', f'{x.ID}.png'), axis=1
        )

        # Se crea la clumna PROCESSED_IMG en la que se volcarán las imagenes preprocesadas
        self.df.loc[:, 'PROCESSED_IMG'] = self.df.apply(lambda x: get_path(self.PROCESED_PATH, f'{x.ID}.png'), axis=1)

        self.get_image_mask(signal, min_value, (min_value + max_value) // 2)

        bar_range = max_value - (min_value + max_value) // 2
        for i, arg in enumerate(list(set([(
                x.CONVERTED_IMG, x.PROCESSED_IMG, x.CONVERTED_MASK, self.PROCESSING_PARAMS['N_BACKGROUND'],
                self.PROCESSING_PARAMS['N_ROI'], self.PROCESSING_PARAMS['OVERLAP'], self.PROCESSING_PARAMS['MARGIN'],
                self.out_path
        )
            for _, x in self.df.iterrows()
        ])), 1):
            txt = f'Preprocessing image {i} of {len(self.df)}'
            signal.emit_update_label_and_progress_bar(
                (min_value + max_value) // 2 + round(bar_range * i / len(self.df)), txt)
            crop_image_pipeline(arg)

        croped_imgs = pd.DataFrame(data=search_files(self.PROCESED_PATH, ext='png', in_subdirs=False), columns=['FILE'])

        croped_imgs.loc[:, 'ID'] = croped_imgs.FILE.apply(lambda x: "_".join(get_filename(x).split('_')[1:-1]))

        self.df = pd.merge(left=self.df, right=croped_imgs, on=['ID'], how='left')

        # Se suprimen los casos que no contienen ningún recorte
        incorrect = self.df[self.df.FILE.isnull()]
        print(f'\tDeleting {len(incorrect)} samples without cropped regions')
        if len(incorrect) > 0:
            self.signal_log.log(
                f'Warning! Deleted {len(incorrect)} without ROI generated images. Deleted files are:\n\t- File:' +
                "\n\t- File:".join(incorrect.ID.values.tolist()))

        self.df.drop(index=self.df[self.df.FILE.isnull()].index, inplace=True)

        self.df.loc[:, 'PROCESSED_IMG'] = self.df.FILE

    def get_image_mask(self, signal: SignalProgressBar, min_value: int, max_value: int):

        bar_range = max_value - min_value
        for i, arg in enumerate([(x.CONVERTED_IMG, x.CONVERTED_MASK, x.X_CORD, x.Y_CORD, x.RAD, self.out_path)
                                 for _, x in self.df.iterrows()], 1):
            txt = f'Cropping image {i} of {len(self.df)}'
            signal.emit_update_label_and_progress_bar(min_value + round(bar_range * i / len(self.df)), txt)
            get_test_mask(arg)

    def get_iterator(self, callback, size: tuple):

        transformtions = Compose([
            Lambda(
                image=lambda x, **kgs: resize_img(x, height=size[0], width=size[1], interpolation=cv2.INTER_LANCZOS4),
                name='image resizing'
            ),
            Lambda(image=lambda x, **kwargs: cast(x, float32), name='floating point conversion'),
            Lambda(image=callback, name='cnn processing function')
        ])

        return Dataloder(ClassificationDataset(self.df, 'PROCESSED_IMG', None, transformtions), batch_size=1)
