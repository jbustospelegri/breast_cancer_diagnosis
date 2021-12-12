from typing import List

import pandas as pd
import numpy as np

from collections import defaultdict

from breast_cancer_dataset.base import GeneralDataBase
from preprocessing.image_processing import crop_image_pipeline
from preprocessing.mask_generator import get_mias_roi_mask
from utils.config import MIAS_DB_PATH, MIAS_CONVERTED_DATA_PATH, MIAS_PREPROCESSED_DATA_PATH, MIAS_CASE_DESC, \
    CROP_CONFIG, CROP_PARAMS
from utils.functions import get_path


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
        print(f'{"=" * 70}\n\tGetting information from database {self.__name__} ({self.IMG_TYPE})\n{"=" * 70}')
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

        # Se procesa la columna RAD estableciendo un tamaño minimo de IMG_SHAPE / 2
        # df.loc[:, 'RAD'] = df.RAD.apply(lambda x: np.max([float(PATCH_SIZE / 2), float(x)]))
        df.loc[:, 'RAD'] = pd.to_numeric(df.RAD, downcast='integer', errors='coerce')

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

    def get_image_mask(self, func: callable = get_mias_roi_mask, args: List = None):

        args = [
            (x.CONVERTED_MASK, x.X_CORD, x.Y_CORD, x.RAD) for _, x in
            self.df_desc[~self.df_desc[['X_CORD', 'Y_CORD', 'RAD']].isna().any(axis=1)].
                groupby('CONVERTED_MASK', as_index=False).agg({'X_CORD': list, 'Y_CORD': list, 'RAD': list}).iterrows()
        ]
        super(DatasetMIAS, self).get_image_mask(func=func, args=args)

    def clean_dataframe(self):

        # Se descartarán aquellas imagenes completas que presenten más de una tipología. (por ejemplo, el seno presenta
        # una zona benigna y otra maligna).
        duplicated_tags = self.df_desc.groupby('ID').IMG_LABEL.nunique()
        print(f'\tExcluding {len(duplicated_tags[duplicated_tags > 1]) * 2} samples for ambiguous pathologys')
        self.df_desc.drop(
            index=self.df_desc[self.df_desc.ID.isin(duplicated_tags[duplicated_tags > 1].index.tolist())].index,
            inplace=True
        )


class DatasetMIASCrop(DatasetMIAS):

    IMG_TYPE: str = get_path('CROP', CROP_CONFIG, create=False)

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

        super(DatasetMIASCrop, self).preproces_images(args=args, func=func)
        super(DatasetMIASCrop, self).get_roi_imgs()

    def delete_observations(self):
        pass

    def clean_dataframe(self):
        pass