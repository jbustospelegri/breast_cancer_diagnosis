from typing import List

import pandas as pd
import numpy as np

from collections import defaultdict

from breast_cancer_dataset.base import GeneralDataBase
from preprocessing.image_processing import crop_image_pipeline
from preprocessing.mask_generator import get_mias_roi_mask
from utils.config import (
    MIAS_DB_PATH, MIAS_CONVERTED_DATA_PATH, MIAS_PREPROCESSED_DATA_PATH, MIAS_CASE_DESC, CROP_CONFIG, CROP_PARAMS
)
from utils.functions import get_path


class DatasetMIAS(GeneralDataBase):

    """
        Clase cuyo objetivo consiste en preprocesar los datos de la base de datos MIAS
    """

    __name__ = 'MIAS'

    def __init__(self):
        super(DatasetMIAS, self).__init__(
            ori_dir=MIAS_DB_PATH, ori_extension='pgm', dest_extension='png', converted_dir=MIAS_CONVERTED_DATA_PATH,
            procesed_dir=MIAS_PREPROCESSED_DATA_PATH, database_info_file_paths=[MIAS_CASE_DESC]
        )

    def get_df_from_info_files(self) -> pd.DataFrame:
        """
        Función que creará un dataframe con información del dataset MIAS. Para cada imagen, se pretende recuperar
        la tipología (IMG_LABEL) detectada, el path de origen de la imagen y su nombre (nombre del estudio que contiene
        el identificador del paciente, el seno sobre el cual se ha realizado la mamografía y el tipo de mamografía).

        :return: pandas dataframe con la base de datos procesada.
        """

        l = []
        # Se obtiene la información del fichero de texto descriptivo del dataset
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

        # Se procesa la columna RAD para que sea del tipo numérico
        df.loc[:, 'RAD'] = pd.to_numeric(df.RAD, downcast='integer', errors='coerce')

        # Debido a que el sistema de coordenadas se centra en el borde inferior izquierdo se deben de modificar las
        # coordenadas Y para adecuarlas al sistema de coordenadas centrado en el borde superior izquerdo.
        df.loc[:, 'Y_CORD'] = 1024 - pd.to_numeric(df.Y_CORD, downcast='integer', errors='coerce')

        # Se corrige el formato de las coordenadas X
        df.loc[:, 'X_CORD'] = pd.to_numeric(df.X_CORD, downcast='integer', errors='coerce')

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(
            df.ABNORMALITY_TYPE.isin(['CIRC', 'SPIC', 'MISC', 'ASYM']), 'MASS', None
        )

        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        # En este caso, no se dispone de dicha información por lo que se dejará vacía.
        df.loc[:, 'BREAST'] = None

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO. No se dispone de esta
        # información, por lo que e dejará vacía.
        df.loc[:, 'BREAST_VIEW'] = None

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno. Para ello se mapean los valores:
        # - 'F' (Fatty): 1
        # - 'G' (Fatty-Glandular): 2
        # - 'D' (Dense-Glandular): 3
        df.loc[:, 'BREAST_DENSITY'] = df.BREAST_TISSUE.map(defaultdict(lambda: None, {'F': '1', 'G': '2', 'D': '3'}))

        # Se crea la columna de ID a partir del nombre de las imagenes.
        df.loc[:, 'ID'] = df['FILE_NAME']

        return df

    def get_image_mask(self, func: callable = get_mias_roi_mask, args: List = None) -> None:
        """
        Función que genera las máscaras del set de datos y une los paths de las imagenes generadas con el dataframe del
        set de datos
        :param func: función para generar las máscaras
        :param args: parámetros a introducir en la función 'func' de los argumentos.
        """
        args = [
            (x.CONVERTED_MASK, x.X_CORD, x.Y_CORD, x.RAD) for _, x in
            self.df_desc[~self.df_desc[['X_CORD', 'Y_CORD', 'RAD']].isna().any(axis=1)].
                groupby('CONVERTED_MASK', as_index=False).agg({'X_CORD': list, 'Y_CORD': list, 'RAD': list}).iterrows()
        ]
        super(DatasetMIAS, self).get_image_mask(func=func, args=args)


class DatasetMIASCrop(DatasetMIAS):

    """
        Clase cuyo objetivo consiste en preprocesar los datos de la base de datos MIAS y generar imagenes
        con las regiones de interes del dataset.
    """

    IMG_TYPE: str = get_path('CROP', CROP_CONFIG, create=False)

    def preproces_images(self, args: list = None, func: callable = crop_image_pipeline) -> None:
        """
        Función utilizada para realizar el preprocesado de las imagenes recortadas.

        :param func: función utilizada para generar los rois del set de datos.
        :param args: lista de argumentos a introducir en la función 'func'.
        """

        # Se recupera la configuración para realizar el preprocesado de las imagenes.
        p = CROP_PARAMS[CROP_CONFIG]

        # Se crea la lista de argumentos para la función func. En concreto se recupera el path de origen
        # (imagen completa convertida), el path de destino, el path de la máscara, el número de muestras de
        # background, el numero de rois a obtener, el overlap de los rois y el margen de cada roi (zona de roi +
        # background).
        args = list(set([
            (x.CONVERTED_IMG, x.PROCESSED_IMG, x.CONVERTED_MASK, p['N_BACKGROUND'], p['N_ROI'], p['OVERLAP'],
             p['MARGIN']) for _, x in self.df_desc.iterrows()
        ]))

        # Se llama a la función que procesa las imagenes para obtener los rois
        super(DatasetMIASCrop, self).preproces_images(args=args, func=func)

        # Se llama a la función que une el path de cada roi con cada imagen original del dataset y con su
        # correspondiente hilera del dataset
        super(DatasetMIASCrop, self).get_roi_imgs()

    def delete_observations(self) -> None:
        """
        Función para eliminar los registros del dataset al realizar las validaciones.
        """
        pass

    def clean_dataframe(self) -> None:
        """
        Debido a que esta implementación es a nivel de rois, se debe de suprimir la funcionalidad de eliminar para una
        misma imagen, patologías duplicadas.
        """
        pass
