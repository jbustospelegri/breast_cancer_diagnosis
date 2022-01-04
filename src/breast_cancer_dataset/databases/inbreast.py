from typing import List

import pandas as pd
import numpy as np

from breast_cancer_dataset.base import GeneralDataBase
from preprocessing.image_processing import crop_image_pipeline
from preprocessing.mask_generator import get_inbreast_roi_mask
from utils.config import (
    INBREAST_DB_PATH, INBREAST_CONVERTED_DATA_PATH, INBREAST_PREPROCESSED_DATA_PATH, INBREAST_CASE_DESC,
    CROP_CONFIG, CROP_PARAMS
)
from utils.functions import get_filename, get_path


class DatasetINBreast(GeneralDataBase):

    """
        Clase cuyo objetivo consiste en preprocesar los datos de la base de datos de INBreast
    """

    __name__ = 'INBreast'

    def __init__(self):
        super().__init__(
            ori_dir=INBREAST_DB_PATH, ori_extension='dcm', dest_extension='png',
            converted_dir=INBREAST_CONVERTED_DATA_PATH, procesed_dir=INBREAST_PREPROCESSED_DATA_PATH,
            database_info_file_paths=[INBREAST_CASE_DESC]
        )

    def get_df_from_info_files(self) -> pd.DataFrame:
        """
        Función que creará un dataframe con información del dataset INBreast. Para cada imagen, se pretende recuperar
        la tipología (IMG_LABEL) detectada, el path de origen de la imagen y su nombre, la densidad y la vista del seno,
        así como si se trata del seno derecho o izquierdo.

        :return: pandas dataframe con la base de datos procesada.
        """

        # Se iteran los csv con información del set de datos para unificaros
        l = []
        print(f'{"=" * 70}\n\tGetting information from database {self.__name__} ({self.IMG_TYPE})\n{"=" * 70}')
        for path in self.database_info_file_paths:
            l.append(pd.read_excel(path, skipfooter=2))
        df = pd.concat(objs=l, ignore_index=True)

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGNA' y 'MALIGNA' en función de la puntición
        # asignada por BIRADS. Los codigos 2 se asignan a la clase benigna; los 4b, 4c, 5 y 6 a la clase maligna.
        # Se excluyen los códigos 0 (Incompletos), 3 (Probablemente benigno), 4a (Probablemente maligno (2-9%)).
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

        # Se crea la columna de ID a partir de la columna file name que contiene el nombre de la imagen
        df.loc[:, 'ID'] = df['File Name'].astype(str)

        # Se crea la columna FILE_NAME a partir de la columna FILE_NAME que contiene el nombre de la imagen.
        df.loc[:, 'FILE_NAME'] = df['File Name'].astype(str)

        return df

    def get_raw_files(self, df: pd.DataFrame, f: callable = lambda x: int(get_filename(x).split('_')[0])) \
            -> pd.DataFrame:
        """
        Función que une los paths de las imagenes del dataset con el dataframe utilizado por la clase.

        :param df: pandas dataframe con la información del dataset de la clase.
        :param f: función utilizada para obtener el campo identificador de cada imagen que permitirá unir cada
                     filepath con su correspondiente hilera en el dataframe.
        :return: Pandas dataframe con la columna que contiene los filepaths de las imagenes.
        """
        return super(DatasetINBreast, self).get_raw_files(df=df, get_id_func=f)

    def get_image_mask(self, func: callable = get_inbreast_roi_mask, args: List = None):
        """
        Función que genera las máscaras del set de datos y une los paths de las imagenes generadas con el dataframe del
        set de datos
        :param func: función para generar las máscaras
        :param args: parámetros a introducir en la función 'func' de los argumentos.
        """
        args = list(set([(x.CONVERTED_IMG, x.FILE_NAME, x.CONVERTED_MASK) for _, x in self.df_desc.iterrows()]))
        super(DatasetINBreast, self).get_image_mask(func=func, args=args)


class DatasetINBreastCrop(DatasetINBreast):

    """
        Clase cuyo objetivo consiste en preprocesar los datos de la base de datos INBreast y generar imagenes
        con las regiones de interes del dataset.
    """

    IMG_TYPE: str = get_path('CROP', CROP_CONFIG, create=False)

    def preproces_images(self, args: list = None, func: callable = crop_image_pipeline) -> None:
        """
        Función utilizada para realizar el preprocesado de las imagenes recortadas.

        :param: func: función utilizada para generar los rois del set de datos
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
        super(DatasetINBreastCrop, self).preproces_images(args=args, func=func)

        # Se llama a la función que une el path de cada roi con cada imagen original del dataset y con su
        # correspondiente hilera del dataset
        super(DatasetINBreastCrop, self).get_roi_imgs()

    def delete_observations(self) -> None:
        """
        Función para eliminar los registros del dataset al realizar las validaciones.
        """
        pass
