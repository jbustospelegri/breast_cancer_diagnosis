import os
from typing import List

import numpy as np
import pandas as pd


from breast_cancer_dataset.base import GeneralDataBase
from preprocessing.image_processing import crop_image_pipeline
from preprocessing.mask_generator import get_cbis_roi_mask
from utils.config import (
    CBIS_DDSM_DB_PATH, CBIS_DDSM_CONVERTED_DATA_PATH, CBIS_DDSM_PREPROCESSED_DATA_PATH, CBIS_DDSM_MASS_CASE_DESC_TRAIN,
    CBIS_DDSM_MASS_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TRAIN, CROP_CONFIG,
    CROP_PARAMS,
)
from utils.functions import get_dirname, get_path


class DatasetCBISDDSM(GeneralDataBase):

    """
        Clase cuyo objetivo consiste en preprocesar los datos de la base de datos CBISDDSM
    """

    __name__ = 'CBIS-DDSM'

    def __init__(self):
        super(DatasetCBISDDSM, self).__init__(
            ori_dir=CBIS_DDSM_DB_PATH, ori_extension='dcm', dest_extension='png',
            converted_dir=CBIS_DDSM_CONVERTED_DATA_PATH, procesed_dir=CBIS_DDSM_PREPROCESSED_DATA_PATH,
            database_info_file_paths=[CBIS_DDSM_MASS_CASE_DESC_TRAIN, CBIS_DDSM_MASS_CASE_DESC_TEST,
                                      CBIS_DDSM_CALC_CASE_DESC_TRAIN, CBIS_DDSM_CALC_CASE_DESC_TEST]
        )

    def get_df_from_info_files(self) -> pd.DataFrame:
        """
        Función que creará un dataframe con información del dataset CBIS-DDSM. Para cada imagen, se pretende recuperar
        la tipología (IMG_LABEL) detectada, el path de origen de la imagen y su nombre (nombre del estudio que contiene
        el identificador del paciente, el seno sobre el cual se ha realizado la mamografía y el tipo de mamografía).

        :return: pandas dataframe con la base de datos procesada.
        """

        # Se crea una lista que contendrá la información de los archivos csv del set de datos
        l = []

        # Se iteran los csv con información del set de datos para unificarlos
        print(f'{"=" * 70}\n\tGetting information from database {self.__name__} ({self.IMG_TYPE})\n{"=" * 70}')
        for path in self.database_info_file_paths:
            l.append(pd.read_csv(path))

        df = pd.concat(objs=l, ignore_index=True)

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGN' y 'MALIGNANT'. Se excluyen los casos de
        # patologias 'benign without callback'.
        df.loc[:, 'IMG_LABEL'] = df.pathology.where(df.pathology != 'BENIGN_WITHOUT_CALLBACK', np.nan)

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(df['abnormality type'] == 'calcification', 'CALC', 'MASS')

        # El campo que contiene información de la localización de las imagenes sin convertir es 'image file path'.
        # Se suprimen posibles duplicidades en este campo para realizar el procesado una única vez.
        df.drop_duplicates(subset=['image file path'], inplace=True)

        # Se obtiene si se trata del seno derecho o izquierdo
        df.loc[:, 'BREAST'] = df['left or right breast']

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO
        df.loc[:, 'BREAST_VIEW'] = df['image view']

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno
        df.loc[:, 'BREAST_DENSITY'] = df.breast_density

        # Se crea la columna filename a partir de la carpeta que contiene cada tipología. Esta columna
        # servirá para unir las imagenes originales con su información.
        df.loc[:, 'FILE_NAME'] = df['image file path'].apply(lambda x: get_dirname(x).split("/")[0])

        # Se crea una columna identificadora de cada registro.
        df.loc[:, 'ID'] = df['image file path'].apply(lambda x: get_dirname(x).split("/")[0])

        return df

    def get_raw_files(self, df: pd.DataFrame, f: callable = lambda x: get_dirname(x).split(os.sep)[-3]) -> pd.DataFrame:
        """
        Función que une los paths de las imagenes del dataset con el dataframe utilizado por la clase.

        :param df: pandas dataframe con la información del dataset de la clase.
        :param f: función utilizada para obtener el campo identificador de cada imagen que permitirá unir cada
                     filepath con su correspondiente hilera en el dataframe.
        :return: Pandas dataframe con la columna que contiene los filepaths de las imagenes.
        """
        return super(DatasetCBISDDSM, self).get_raw_files(df=df, get_id_func=f)

    def get_image_mask(self, func: callable = get_cbis_roi_mask, args: List = None) -> None:
        """
        Función que genera las máscaras del set de datos y une los paths de las imagenes generadas con el dataframe del
        set de datos
        :param func: función para generar las máscaras
        :param args: parámetros a introducir en la función 'func' de los argumentos.
        """
        super(DatasetCBISDDSM, self).get_image_mask(
            func=func, args=[(x.ID, x.CONVERTED_MASK) for _, x in self.df_desc.iterrows()]
        )


class DatasetCBISDDSMCrop(DatasetCBISDDSM):

    """
        Clase cuyo objetivo consiste en preprocesar los datos de la base de datos CBISDDSM y generar imagenes
        con las regiones de interes del dataset.
    """

    IMG_TYPE: str = get_path('CROP', CROP_CONFIG, create=False)

    def preproces_images(self, args: list = None, func: callable = crop_image_pipeline) -> None:
        """
        Función utilizada para realizar el preprocesado de las imagenes recortadas.

        :param func: función utilizada para generar los rois del set de datos
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
        super(DatasetCBISDDSMCrop, self).preproces_images(args=args, func=func)

        # Se llama a la función que une el path de cada roi con cada imagen original del dataset y con su
        # correspondiente hilera del dataset
        super(DatasetCBISDDSMCrop, self).get_roi_imgs()

    def delete_observations(self) -> None:
        """
        Función para eliminar los registros del dataset al realizar las validaciones.
        """
        pass
