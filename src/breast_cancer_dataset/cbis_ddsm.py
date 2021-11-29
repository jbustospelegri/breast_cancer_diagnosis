import os
import numpy as np
import pandas as pd

from typing import io, List
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from preprocessing import image_processing
from preprocessing.image_conversion import convert_img
from utils.config import (
    CBIS_DDSM_DB_PATH, CBIS_DDSM_CONVERTED_DATA_PATH, CBIS_DDSM_PREPROCESSED_DATA_PATH, CBIS_DDSM_MASS_CASE_DESC_TRAIN,
    CBIS_DDSM_MASS_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TRAIN, MODEL_FILES,
)
from utils.functions import get_dirname, search_files, get_path, get_filename


class DatasetCBISDDSM:

    __name__ = 'CBIS-DDSM'
    IMG_TYPE: str = 'FULL'

    def __init__(
            self,
            ori_dir: io = CBIS_DDSM_DB_PATH,
            ori_extension: str = 'dcm',
            dest_extension: str = 'png',
            converted_dir: io = CBIS_DDSM_CONVERTED_DATA_PATH,
            procesed_dir: io = CBIS_DDSM_PREPROCESSED_DATA_PATH,
            database_info_file_paths: List[io] = None
    ):

        if database_info_file_paths is None:
            database_info_file_paths = [CBIS_DDSM_MASS_CASE_DESC_TRAIN, CBIS_DDSM_MASS_CASE_DESC_TEST,
                                        CBIS_DDSM_CALC_CASE_DESC_TRAIN, CBIS_DDSM_CALC_CASE_DESC_TEST]
        else:
            for p in database_info_file_paths:
                assert os.path.exists(p), f"Directory {p} doesn't exists."

        assert os.path.exists(ori_dir), f"Directory {ori_dir} doesn't exists."

        self.ori_extension = ori_extension
        self.dest_extension = dest_extension
        self.ori_dir = ori_dir
        self.conversion_dir = converted_dir
        self.procesed_dir = procesed_dir
        self.database_info_file_paths = database_info_file_paths
        self.df_desc = self.get_dataframe()

    def get_dataframe(self) -> pd.DataFrame:
        """
        Función que creará un dataframe con información del dataset CBIS-DDSM. Para cada imagen, se pretende recuperar
        la tipología (clase) detectada, el path de origen de la imagen y su nombre, el tipo de patología (massa o
        calcificación) y el nombre del estudio que contiene el identificador del paciente, el seno sobre el cual se ha
        realizado la mamografía, el tipo de mamografía (CC o MLO) y un índice para aquellas imagenes recortadas o
        mascaras. Adicionalmente, se indicarán los paths de destino para la conversión y el procesado de las imagenes.

        :return: Pandas dataframe con las columnas especificadas en la descripción
        """

        # Se recupera la información de los csv de información
        df = self.__get_df_from_info_files()

        # Se obtienen las imagenes full de cada tipología.
        df.drop_duplicates(subset='image file path', inplace=True)

        # Se crea la columna ID para poder linkar la información del excel con la información de las imagenes
        # almacenadas en la carpeta RAW. En este caso, se utilizará la jerarquía de carpetas en la que se almacena
        # cada imagen
        df.loc[:, 'ID'] = df['image file path'].apply(lambda x: get_dirname(x))

        # Se descartarán aquellas imagenes completas que presenten más de una tipología. (por ejemplo, el seno presenta
        # una zona benigna y otra maligna).
        duplicated_tags = df.groupby(['image file path']).IMG_LABEL.nunique()
        print(f'\tExcluding {len(duplicated_tags[duplicated_tags > 1])} images for ambiguous pathologys')
        df.drop(index=df[df.ID.isin(duplicated_tags[duplicated_tags > 1].index.tolist())].index, inplace=True)

        print(f'\tExcluding {len(df[df.ID.duplicated()])} samples duplicated pathologys')
        df.drop(index=df[df.ID.duplicated()].index, inplace=True)

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

    def __get_df_from_info_files(self) -> pd.DataFrame:

        # Se crea una lista que contendrá la información de los archivos csv del set de datos
        l = []

        # Se iteran los csv con información del set de datos para unificaros
        print(f'{"-" * 70}\n\tGetting information from database {self.__name__} ({self.IMG_TYPE})\n{"-" * 70}')
        for path in self.database_info_file_paths:
            l.append(pd.read_csv(path))

        df = pd.concat(objs=l, ignore_index=True)

        # Se crea una columna con información acerca de qué dataset se trata.
        df.loc[:, 'DATASET'] = self.__name__

        # Se crea la columna IMG_TYPE que indicará si se trata de una imagen completa (FULL) o bien de una imagen
        # recortada (CROP) o mascara (MASK). En este caso, todas las imagenes son FULL
        df.loc[:, 'IMG_TYPE'] = self.IMG_TYPE

        # Se descartan aquellas imagenes que pertenecen a la clase Benign without callback debido a que tal vez no
        # contienen ningúna pat
        print(f'\tExcluding {len(df[df.pathology == "BENIGN_WITHOUT_CALLBACK"])} Benign without callback cases')
        df.drop(index=df[df.pathology == "BENIGN_WITHOUT_CALLBACK"].index, inplace=True)

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGN' y 'MALIGNANT'. Se excluyen los casos de
        # patologias 'benign without callback'.
        df.loc[:, 'IMG_LABEL'] = df.pathology

        # Se añaden las columnas adicionales para el dataframe
        self.__add_extra_columns(df)

        return df

    @staticmethod
    def __add_extra_columns(df: pd.DataFrame):
        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        df.loc[:, 'BREAST'] = df['left or right breast']

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO
        df.loc[:, 'BREAST_VIEW'] = df['image view']

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(df['abnormality type'] == 'calcification', 'CALC', 'MASS')

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno
        df.loc[:, 'BREAST_DENSITY'] = df.breast_density

    def convert_images_format(self) -> None:
        """
        Función para convertir las imagenes del formato de origen al formato de destino.
        """

        print(f'{"-" * 75}\n\tStarting conversion of file format: {len(self.df_desc.RAW_IMG.unique())} '
              f'{self.ori_extension} files finded.')

        # Se crea el iterador con los argumentos necesarios para realizar la función a través de un multiproceso.
        arg_iter = [(row.RAW_IMG, row.CONVERTED_IMG) for _, row in self.df_desc.iterrows()]

        # Se crea un pool de multihilos para realizar la tarea de conversión de forma paralelizada.
        with Pool(processes=cpu_count() - 2) as pool:
            results = tqdm(pool.imap(convert_img, arg_iter), total=len(arg_iter), desc='conversion to png')
            tuple(results)

        # Se recuperan las imagenes modificadas y se crea un dataframe
        converted_imgs = list(search_files(file=self.conversion_dir, ext=self.dest_extension))
        print(f"\tConverted {len(converted_imgs)} images to {self.dest_extension} format.\n{'-' * 75}")

    def preproces_images(self, show_example: bool = False) -> None:
        """
        Función utilizara para realizar el preprocesado de las imagenes completas.

        :param show_example: booleano para almacenar 5 ejemplos aleatorios en la carpeta de resultados para la
        prueba realizada.
        """
        full_img_df = self.df_desc.assign(example_dir=None).copy()

        if show_example:
            photos = np.random.sample(full_img_df.index.tolist(), 5)
            full_img_df.loc[photos, 'example_dir'] = full_img_df.loc[photos, :].apply(
                lambda x: get_path(MODEL_FILES.model_viz_preprocesing_dir, x.DATASET, get_filename(x.PREPROCESSED_IMG)),
                axis=1
            )

        conv_imgs = list(search_files(file=self.conversion_dir, ext=self.dest_extension))
        print(f'{"-" * 75}\n\tStarting preprocessing of {len(conv_imgs)} images')

        args = [(row.CONVERTED_IMG, row.PREPROCESSED_IMG, row.example_dir) for _, row in full_img_df.iterrows()]

        with Pool(processes=cpu_count() - 2) as pool:
            results = tqdm(pool.imap(image_processing, args), total=len(args), desc='preprocessing full images')
            tuple(results)

        # Se recuperan las imagenes modificadas y se crea un dataframe
        proc_imgs = list(search_files(file=self.procesed_dir, ext=self.dest_extension))
        print(f'\tProcessed {len(proc_imgs)} images.\n{"-" * 75}')

    def start_pipeline(self):
        self.convert_images_format()
        self.preproces_images()


class DatasetCBISDDSMCrop(DatasetCBISDDSM):

    IMG_TYPE = 'CROP'

    def get_dataframe(self) -> pd.DataFrame:

        # Se recupera la información de los csv de información
        df = self.__get_df_from_info_files()

        return df
