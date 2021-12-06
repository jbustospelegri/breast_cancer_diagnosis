
import os
import random
import tqdm

import pandas as pd

from typing import io, List
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from src.preprocessing.image_conversion import convert_img
from src.preprocessing.image_processing import full_image_pipeline
from src.utils.functions import search_files


class GeneralDataBase:

    __name__ = 'GeneralDataBase'
    IMG_TYPE: str = 'FULL'
    BINARY: bool = False
    DF_COLS = [
        'ID', 'DATASET', 'BREAST', 'BREAST_VIEW', 'BREAST_DENSITY', 'IMG_TYPE', 'RAW_IMG', 'CONVERTED_IMG',
        'PREPROCESSED_IMG', 'IMG_LABEL'
    ]
    df_desc = pd.DataFrame(columns=DF_COLS, index=[0])

    def __init__(self, ori_dir: io, ori_extension: str, dest_extension: str, converted_dir: io, procesed_dir: io,
                 database_info_file_paths: List[io]):

        for p in database_info_file_paths:
            assert os.path.exists(p), f"Directory {p} doesn't exists."

        assert os.path.exists(ori_dir), f"Directory {ori_dir} doesn't exists."

        self.ori_extension = ori_extension
        self.dest_extension = dest_extension
        self.ori_dir = ori_dir
        self.conversion_dir = converted_dir
        self.procesed_dir = procesed_dir
        self.database_info_file_paths = database_info_file_paths

    def get_df_from_info_files(self) -> pd.DataFrame:
       pass

    def add_extra_columns(self, df: pd.DataFrame):
        pass

    def add_dataset_columns(self, df: pd.DataFrame):
        # Se crea una columna con información acerca de qué dataset se trata.
        df.loc[:, 'DATASET'] = self.__name__

        # Se crea la columna IMG_TYPE que indicará si se trata de una imagen completa (FULL) o bien de una imagen
        # recortada (CROP) o mascara (MASK). En este caso, todas las imagenes son FULL
        df.loc[:, 'IMG_TYPE'] = self.IMG_TYPE

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Función que creará un dataframe con información del dataset CBIS-DDSM. Para cada imagen, se pretende recuperar
        la tipología (clase) detectada, el path de origen de la imagen y su nombre, el tipo de patología (massa o
        calcificación) y el nombre del estudio que contiene el identificador del paciente, el seno sobre el cual se ha
        realizado la mamografía, el tipo de mamografía (CC o MLO) y un índice para aquellas imagenes recortadas o
        mascaras. Adicionalmente, se indicarán los paths de destino para la conversión y el procesado de las imagenes.

        :return: Pandas dataframe con las columnas especificadas en la descripción
        """
        return pd.DataFrame(index=[0], columns=[*self.DF_COLS, 'TRAIN_VAL'])

    def clean_dataframe(self):

        # Se descartarán aquellas imagenes completas que presenten más de una tipología. (por ejemplo, el seno presenta
        # una zona benigna y otra maligna).
        duplicated_tags = self.df_desc.groupby('ID').IMG_LABEL.nunique()
        print(f'\tExcluding {len(duplicated_tags[duplicated_tags > 1])} images for ambiguous pathologys')
        self.df_desc.drop(
            index=self.df_desc[self.df_desc.ID.isin(duplicated_tags[duplicated_tags > 1].index.tolist())].index,
            inplace=True
        )

        print(f'\tExcluding {len(self.df_desc[self.df_desc.ID.duplicated()])} samples duplicated pathologys')
        self.df_desc.drop(index=self.df_desc[self.df_desc.ID.duplicated()].index, inplace=True)

    def convert_images_format(self) -> None:
        """
        Función para convertir las imagenes del formato de origen al formato de destino.
        """

        print(f'{"-" * 75}\n\tStarting conversion of file format: {len(self.df_desc)} {self.ori_extension} files.')

        # Se crea el iterador con los argumentos necesarios para realizar la función a través de un multiproceso.
        arg_iter = [(row.RAW_IMG, row.CONVERTED_IMG, self.BINARY) for _, row in self.df_desc.iterrows()]

        # Se crea un pool de multihilos para realizar la tarea de conversión de forma paralelizada.
        with Pool(processes=cpu_count() - 2) as pool:
            results = tqdm(pool.imap(convert_img, arg_iter), total=len(arg_iter), desc='conversion to png')
            tuple(results)

        # Se recuperan las imagenes modificadas y se crea un dataframe
        converted_imgs = pd.DataFrame(
            data=search_files(file=f'{self.conversion_dir}{os.sep}**{os.sep}{self.IMG_TYPE}',
                              ext=self.dest_extension),
            columns=['CONVERTED_IMG']
        )
        print(f"\tConverted {len(converted_imgs)} images to {self.dest_extension} format.\n{'-' * 75}")

    def preproces_images(self) -> None:
        """
        Función utilizara para realizar el preprocesado de las imagenes completas.

        :param show_example: booleano para almacenar 5 ejemplos aleatorios en la carpeta de resultados para la
        prueba realizada.
        """
        processed_imgs = pd.DataFrame(
            data=search_files(file=f'{self.conversion_dir}{os.sep}**{os.sep}{self.IMG_TYPE}', ext=self.dest_extension),
            columns=['CONVERTED_IMG']
        )
        print(f'{"-" * 75}\n\tStarting preprocessing of {len(processed_imgs)} images')

        args = [(row.CONVERTED_IMG, row.PREPROCESSED_IMG) for _, row in self.df_desc.iterrows()]

        with Pool(processes=cpu_count() - 2) as pool:
            results = tqdm(pool.imap(full_image_pipeline, args), total=len(args), desc='preprocessing full images')
            tuple(results)

        # Se recuperan las imagenes modificadas y se crea un dataframe
        proc_imgs = list(
            search_files(file=f'{self.procesed_dir}{os.sep}**{os.sep}{self.IMG_TYPE}', ext=self.dest_extension)
        )
        print(f'\tProcessed {len(proc_imgs)} images.')

    def delete_observations(self):
        proc_imgs = list(
            search_files(file=f'{self.procesed_dir}{os.sep}**{os.sep}{self.IMG_TYPE}', ext=self.dest_extension)
        )
        print(f'\tFailed processing of {len(self.df_desc[~self.df_desc.PREPROCESSED_IMG.isin(proc_imgs)])} images\n'
              f'{"-" * 75}')
        self.df_desc.drop(index=self.df_desc[~self.df_desc.PREPROCESSED_IMG.isin(proc_imgs)].index, inplace=True)

    def start_pipeline(self):

        # Funciones para obtener el dataframe de los ficheros planos
        df = self.get_df_from_info_files()

        # Se suprimen los casos que no contienen ninguna patología
        print(f'\tExcluding {len(df[df.IMG_LABEL.isnull()].index.drop_duplicates())} samples without pathologies.')
        df.drop(index=df[df.IMG_LABEL.isnull()].index, inplace=True)

        # Se añaden columnas adicionales para completar las columnas del dataframe
        self.add_extra_columns(df)

        # Se suprimen los casos cuya patología no sea massas
        print(f'\tExcluding {len(df[df.ABNORMALITY_TYPE != "MASS"].index.drop_duplicates())} non mass pathologies.')
        df.drop(index=df[df.ABNORMALITY_TYPE != 'MASS'].index, inplace=True)

        # Se añaden columnas informativas sobre la base de datos utilizada
        self.add_dataset_columns(df)

        # Se preprocesan la columnas del dataframe
        self.df_desc = self.process_dataframe(df)

        # Se limpia el dataframe de posibles duplicidades y imagenes no convertidas
        self.clean_dataframe()

        # Se realiza la conversión de las imagenes
        self.convert_images_format()

        # Se preprocesan las imagenes.
        self.preproces_images()

        # Se eliminan las imagenes que no hayan podido ser procesadas
        self.delete_observations()
