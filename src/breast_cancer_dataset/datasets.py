import os

import cv2
import pandas as pd
import numpy as np

from glob import glob
from multiprocessing import Pool, cpu_count
from typing import io
from tqdm import tqdm

from utils.config import (
    CBIS_DDSM_DB_PATH, CBIS_DDSM_CALC_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TRAIN, CBIS_DDSM_MASS_CASE_DESC_TEST,
    CBIS_DDSM_MASS_CASE_DESC_TRAIN, CBIS_DDSM_CONVERTED_DATA_PATH, CBIS_DDSM_PROCESSED_DATA_PATH,
    MIAS_DB_PATH, MIAS_CASE_DESC, MIAS_CONVERTED_DATA_PATH, MIAS_PROCESSED_DATA_PATH,
    INBREAST_DB_PATH, INBREAST_CASE_DESC, INBREAST_CONVERTED_DATA_PATH, INBREAST_PROCESSED_DATA_PATH
)
from utils.image_processing import (
    convert_dcm_imgs, crop_borders, min_max_normalize, binarize_img, edit_mask,  remove_artifacts
)


class DatasetCBISDDSM:

    def __init__(self, ori_dir: io = CBIS_DDSM_DB_PATH, ori_extension: str = 'dcm', dest_extension: str = 'png',
                 converted_dir: io = CBIS_DDSM_CONVERTED_DATA_PATH, procesed_dir: io = CBIS_DDSM_PROCESSED_DATA_PATH):

        assert os.path.exists(ori_dir), f"Directory {ori_dir} doesn't exists."
        self.ori_extension = ori_extension
        self.dest_extension = dest_extension
        self.ori_dir = ori_dir
        self.conversion_dir = converted_dir
        self.procesed_dir = procesed_dir
        self.df_desc = self.get_desc_from_data()

    def get_desc_from_data(self) -> pd.DataFrame:
        """
        Función que creará un dataframe con información del dataset CBIS-DDSM. Para cada imagen, se pretende recuperar
        la tipología (clase) detectada, el path de origen de la imagen y su nombre, el tipo de patología (massa o
        calcificación) y el nombre del estudio que contiene el identificador del paciente, el seno sobre el cual se ha
        realizado la mamografía, el tipo de mamografía (CC o MLO) y un índice para aquellas imagenes recortadas o
        mascaras. Adicionalmente, se indicarán los paths de destino para la conversión y el procesado de las imagenes.

        :return: Pandas dataframe con las columnas especificadas en la descripción
        """

        # Se crea una lista que contendrá la información de los archivos csv del set de datos
        l = []

        # Se iteran los csv con información del set de datos para unificaros
        for path in [CBIS_DDSM_MASS_CASE_DESC_TRAIN, CBIS_DDSM_MASS_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TRAIN,
                     CBIS_DDSM_CALC_CASE_DESC_TEST]:
            l.append(pd.read_csv(path))
        df = pd.concat(objs=l, ignore_index=True)

        # Se obiene el nombre del studio realizado a partir del path especificado para la imagen. Este studio
        # se obtiene para las imagenes completas, recortadas y las mascaras y servirá para unir las etiquetas de clase
        # con las imagenes del dataset.
        df.loc[:, 'studio'] = df.apply(
            lambda x: list(set([os.path.dirname(x[f'{c} file path']) for c in ['image', 'cropped image', 'ROI mask']])),
            axis=1
        )

        # Se realiza el explode de los datos para obtener, para cada imagen su patología, el nombre del estudio y si se
        # trata de una calcificación o masas.
        df_desc = df[['patient_id', 'left or right breast', 'image view', 'studio', 'pathology', 'abnormality type']].\
            explode('studio', ignore_index=True)

        # Se eliminan posibles duplicidades.
        df_desc.drop_duplicates(inplace=True)

        # Se descartarán aquellas imagenes completas (identificadas por no terminar por _1/_2..) que presenten más de
        # una tipología.
        duplicated_tags = df_desc.groupby(['studio']).pathology.nunique()
        duplicated_tags.drop(index=duplicated_tags[duplicated_tags < 2].index, inplace=True)

        print(f'Detectados {len(duplicated_tags.index.drop_duplicates())} estudios con clases distintas.\n'
              f'Se descartarán dichos casos.')

        # Se eliminan estas imagenes.
        df_desc.drop(index=df_desc[df_desc.studio.isin(duplicated_tags.index.drop_duplicates())].index, inplace=True)

        # Se recuperan los paths de las imagenes almacenadas con el formato específico (por defecto dcm) en la carpeta
        # de origen (por defecto CBIS_DDSM_DB_PATH
        db_files_df = pd.DataFrame(
            data=glob(os.path.join(self.ori_dir, '**', f'*.{self.ori_extension}'), recursive=True),
            columns=['ori_path']
        )

        # Se obtiene el studio de la imagen a partir de la carpeta del estudio (antepenultima) que se guarda la imagen
        db_files_df.loc[:, 'studio'] = db_files_df.ori_path.apply(
            lambda x: '/'.join(os.path.dirname(x).split(os.sep)[-3:]))

        # Se une el directorio de cada imagen con la información extraída de los excels.
        db_files_with_label = pd.merge(left=db_files_df, right=df_desc, on='studio', how='left')

        # Se crea un id de imagen compuesto por el time series del estudio en el cual se almacena la imagen y el nombre
        # de la propia imagen.
        db_files_with_label.loc[:, 'img_id'] = db_files_with_label.apply(
            lambda x: "_".join([os.path.basename(x.studio), os.path.splitext(os.path.basename(x.ori_path))[0]]), axis=1
        )

        # Aquellas imagenes descartadas no contendrán ningún label, por lo que se descartarán.
        db_files_with_label.dropna(how='any', subset=['pathology'], inplace=True)

        # Se crea una columna para hacer una primera segregación entre imagen completa o recortada/mascara. Para ello
        # se analiza el nombre del estudio realizado en cada carpeta.
        db_files_with_label.loc[:, 'img_type'] = np.where(
            db_files_with_label.studio.str.split('/').str[0].str.split('_').str[-1].str.isdigit(), 'MASK/CROP', 'FULL'
        )

        # Se crea una columna con el directorio destino para la conversion.
        db_files_with_label.loc[:, 'conversion_dest_dir'] = db_files_with_label.pathology.apply(
            lambda x: os.path.join(self.conversion_dir, x))

        # Se crea una columna con el directorio destino para el preprocesado.
        db_files_with_label.loc[:, 'preprocessing_dest_dir'] = db_files_with_label.pathology.apply(
            lambda x: os.path.join(self.procesed_dir, x))

        return db_files_with_label

    def convert_images_format(self) -> None:
        """
        Función para convertir las imagenes del formato de origen al formato de destino.
        """

        print(f'{"-" * 50}\n\t{len(self.df_desc.ori_path.unique())} {self.ori_extension} files finded.\n{"-" * 50}')

        # Se crea el iterador con los argumentos necesarios para realizar la función a través de un multiproceso.
        # arg_iter = [(row.ori_path, row.conversion_dest_dir, row.img_id, self.dest_extension, row.img_type)
        #             for _, row in self.df_desc.iterrows()]
        #
        # # Se crea un pool de multihilos para realizar la tarea de conversión de forma paralelizada.
        # with Pool(processes=cpu_count() - 2) as pool:
        #     results = tqdm(
        #         pool.imap(convert_dcm_imgs, arg_iter), total=len(arg_iter), desc='CBIS-DDSM conversion dcm to png'
        #     )
        #     tuple(results)

        # Se recuperan las imagenes modificadas y se crea un dataframe
        converted_imgs = pd.DataFrame(
            data=list(glob(os.path.join(self.conversion_dir, '**', f'*.{self.dest_extension}'), recursive=True)),
            columns=[f'{self.dest_extension}_path']
        )
        print(f"Converted {len(converted_imgs)} images to {self.dest_extension} format")

        # Se recupera el nombre de la imagen para unirlo con el dataset original
        converted_imgs.loc[:, 'img_id'] = converted_imgs[f'{self.dest_extension}_path'].apply(
            lambda x: os.path.basename(os.path.splitext(x)[0]))

        # Se recuepra el tipo de fotografía (completa, recortada o mascara).
        converted_imgs.loc[:, 'img_type'] = converted_imgs[f'{self.dest_extension}_path'].apply(
            lambda x: os.path.basename(os.path.dirname(x)))

        # Se asigna el tipo de imagen (full, recortada, mascara) en función de la carpeta almacenada.
        self.df_desc.loc[:, 'img_type'] = self.df_desc.img_id.map(
            converted_imgs[['img_id', 'img_type']].set_index('img_id').to_dict()['img_type'])

        # Se escribe la información del dataset en un excel en la carpeta de destino de las imagenes convertidas.
        self.data_desc_to_excel()

    def data_desc_to_excel(self) -> None:
        """
        Función escribe el feedback del dataset en la carpeta de destino de la conversión.
        """

        self.df_desc.to_excel(os.path.join(self.conversion_dir, f'{self.__class__.__name__}.xlsx'), index=False)

    def preproces_full_imges(self, show_example: bool = False) -> None:
        """
        Función utilizara para realizar el preprocesado de las imagenes completas.
        """
        full_img_df = self.df_desc[self.df_desc.img_type == 'FULL'].copy()

        args = [
            (
                os.path.join(row.conversion_dest_dir, row.img_type, f'{row.img_id}.{self.dest_extension}'),
                os.path.join(row.preprocessing_dest_dir, row.img_type, f'{row.img_id}.{self.dest_extension}'),
                self.dest_extension, False
             ) for _, row in full_img_df.iterrows()
        ]

        with Pool(processes=cpu_count() - 2) as pool:
            results = tqdm(
                pool.imap(self.image_processing, args), total=len(args), desc='CBIS-DDSM full images preprocessing'
            )
            tuple(results)

    @staticmethod
    def image_processing(args) -> None:
        """
        Función utilizada para realizar el preprocesado de las mamografías. Este preprocesado consiste en:
            1 - Recortar los bordes de las imagenes.
            2 - Realziar una normalización min-max para estandarizar la luminosidad producida por distintos escáneres.
            3 - Quitar anotaciones realziadas sobre las iamgenes.
            4 - Relizar un flip horizontal para estandarizar la orientacion de los cenos.
            5 - Mejorar el contraste de las imagenes en blanco y negro  mediante CLAHE.
            6 - Recortar las imagenes para que queden cuadradas.
            7 - Normalización min-max para estandarizar el valor de los píxeles entre 0 y 255 previo a guardar
                las imagenes finales


        :param args: listado de argumentos cuya posición debe ser:
            1 - path de la imagen sin procesad.
            2 - path de destino de la imagen procesada.
            3 - booleano para almacenar cada step de preprocesado realizado
        """


        img_filepath: io = args[0]
        dest_dirpath: io = args[1]
        save_ext: io = args[2]
        save_process: bool = args[3]

        def save_img(img: np.ndarray, save_flag: bool, *args):

            # Se obtiene el nombre de la imagen
            img_name = os.path.basename(os.path.splitext(img_filepath)[0])

            if save_flag:
                cv2.imwrite(os.path.join(dest_dirpath, f'{img_name}{"_".join(args)}.{save_ext}'), img=img)

        # Se lee la imagen origianl sin procesar. Esta imagen se leerá en escala de grises
        img = cv2.imread(img_filepath)

        # Se recortan las imagenes.
        img_cropped = crop_borders(img=img, left=0.01, right=0.01, top=0.04, bottom=0.04)
        save_img(img_cropped, save_process, '_crop')

        # Se estandarizan las imagenes con la normalización min_max
        img_stand = min_max_normalize(img=img_cropped)
        save_img(img_stand, save_process, '_min_max_norm')

        # Se eliminan los artefactos
        img_without_artifacts = remove_artifacts(img=img_stand, bin_kwargs=dict(thresh=0.1, maxval=1),
                                                 mask_kwargs=dict(kernel_size=(23, 23)))
        save_img(img_without_artifacts, save_process, '_wihtout_artifacts')





class DatasetINBreast(DatasetCBISDDSM):

    def __init__(self):
        super().__init__(ori_dir=INBREAST_DB_PATH, converted_dir=INBREAST_CONVERTED_DATA_PATH,
                         procesed_dir=INBREAST_PROCESSED_DATA_PATH)

    def get_desc_from_data(self) -> pd.DataFrame:
        pass


class DatasetMIAS(DatasetCBISDDSM):

    def __init__(self):
        super().__init__(ori_dir=MIAS_DB_PATH, converted_dir=MIAS_CONVERTED_DATA_PATH, ori_extension='pgm',
                         procesed_dir=MIAS_PROCESSED_DATA_PATH)

    def get_desc_from_data(self) -> pd.DataFrame:
        pass
