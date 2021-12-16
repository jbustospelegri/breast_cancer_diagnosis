import os
import warnings

import tqdm
import cv2

import pandas as pd
import numpy as np

from typing import io, List
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
from albumentations import Compose

from src.preprocessing.image_conversion import convert_img
from src.preprocessing.image_processing import full_image_pipeline
from src.utils.functions import search_files, get_filename, get_path


class GeneralDataBase:

    __name__ = 'GeneralDataBase'
    IMG_TYPE: str = 'FULL'
    DF_COLS = [
        'ID', 'DATASET', 'BREAST', 'BREAST_VIEW', 'BREAST_DENSITY', 'IMG_TYPE', 'FILE_NAME', 'RAW_IMG', 'CONVERTED_IMG',
        'PROCESSED_IMG', 'CONVERTED_MASK', 'PROCESSED_MASK', 'IMG_LABEL'
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

    def add_dataset_columns(self, df: pd.DataFrame):
        # Se crea una columna con información acerca de qué dataset se trata.
        df.loc[:, 'DATASET'] = self.__name__

        # Se crea la columna IMG_TYPE que indicará si se trata de una imagen completa (FULL) o bien de una imagen
        # recortada (CROP) o mascara (MASK). En este caso, todas las imagenes son FULL
        df.loc[:, 'IMG_TYPE'] = self.IMG_TYPE

    def add_extra_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_raw_files(self, df: pd.DataFrame, get_id_func: callable = lambda x: get_filename(x)) -> pd.DataFrame:
        """
        Función que creará un dataframe con información del dataset CBIS-DDSM. Para cada imagen, se pretende recuperar
        la tipología (clase) detectada, el path de origen de la imagen y su nombre, el tipo de patología (massa o
        calcificación) y el nombre del estudio que contiene el identificador del paciente, el seno sobre el cual se ha
        realizado la mamografía, el tipo de mamografía (CC o MLO) y un índice para aquellas imagenes recortadas o
        mascaras. Adicionalmente, se indicarán los paths de destino para la conversión y el procesado de las imagenes.

        :return: Pandas dataframe con las columnas especificadas en la descripción
        """
        # Se recuperan los paths de las imagenes almacenadas con el formato específico (por defecto dcm) en la carpeta
        # de origen (por defecto INBREAST_DB_PATH)
        db_files_df = pd.DataFrame(data=search_files(self.ori_dir, self.ori_extension), columns=['RAW_IMG'])

        # Se procesa la columna ori path para poder lincar cada path con los datos del excel. Para ello, se separa
        # los nombres de cara archivo a partir del símbolo _ y se obtiene la primera posición.
        db_files_df.loc[:, 'FILE_NAME'] = db_files_df.RAW_IMG.apply(get_id_func).astype(str)

        # Se crea la columna RAW_IMG con el path de la imagen original
        df_def = pd.merge(left=df, right=db_files_df, on='FILE_NAME', how='left')

        return df_def

    def add_img_files(self, df: pd.DataFrame) -> pd.DataFrame:

        # Se crea la clumna PROCESSED_IMG en la que se volcarán las imagenes preprocesadas
        df.loc[:, 'PROCESSED_IMG'] = df.apply(
            lambda x: get_path(self.procesed_dir, x.IMG_TYPE, f'{x.ID}.{self.dest_extension}'), axis=1
        )

        # Se crea la clumna CONVERTED_IMG en la que se volcarán las imagenes convertidas de formato
        df.loc[:, 'CONVERTED_IMG'] = df.apply(
            lambda x: get_path(self.conversion_dir, 'FULL', f'{x.FILE_NAME}.{self.dest_extension}'), axis=1
        )

        return df

    def add_mask_files(self, df: pd.DataFrame):
        df.loc[:, 'CONVERTED_MASK'] = df.apply(
            lambda x: get_path(self.conversion_dir, f'MASK', f'{x.ID}.png'), axis=1
        )
        df.loc[:, 'PROCESSED_MASK'] = df.apply(
            lambda x: get_path(self.procesed_dir, f'MASK', f'{x.FILE_NAME}.png'), axis=1
        )

    def clean_dataframe(self):

        # Se descartarán aquellas imagenes completas que presenten más de una tipología. (por ejemplo, el seno presenta
        # una zona benigna y otra maligna).
        duplicated_tags = self.df_desc.groupby('ID').IMG_LABEL.nunique()
        print(f'\tExcluding {len(duplicated_tags[duplicated_tags > 1]) * 2} samples for ambiguous pathologys')
        self.df_desc.drop(
            index=self.df_desc[self.df_desc.ID.isin(duplicated_tags[duplicated_tags > 1].index.tolist())].index,
            inplace=True
        )

        print(f'\tExcluding {len(self.df_desc[self.df_desc.ID.duplicated()])} samples duplicated pathologys')
        self.df_desc.drop(index=self.df_desc[self.df_desc.ID.duplicated()].index, inplace=True)

    def get_image_mask(self, func: callable = None, args: List = None):

        print(f'{"-" * 75}\n\tGetting masks: {self.df_desc.CONVERTED_MASK.nunique()} existing files.')
        # Se crea un pool de multihilos para realizar la tarea de conversión de forma paralelizada.
        with ThreadPool(processes=cpu_count() - 2) as pool:
            results = tqdm(pool.imap(func, args), total=len(args), desc='getting mask files')
            tuple(results)
        # Se recuperan las imagenes modificadas y se crea un dataframe
        completed = list(search_files(
            file=f'{self.conversion_dir}{os.sep}**{os.sep}MASK', ext='png', in_subdirs=False))
        print(f"\tGenerated {len(completed)} masks.\n{'-' * 75}")

    def convert_images(self, func: callable = convert_img, args: List = None) -> None:
        """
        Función para convertir las imagenes del formato de origen al formato de destino.
        """

        print(f'{"-" * 75}\n\tStarting conversion: {self.df_desc.CONVERTED_IMG.nunique()} {self.ori_extension} files.')

        # Se crea el iterador con los argumentos necesarios para realizar la función a través de un multiproceso.
        if args is None:
            args = list(set([(row.RAW_IMG, row.CONVERTED_IMG) for _, row in self.df_desc.iterrows()]))

        # Se crea un pool de multihilos para realizar la tarea de conversión de forma paralelizada.
        with ThreadPool(processes=cpu_count() - 2) as pool:
            results = tqdm(pool.imap(func, args), total=len(args), desc='converting images')
            tuple(results)

        # Se recuperan las imagenes modificadas y se crea un dataframe
        completed = list(search_files(
            file=f'{self.conversion_dir}{os.sep}**{os.sep}FULL', ext=self.dest_extension, in_subdirs=False))
        print(f"\tConverted {len(completed)} images to {self.dest_extension} format.\n{'-' * 75}")

    def preproces_images(self, args: list = None, func: callable = full_image_pipeline) -> None:
        """
        Función utilizara para realizar el preprocesado de las imagenes completas.

        """
        print(f'{"-" * 75}\n\tStarting preprocessing of {self.df_desc.PROCESSED_IMG.nunique()} images')

        if args is None:
            args = list(set([
                (x.CONVERTED_IMG, x.PROCESSED_IMG, False, x.PROCESSED_MASK, x.CONVERTED_MASK) for _, x in
                self.df_desc.iterrows()
            ]))

        with ThreadPool(processes=cpu_count() - 2) as pool:
            results = tqdm(pool.imap(func, args), total=len(args), desc='preprocessing full images')
            tuple(results)

        # Se recuperan las imagenes modificadas y se crea un dataframe
        completed = list(search_files(
            file=f'{self.procesed_dir}{os.sep}**{os.sep}{self.IMG_TYPE}', ext=self.dest_extension, in_subdirs=False))
        print(f'\tProcessed {len(completed)} images.\n{"-" * 75}')

    def get_roi_imgs(self):
        # Se debe de modificar el dataframe en función de los crops obtenidos
        croped_imgs = pd.DataFrame(
            data=search_files(get_path(self.procesed_dir, self.IMG_TYPE), ext=self.dest_extension, in_subdirs=False),
            columns=['FILE']
        )
        croped_imgs.loc[:, 'FILE_NAME'] = croped_imgs.FILE.apply(lambda x: "_".join(get_filename(x).split('_')[1:-1]))
        croped_imgs.loc[:, 'N_CROP'] = croped_imgs.FILE.apply(lambda x: get_filename(x).split('_')[-1])
        croped_imgs.loc[:, 'LABEL_BACKGROUND_CROP'] = croped_imgs.FILE.apply(lambda x: get_filename(x).split('_')[0])

        self.df_desc = pd.merge(left=self.df_desc, right=croped_imgs, on=['FILE_NAME'], how='left')

        # Se suprimen los casos que no contienen ningún recorte
        print(f'\tDeleting {len(self.df_desc[self.df_desc.FILE.isnull()])} samples without cropped regions')
        self.df_desc.drop(index=self.df_desc[self.df_desc.FILE.isnull()].index, inplace=True)

        # Se modifica el identificador único de la imagen
        self.df_desc.loc[:, 'ID'] = self.df_desc.ID + '_' + self.df_desc.N_CROP

        # Se modifica la columna PROCESSED_IMG
        self.df_desc.loc[:, 'PROCESSED_IMG'] = self.df_desc.FILE

        # Se modifica la patología dependiendo de si es background o no
        self.df_desc.loc[:, 'IMG_LABEL'] = self.df_desc.IMG_LABEL.where(
            self.df_desc.LABEL_BACKGROUND_CROP == 'roi', 'BACKGROUND'
        )

    def delete_observations(self):
        proc_imgs = list(search_files(get_path(self.procesed_dir, 'FULL'), ext=self.dest_extension, in_subdirs=False))
        print(f'\tFailed processing of {len(self.df_desc[~self.df_desc.PROCESSED_IMG.isin(proc_imgs)])} images\n'
              f'{"-" * 75}')
        self.df_desc.drop(index=self.df_desc[~self.df_desc.PROCESSED_IMG.isin(proc_imgs)].index, inplace=True)

        proc_mask = list(search_files(get_path(self.procesed_dir, 'MASK'), ext=self.dest_extension, in_subdirs=False))
        print(f'\tFailed processing of {len(self.df_desc[~self.df_desc.PROCESSED_MASK.isin(proc_mask)])} masks\n'
              f'{"-" * 75}')
        self.df_desc.drop(index=self.df_desc[~self.df_desc.PROCESSED_MASK.isin(proc_mask)].index, inplace=True)

    def start_pipeline(self):

        # Funciones para obtener el dataframe de los ficheros planos
        df = self.get_df_from_info_files()

        # Se suprimen los casos que no contienen ninguna patología
        print(f'\tExcluding {len(df[df.IMG_LABEL.isnull()].index.drop_duplicates())} samples without pathologies.')
        df.drop(index=df[df.IMG_LABEL.isnull()].index, inplace=True)

        # Se suprimen los casos cuya patología no sea massas
        print(f'\tExcluding {len(df[df.ABNORMALITY_TYPE != "MASS"].index.drop_duplicates())} non mass pathologies.')
        df.drop(index=df[df.ABNORMALITY_TYPE != 'MASS'].index, inplace=True)

        # Se añaden columnas informativas sobre la base de datos utilizada
        self.add_dataset_columns(df)

        # Se realiza la búsqueda de las imagenes crudas
        df_with_raw_img = self.get_raw_files(df)

        # Se añaden columnas adicionales para completar las columnas del dataframe
        df_def = self.add_extra_columns(df_with_raw_img)

        # Se añaden las columnas con el destino de las máscaras.
        self.add_mask_files(df_def)

        # Se añaden las columnas con el destino de las imagenes
        # Se preprocesan la columnas del dataframe
        self.df_desc = self.add_img_files(df_def)

        # Se limpia el dataframe de posibles duplicidades
        self.clean_dataframe()

        # Se realiza la conversión de las imagenes
        self.convert_images()

        # Se obtienen las mascaras
        self.get_image_mask()

        # Se preprocesan las imagenes.
        self.preproces_images()

        # Se eliminan las imagenes que no hayan podido ser procesadas y se obtiene un único registro a nivel de label
        # e imagen.
        self.delete_observations()


class SegmentationDataset:

    def __init__(self, df: pd.DataFrame, img_col: str, mask_col: str, augmentation: Compose = None):

        assert img_col in df.columns, f'{img_col} not in df'
        assert mask_col in df.columns, f'{mask_col} not in df'

        self.df = self._filter_valid_filepaths(df.copy(), img_col)
        self.img_col = img_col
        self.mask_col = mask_col
        self.augmentation = augmentation

    def __getitem__(self, i):

        # read data
        image = cv2.cvtColor(cv2.imread(self.df.iloc[i][self.img_col]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.df.iloc[i][self.mask_col], cv2.IMREAD_GRAYSCALE)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.df)

    def _filter_valid_filepaths(self, df, x_col):
        """Keep only dataframe rows with valid filenames

        # Arguments
            df: Pandas dataframe containing filenames in a column
            x_col: string, column in `df` that contains the filenames or filepaths

        # Returns
            absolute paths to image files
        """

        mask = df[x_col].apply(lambda x: os.path.isfile(x))
        n_invalid = (~mask).sum()
        if n_invalid:
            warnings.warn(
                'Found {} invalid image filename(s) in x_col="{}". '
                'These filename(s) will be ignored.'
                .format(n_invalid, x_col)
            )
        return df[mask]


class ClassificationDataset:

    def __init__(self, df: pd.DataFrame, x_col: str, y_col: str = None, augmentation: Compose = None,
                 class_mode: str = 'categorical', classes: list = None):

        assert x_col in df.columns, f'{x_col} not in df'

        self.df = self._filter_valid_filepaths(df.copy(), x_col)
        self.x_col = x_col
        self.augmentation = augmentation
        self.filenames = self.df[x_col].values.tolist()

        if class_mode not in ['categorical', 'binary', 'sparse']:
            raise NotImplementedError(
                f'{class_mode} not implemented. Available class_mode params: categorical, binary, sparse'
            )

        if y_col:
            assert y_col in df.columns, f'{y_col} not in df'

            if (class_mode == 'binary') & (df[y_col].nunique() > 2):
                raise ValueError(f'Incorrect number of classes for binary mode')

            if classes is None:
                classes = sorted(df[y_col].unique().tolist())

            self.y_col = y_col
            self.class_indices = {lbl: indx for indx, lbl in enumerate(classes)}
            self.classes = self.df[y_col].values.tolist()
            self.class_list = classes
            self.class_mode = class_mode

            self.get_class()
        else:
            self.y_col = None
            self.class_indices = None
            self.classes = None
            self.class_list = None
            self.class_mode = None

    def get_class(self):
        if self.class_mode == 'binary' or self.class_mode == 'sparse':
            self.df.loc[:, self.y_col] = self.df[self.y_col].map(self.class_indices).astype(np.float32)
        elif self.class_mode == 'categorical':
            self.df.loc[:, self.y_col] = \
                pd.get_dummies(self.df, columns=[self.y_col], dtype=np.float32, prefix='dummy_') \
                    [[f'dummy__{c}' for c in self.class_list]].apply(lambda x: list(x), axis=1)

    def __getitem__(self, i):

        # read data
        image = cv2.cvtColor(cv2.imread(self.df.iloc[i][self.x_col]), cv2.COLOR_BGR2RGB)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        if self.y_col:
            return image, np.float32(self.df.iloc[i][self.y_col])
        else:
            return image

    def __len__(self):
        return len(self.df)

    def _filter_valid_filepaths(self, df, x_col):
        """Keep only dataframe rows with valid filenames

        # Arguments
            df: Pandas dataframe containing filenames in a column
            x_col: string, column in `df` that contains the filenames or filepaths

        # Returns
            absolute paths to image files
        """

        mask = df[x_col].apply(lambda x: os.path.isfile(x))
        n_invalid = (~mask).sum()
        if n_invalid:
            warnings.warn(
                'Found {} invalid image filename(s) in x_col="{}". '
                'These filename(s) will be ignored.'
                .format(n_invalid, x_col)
            )
        return df[mask]


class Dataloder(Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, seed=0):
        self.dataset = dataset
        self.filenames = dataset.filenames
        self.class_indices = dataset.class_indices
        self.indexes = np.arange(len(dataset))

        if dataset.classes:
            self.classes = dataset.classes

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        data = []
        for j in range(i * self.batch_size, (i + 1) * self.batch_size):
            data.append(self.dataset[self.indexes[j]])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.RandomState(seed=self.seed).permutation(self.indexes)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
